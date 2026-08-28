# pixel_art.py
from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from textwrap import dedent
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import openai
from arcengine import FrameData, GameAction, GameState
from openai import OpenAI as OpenAIClient
from openai.types.chat import ChatCompletion
from PIL import Image

from ..agent import Agent

logger = logging.getLogger()

# 16-color palette (RGBA, hex -> tuple)
_PALETTE: List[tuple[int, int, int, int]] = [
    (0xFF, 0xFF, 0xFF, 0xFF),  # 0 White
    (0xCC, 0xCC, 0xCC, 0xFF),  # 1 Off-white
    (0x99, 0x99, 0x99, 0xFF),  # 2 Neutral light
    (0x66, 0x66, 0x66, 0xFF),  # 3 Neutral
    (0x33, 0x33, 0x33, 0xFF),  # 4 Off-black
    (0x00, 0x00, 0x00, 0xFF),  # 5 Black
    (0xE5, 0x3A, 0xA3, 0xFF),  # 6 Magenta
    (0xFF, 0x7B, 0xCC, 0xFF),  # 7 Magenta light
    (0xF9, 0x3C, 0x31, 0xFF),  # 8 Red
    (0x1E, 0x93, 0xFF, 0xFF),  # 9 Blue
    (0x88, 0xD8, 0xF1, 0xFF),  # 10 Blue light
    (0xFF, 0xDC, 0x00, 0xFF),  # 11 Yellow
    (0xFF, 0x85, 0x1B, 0xFF),  # 12 Orange
    (0x92, 0x12, 0x31, 0xFF),  # 13 Maroon
    (0x4F, 0xCC, 0x30, 0xFF),  # 14 Green
    (0xA3, 0x56, 0xD6, 0xFF),  # 15 Purple
]

_SCALE = 2  # 64 px -> 128 px
_TARGET_SIZE = 64 * _SCALE


def _validate_grid(grid: Sequence[Sequence[int]]) -> None:
    if len(grid) != 64 or any(len(row) != 64 for row in grid):
        raise ValueError("Grid must be 64×64.")
    if any(cell not in range(16) for row in grid for cell in row):
        raise ValueError("Grid values must be integers 0–15.")


def grid_to_image(grid: Sequence[Sequence[int]]) -> Image.Image:
    """
    Convert a 64×64 int grid to a 256×256 RGBA Pillow Image.
    """
    _validate_grid(grid)

    # Flatten grid into raw bytes (R, G, B, A per pixel)
    raw = bytearray()
    for row in grid:
        for idx in row:
            raw.extend(_PALETTE[idx])

    img = Image.frombytes("RGBA", (64, 64), bytes(raw))
    # Nearest-neighbor upscale keeps crisp pixel art
    img = img.resize((_TARGET_SIZE, _TARGET_SIZE), Image.NEAREST)
    return img


def image_to_base64(img: Image.Image) -> str:
    """
    Return a base-64 encoded PNG (no data-URL prefix).
    """
    buffer = io.BytesIO()
    img.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def make_image_block(b64_string: str) -> dict[str, Any]:
    """
    Return the JSON block OpenAI expects for an inline base-64 image.
    """
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{b64_string}"},
    }


def image_diff(
    img_a: Image.Image,
    img_b: Image.Image,
    highlight_rgb: Tuple[int, int, int] = (255, 0, 0),  # red
) -> Image.Image:
    """
    Compare img_a vs img_b (any common format), write a visual diff.

    • If the images are identical ➜ diff.png is pure black and the
      function returns True.
    • If they differ ➜ only the changed pixels are tinted highlight_rgb
      on a black background and the function returns False.
    """
    a = np.asarray(img_a.convert("RGB"))
    b = np.asarray(img_b.convert("RGB"))

    if a.shape != b.shape:
        raise ValueError(
            f"Images must have the same dimensions; got {a.shape} vs {b.shape}"
        )

    # Boolean mask: True where *any* channel differs
    diff_mask = np.any(a != b, axis=-1)

    # Fast equality check
    if not diff_mask.any():
        # identical – just save a black image
        return Image.new("RGB", (a.shape[1], a.shape[0]), (0, 0, 0))

    # Start with black canvas, paint the differing pixels
    diff_img = np.zeros_like(a)
    diff_img[diff_mask] = highlight_rgb

    return Image.fromarray(diff_img)


def extract_json(resp: ChatCompletion) -> Any:
    """
    Given the raw OpenAI response object, return the assistant's JSON payload
    as a Python dict.  Works whether the assistant used:
      • bare JSON         { ... }
      • fenced JSON       ```json ... ```
      • generic fence     ``` ... ```
      • wrapper text      "Here's the object you asked for:\n{ ... }"
    Raises ValueError if no JSON object is found.
    """
    content: str = resp.choices[0].message.content
    # 1 - Prefer fenced ```json ... ``` blocks
    fence = re.search(r"```json\s*(\{.*?\})\s*```", content, re.S | re.I)
    if fence:
        json_str = fence.group(1)
    else:
        # 2 - Try any ``` ... ``` fence
        fence = re.search(r"```\s*(\{.*?\})\s*```", content, re.S)
        if fence:
            json_str = fence.group(1)
        else:
            # 3 - Fallback: first '{' … last '}'  (naïve but usually safe)
            start, end = content.find("{"), content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("No JSON object detected in assistant reply")
            json_str = content[start : end + 1]

    # 4 - Load → dict (raises json.JSONDecodeError on bad syntax)
    return json.loads(json_str)


human_actions: dict[GameAction, str] = {
    GameAction.ACTION1: "Move Up",
    GameAction.ACTION2: "Move Down",
    GameAction.ACTION3: "Move Left",
    GameAction.ACTION4: "Move Right",
    GameAction.ACTION5: "Perform Action",
    GameAction.ACTION6: "Click object on screen, make sure to describe object and its relative, not absolute, position",
    GameAction.ACTION7: "Undo",
}


def get_human_inputs_from(available_actions: list[GameAction]) -> str:
    s = ""
    for action in available_actions:
        if action in human_actions:
            s += "\n" + human_actions[action]
    return s


class MultiModalLLM(Agent):
    """An agent that always selects actions at random."""

    # MAX_ACTIONS = 100
    # MODEL: str = "google/gemini-3-pro-preview"
    MAX_ACTIONS = 40
    MODEL: str = "gpt-4o-mini"
    # MODEL: str = "nectarine-alpha-new-reasoning-effort-2025-07-25"
    # MODEL: str = "o3-high"
    REASONING_EFFORT: Optional[str] = None
    MODEL_REQUIRES_TOOLS: bool = False

    SYSTEM_PROMOT = dedent("""\
        You are an abstract reasoning agent that is attempting to solve
        turn-based interactive environments displayed to you as PNGs along
        text for goals, analysis, and planning.
    
        All games have simple abtract graphics and problems that can be 
        solved using nothing but core knowledge.
    """).strip()

    ACTION_INSTRUCT = dedent("""\
        Given the frames and the provided game information above, provide
        your desired action as if you were a human playing the game describing
        your next action to an LLM which will figure out how to perform it.
                             
        ```json
        {
            "human_action": "Click on the red square near the bottom left corner",
            "reasoning": "...",
            "expected_result": "..."                             
        }
                             
        These are going to be multistep games, but only concern yourself with
        the next action.  You should favor moves/actions before trying to click
        on objects, only start clicking once you're sure movement/actions do nothing.

                             
        Only response with the JSON, nothing else.
    """).strip()

    ANALYSE_INSTRUCT = dedent("""\
        ## Instruct

        Given your action, including your expected outcome, and the provided results
        via the associated images provide a complet analysis of the outcome, thinking
        though what happened.  When analizing the images think about the x,y location
        of objects, their colors, and how they relate to the game state.
                              
        The images attached here are as follows (Zero Indexed):
        - 0: Final Frame before your Action
        - 1-N: Frames as a result of your action.
        - A Helper image showing pixels in red that changed between the Final Frame 
          before your action and the last frame after your action.  Any changes 
          larger than a few pixels should be considered significant.
                              
        When examining the images try to identify objects or environmental patterns
        and their locations.
                              
        Provide your analysis and then after providing `---` update the following
        information as you see fit while leaving the structure intact, including what
        you've tried or would like to try in the future.  Note the "Known Human Game
        Inputs" should never be changed as these are provided by the game itself. When
        building the Action Long indicating what input was tried and the outcome 
        you should be as specific as possible, while also indicating how confident you
        are in that assertion while keeping in mind that certain actions might currently
        be blocked before of the game environment.  All of this information should be used
        to understand the game environment and rules in an attempt to beat the game in
        as few moves as possible.        
        ---
    """).strip()

    FIND_ACTION_INSTRUCT = dedent("""\
        Instruct: Given the provided image and the desired action above decide what to do
        base on the following information:
                                  
        "Move Up" = ACTION1
        "Move Down" = ACTION2
        "Move Left" = ACTION3
        "Move Right" = ACTION4
        "Perform Action" = ACTION5
        "Click on Object" = ACTION6, You will need to pull the x, y out of the
            provided image in exact pixels and provide it.
        
        ```json
        {
            "action": "ACTION1",
            "x": 0,
            "y": 0
        }

        Respond with the JSON, nothing else.
    """).strip()

    input_tokens = 0
    output_tokens = 0
    _memory_prompt = ""
    _previous_prompt = ""
    _previous_action = ""
    _previous_images: List[Image.Image] = []
    _previous_score = 0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.input_tokens = 0
        self.output_tokens = 0
        self._memory_prompt = dedent("""\
            ## Known Human Game Inputs
            {{human.inputs}}
                                     
            ## Current Goal
            Use the known human game input to interact with the game environment and learn the rules of the game.
                                     
            ## Game Rules
            Nothing is known currently other than this is a turn based game that I need to solve.
                                     
            ## Action Log
            No Actions So Far
        """).strip()
        _previous_prompt = ""
        _previous_action = ""
        _previous_images: List[Image.Image] = []
        _previous_score = 0

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing or not."""
        return latest_frame.state == GameState.WIN  # type: ignore[no-any-return]

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Using a MultiModal LLM (one that can accept PNG Images and text) decide what to do"""

        # 1 - If the state is NOT_PLAYED or GAME_OVER no choices to be made, must RESET
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            # if game is not started (at init or after GAME_OVER) we need to reset
            # add a small delay before resetting after GAME_OVER to avoid timeout
            return GameAction.RESET

        client = OpenAIClient(api_key=os.environ.get("OPENAI_SECRET_KEY", ""))
        # client = OpenAIClient(
        #     base_url="https://openrouter.ai/api/v1",
        #     api_key=os.environ.get("OPEN_ROUTER_KEY", "")
        # )
        # client = OpenAIClient(
        #     api_key=os.environ.get("GROK_API_KEY", ""),
        #     base_url="https://api.x.ai/v1"
        # )

        image_blocks = [grid_to_image(g) for g in latest_frame.frame]

        # If we haven't provided the known inputs to this game yet, we need to do so.
        if self._memory_prompt.count("{{human.inputs}}") > 0:
            self._memory_prompt = self._memory_prompt.replace(
                "{{human.inputs}}",
                get_human_inputs_from(latest_frame.available_actions),
            )

        # 2 - If we had en expected outcome, present to the LLM in the following format
        #   a. System
        #   b. Previous Frame Images
        #   c. Previous Prompt
        #   d. Previous Response
        #   e. New Frame Images
        #   f. Instruct to Analize Results, including if they lined up with expecations.
        #
        #  This result updates prompts going forward (Allowing the LLM to "program" itself
        #   store memory)
        analysis = "no previous action"
        if self._previous_action:
            level_complete = (
                "NEW LEVEL!!!! - Whatever you did must have beeon good!"
                if latest_frame.levels_completed > self._previous_score
                else ""
            )

            analysze_prompt = (
                f"{level_complete}\n\n{self.ANALYSE_INSTRUCT}\n\n{self._memory_prompt}"
            )

            all_imgs = [
                self._previous_images[-1],
                *image_blocks,
                image_diff(self._previous_images[-1], image_blocks[-1]),
            ]

            # 2. Build a **flat** list of message parts
            msg_parts = [
                make_image_block(image_to_base64(img))  # just append the dict
                for img in all_imgs
            ] + [{"type": "text", "text": analysze_prompt}]
            self.messages = (
                [
                    {"role": "system", "content": self.SYSTEM_PROMOT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self._previous_prompt},
                            # *self._previous_images
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": f"```json\n{json.dumps(self._previous_action)}\n```",
                    },
                    {
                        "role": "user",
                        "content": msg_parts,
                    },
                ],
            )

            response = client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMOT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self._previous_prompt},
                            # *self._previous_images
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": f"```json\n{json.dumps(self._previous_action)}\n```",
                    },
                    {
                        "role": "user",
                        "content": msg_parts,
                    },
                ],
                # extra_body={"reasoning": {"enabled": True}}
            )
            analysis_message = response.choices[0].message.content
            logger.info(f"Assistant - Analysis: {analysis_message}")
            before, _, after = analysis_message.partition("---")  # fastest single-split
            analysis = before.strip()
            self._memory_prompt = after.strip()

        # 3 - Ask the LLM for the next action that we should take, preset the LLM the following
        #   a. System
        #   b. New Frame Images
        #   c. Default or Updated memory prompt from Step 2
        #   d. Instruct for next action, reason for action, and expected outcome
        if len(analysis) > 20:
            self._previous_prompt = (
                f"{analysis}\n\n{self._memory_prompt}\n\n{self.ACTION_INSTRUCT}"
            )
        else:
            self._previous_prompt = f"{self._memory_prompt}\n\n{self.ACTION_INSTRUCT}"
        try:
            response = client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMOT},
                    {
                        "role": "user",
                        "content": [
                            *[
                                make_image_block(image_to_base64(i))
                                for i in image_blocks
                            ],
                            {"type": "text", "text": self._previous_prompt},
                        ],
                    },
                ],
                # extra_body={"reasoning": {"enabled": True}}
                # reasoning_effort=self.REASONING_EFFORT,
            )
        except openai.BadRequestError as e:
            logger.info(f"Message dump: {self.messages}")
            raise e

        self.track_tokens(
            response.usage.prompt_tokens, response.usage.completion_tokens
        )
        action_message = response.choices[0].message.content

        desired_action = extract_json(response)

        logger.info(f"Assistant - Picking Human Move: {action_message}")

        # 4 - Convert the action into a GameAction
        human_action = desired_action.get("human_action")
        if not human_action:
            raise ValueError("No 'human_action' field in the response JSON")

        try:
            response = client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMOT},
                    {
                        "role": "user",
                        "content": [
                            make_image_block(image_to_base64(image_blocks[-1])),
                            {
                                "type": "text",
                                "text": human_action
                                + "\n\n"
                                + self.FIND_ACTION_INSTRUCT,
                            },
                        ],
                    },
                ],
                # extra_body={"reasoning": {"enabled": True}}
                # reasoning_effort=self.REASONING_EFFORT,
            )
        except openai.BadRequestError as e:
            logger.info(f"Message dump: {e}")
            raise e

        print(f"Assistant - Finding Action: {response.choices[0].message.content}")
        self.track_tokens(
            response.usage.prompt_tokens, response.usage.completion_tokens
        )
        current_action = extract_json(response)

        logger.info(f"Assistant - Picking ARC Move: {current_action}")

        action = GameAction.from_name(current_action["action"])
        if action.is_complex():
            action.set_data(
                {
                    "x": max(0, min(current_action["x"], 127)) // _SCALE,
                    "y": max(0, min(current_action["y"], 127)) // _SCALE,
                }
            )
        action.reasoning = {
            "analysis": analysis[:1000] + "..." if len(analysis) > 1000 else analysis,
            "action": current_action["action"]
            if action != GameAction.ACTION6
            else f"{current_action['action']}: [{action.action_data}]",
            "human_action": desired_action["human_action"],
            "reasoning": desired_action["reasoning"][:300] + "..."
            if len(desired_action["reasoning"]) > 300
            else desired_action["reasoning"],
            "expected": desired_action["expected_result"][:300] + "..."
            if len(desired_action["expected_result"]) > 300
            else desired_action["expected_result"],
            "tokens:": [self.input_tokens, self.output_tokens],
        }

        self._previous_action = desired_action
        self._previous_images = image_blocks
        self._previous_score = latest_frame.levels_completed

        return action

    def track_tokens(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
