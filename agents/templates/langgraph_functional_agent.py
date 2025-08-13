"""Uses LangGraph's functional API to build an agent."""

import base64
import io
import json
import logging
import uuid
from typing import Any, TypedDict, TypeVar, cast

import langsmith as ls
import PIL
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint
from langgraph.pregel import Pregel
from langsmith.schemas import Attachment
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage

from agents.templates.llm_agents import LLM

from ..agent import Agent
from ..structs import FrameData, GameAction

logger = logging.getLogger(__name__)


class State(TypedDict, total=False):
    frames: list[FrameData]
    latest_frame: FrameData


MESSAGES = TypeVar("MESSAGES", bound=list[dict[str, Any] | ChatCompletionMessage])

SYS_PROMPT = """# CONTEXT:
You are an agent playing a dynamic game. Your objective is to
WIN and avoid GAME_OVER while minimizing actions.

One action produces one Frame. One Frame is made of one or more sequential
Grids. Each Grid is a matrix size INT<0,63> by INT<0,63> filled with
INT<0,15> values.

# TURN:
Call exactly one action.
"""


def build_agent(
    model: str = "o4-mini",
    tools: list[dict[str, Any]] = [],
    reasoning_effort: str | None = None,
    as_image: bool = True,
) -> Pregel[State, entrypoint.final[ChatCompletionMessage, State]]:
    """Define the agent logic."""
    # Modify this code to add things like reasoning, planning, etc.
    openai_client = OpenAI()
    model_kwargs = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}

    @ls.traceable(run_type="prompt")  # type: ignore[misc]
    def prompt(latest_frame: FrameData, messages: MESSAGES) -> MESSAGES:
        """Build the user prompt for the LLM. Override this method to customize the prompt."""
        content = format_frame(latest_frame, as_image)
        if len(messages) == 0:
            inbound = {
                "role": "user",
                "content": content,
            }
        else:
            inbound = {
                "role": "tool",
                "tool_call_id": cast(ChatCompletionMessage, messages[-1])
                .tool_calls[0]
                .id,
                "content": content,
            }

        return cast(
            MESSAGES,
            [
                {"role": "system", "content": SYS_PROMPT},
                *messages,
                inbound,
            ],
        )

    @ls.traceable  # type: ignore[misc]
    def llm(
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "required",
        **kwargs: Any,
    ) -> ChatCompletionMessage:
        return openai_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

    @entrypoint(checkpointer=InMemorySaver())  # type: ignore[misc]
    def agent(
        state: State, *, previous: list[dict[str, Any]] | None = None
    ) -> entrypoint.final[ChatCompletionMessage, State]:
        # TODO: handle the frame bursts; explore + learn
        # kinda funny bcs this is really just an llm call rn :)
        sys_messages, *convo = prompt(state["latest_frame"], previous or [])
        response = llm(
            model=model,
            messages=[sys_messages, *convo],
            tools=tools,
            tool_choice="required",
            **model_kwargs,
        )
        ai_msg = response.choices[0].message
        ai_msg.tool_calls = ai_msg.tool_calls[:1]  # ensure no extra tools are called
        return entrypoint.final(value=ai_msg, save=[*convo, ai_msg])

    agent.name = "Agent"
    return agent


# Required API


class LangGraphFunc(LLM, Agent):
    """An agent that always selects actions at random."""

    MAX_ACTIONS = 80
    MODEL: str = "o4-mini"
    USE_IMAGE: bool = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._thread_id = uuid.uuid5(uuid.NAMESPACE_DNS, self.game_id)
        self.agent = build_agent(
            self.MODEL,
            tools=self.build_tools(),
            reasoning_effort=self.REASONING_EFFORT,
            as_image=self.USE_IMAGE,
        )

    @ls.traceable  # type: ignore[misc]
    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        msg: ChatCompletionMessage = self.agent.invoke(
            {"frames": frames, "latest_frame": latest_frame},
            {"configurable": {"thread_id": self._thread_id}},
        )
        func = msg.tool_calls[0].function
        action = GameAction.from_name(func.name)
        try:
            args = json.loads(func.arguments) if func.arguments else {}
        except Exception as e:
            args = {}
            logger.warning(f"JSON parsing error on LLM function response: {e}")
        action.set_data(args)
        return action

    def main(self) -> None:
        with ls.trace(
            "LangGraph Agent",
            input={"state": self.state},
            metadata={
                "game_id": self.game_id,
                "card_id": self.card_id,
                "agent_name": self.agent_name,
                "thread_id": self._thread_id,
            },
        ) as rt:
            super().main()
            rt.end(outputs={"state": self.state})


class LangGraphTextOnly(LangGraphFunc, Agent):
    USE_IMAGE = False


def format_frame(latest_frame: FrameData, as_image: bool) -> list[dict[str, Any]]:
    img = g2im(latest_frame.frame) if latest_frame.frame else None
    if as_image and img:
        frame_block = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64.b64encode(img).decode('ascii')}",
            },
        }
    else:
        if (rt := ls.get_current_run_tree()) and img:
            # Save as an attachment so you can easily view while you develop
            rt.attachments["frame"] = Attachment(
                mime_type="image/png",
                data=img,
            )
        lines = []
        for i, block in enumerate(latest_frame.frame):
            lines.append(f"Grid {i}:")
            for row in block:
                lines.append(f"  {row}")
            lines.append("")
        frame_block = {"type": "text", "text": "\n".join(lines)}
    return [
        {
            "type": "text",
            "text": f"""# State:
{latest_frame.state.name}

# Score:
{latest_frame.score}

# Frame:
""",
        },
        frame_block,
        {
            "type": "text",
            "text": """
# TURN:
Reply with a few sentences of plain-text strategy observation about the frame to inform your next action.""",
        },
    ]


def g2im(g: list[list[list[int]]]) -> bytes:
    C = [
        (0, 0, 0),
        (0, 0, 170),
        (0, 170, 0),
        (0, 170, 170),
        (170, 0, 0),
        (170, 0, 170),
        (170, 85, 0),
        (170, 170, 170),
        (85, 85, 85),
        (85, 85, 255),
        (85, 255, 85),
        (85, 255, 255),
        (255, 85, 85),
        (255, 85, 255),
        (255, 255, 85),
        (255, 255, 255),
    ]

    h, w = len(g[0]), len(g[0][0])
    good = [block for block in g if len(block) == h and len(block[0]) == w]
    n = len(good)
    s = 5 * (n > 1)
    W = w * n + s * (n - 1)

    im = PIL.Image.new("RGB", (W, h), "white")
    px = im.load()
    for i, block in enumerate(good):
        ox = i * (w + s)
        for y, row in enumerate(block):
            for x, val in enumerate(row):
                px[ox + x, y] = C[val & 15]

    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()
