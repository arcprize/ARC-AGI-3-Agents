"""OpenClaw agent.

Thin Python shim that routes ARC-AGI-3 actions through the OpenClaw Gateway's
OpenAI-compatible HTTP API (https://docs.openclaw.ai/gateway/openai-http-api).
The actual agent runs in the OpenClaw daemon (Node, BYO LLM key); this class
only translates between the ARC Agent contract and OpenClaw's chat-completions
endpoint, so it plugs into the existing Swarm + agent router unchanged.
"""

import json
import logging
import os
import re
import textwrap
from typing import Any, Optional

import openai
from arcengine import FrameData, GameAction, GameState
from openai import OpenAI as OpenAIClient

from ...agent import Agent

logger = logging.getLogger()


class OpenClaw(Agent):
    """An agent that uses an OpenClaw Gateway to play games."""

    MAX_ACTIONS: int = 80

    DEFAULT_BASE_URL = "http://127.0.0.1:18789/v1"
    DEFAULT_AGENT = "openclaw/default"

    token_counter: int

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Parent __init__ uses self.name, which reads self.model — set it first.
        self.model = os.environ.get("OPENCLAW_AGENT", self.DEFAULT_AGENT)
        super().__init__(*args, **kwargs)
        self.token_counter = 0
        base_url = os.environ.get("OPENCLAW_BASE_URL", self.DEFAULT_BASE_URL)
        token = os.environ.get("OPENCLAW_GATEWAY_TOKEN", "")
        # OpenClaw is stateless per request by default. The session key
        # below pins all turns of one game to a persistent agent session
        # so OpenClaw retains conversation history server-side — that's
        # why choose_action only sends the new user message each turn.
        self._session_key = f"arc:{self.card_id}:{self.game_id}"
        self._client = OpenAIClient(
            base_url=base_url,
            api_key=token or "no-auth",  # required by SDK; ignored when auth.mode=none
            default_headers={"x-openclaw-session-key": self._session_key},
        )
        logger.info(
            f"OpenClaw agent for {self.game_id} -> {base_url} model={self.model} "
            f"session={self._session_key}"
        )

    @property
    def name(self) -> str:
        sanitized = self.model.replace("/", "-").replace(":", "-")
        return f"{super().name}.{sanitized}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        # NOT_PLAYED on first call, GAME_OVER after a fail. Either way the only
        # legal action is RESET; spend zero tokens on it.
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET

        # OpenClaw is a stateful gateway: the x-openclaw-session-key header
        # we set in __init__ scopes a persistent agent session, so we only
        # send the NEW user message each turn. The OpenClaw daemon stitches
        # this onto its server-side conversation history before forwarding
        # to the upstream provider. Sending our own accumulated history
        # here would duplicate the conversation and defeat prompt caching.
        # OpenClaw's OpenAI-compat layer also silently drops the `tools`
        # field for some providers (verified May 2026 against Anthropic),
        # so the agent uses a JSON-in-text protocol instead.
        prompt = self._build_prompt(latest_frame)
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
        except openai.BadRequestError as e:
            logger.error(f"OpenClaw 400: {e}")
            logger.error(f"prompt: {prompt[:500]}")
            raise

        msg = response.choices[0].message

        if response.usage:
            self._track_tokens(response.usage.total_tokens, msg.content or "")

        return self._parse_action(msg, latest_frame)

    _JSON_BLOB = re.compile(r"\{[^{}]*\"action\"[^{}]*\}", re.DOTALL)

    def _parse_action(
        self, msg: Any, latest_frame: FrameData
    ) -> GameAction:
        # The inline prompt asks for one JSON object naming the action. Tolerate
        # stray whitespace, markdown fences, or leading prose by extracting the
        # first {...} block containing "action".
        text = (msg.content or "").strip()
        blob = None
        if text:
            try:
                blob = json.loads(text)
            except json.JSONDecodeError:
                match = self._JSON_BLOB.search(text)
                if match:
                    try:
                        blob = json.loads(match.group(0))
                    except json.JSONDecodeError:
                        blob = None

        if not isinstance(blob, dict) or "action" not in blob:
            logger.warning(
                f"OpenClaw reply did not parse to action JSON; "
                f"falling back to ACTION5. raw={text[:200]!r}"
            )
            return GameAction.ACTION5

        raw = str(blob.get("action", "")).upper().strip()
        # Accept either the canonical name ("ACTION1", "RESET") or the integer
        # id ("1", "0"). GameAction is a plain Enum (not IntEnum) so we look
        # up by .value rather than constructor.
        action: Optional[GameAction] = None
        try:
            action = GameAction.from_name(raw)
        except (KeyError, ValueError, AttributeError):
            try:
                wanted = int(raw)
                action = next((a for a in GameAction if a.value == wanted), None)
            except (TypeError, ValueError):
                action = None
        if action is None:
            logger.warning(
                f"OpenClaw returned unknown action {raw!r}; falling back to ACTION5"
            )
            return GameAction.ACTION5

        if action.is_complex():
            try:
                action.set_data(
                    {"x": int(blob.get("x", 32)), "y": int(blob.get("y", 32))}
                )
            except (TypeError, ValueError):
                action.set_data({"x": 32, "y": 32})
        return action

    def _track_tokens(self, tokens: int, content: str) -> None:
        self.token_counter += tokens
        if hasattr(self, "recorder") and not self.is_playback:
            self.recorder.record(
                {
                    "tokens": tokens,
                    "total_tokens": self.token_counter,
                    "assistant": content,
                }
            )
        logger.info(
            f"OpenClaw used {tokens} tokens (total {self.token_counter}) "
            f"for {self.game_id}"
        )

    def _build_prompt(self, latest_frame: FrameData) -> str:
        return textwrap.dedent(
            """
            You are playing an unfamiliar turn-based grid game. Reach state=WIN
            to win. Each turn provides the latest observed frame and the legal
            actions for that state.

            You may use OpenClaw's built-in memory and file tools to keep
            persistent notes between turns about anything you've figured out:
            object identities, control effects, goals, hazards, counters,
            positions, repeated failures, hypotheses to test, and action
            sequences that helped. Read your notes at the start of a turn;
            update them when you observe something new. The session retains the
            conversation history but notes give you a stable scratchpad you
            control.

            After any tool use, your FINAL response must be one JSON object
            naming the action to take. No prose around it, no markdown fence.

            Examples (use these literal string values for "action"):
              {{"action":"ACTION1"}}
              {{"action":"ACTION3"}}
              {{"action":"ACTION6","x":12,"y":34}}
              {{"action":"RESET"}}

            Action meanings:
              "RESET"=start/restart.
              "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", and
              "ACTION7" are simple inputs. Many games use ACTION1/ACTION2/
              ACTION3/ACTION4 as directional inputs and ACTION5 as an
              interaction input, but you must infer each action's effect from
              observations in the current game.
              "ACTION6"=click/point at x,y with both coordinates in [0,63].

            Rules:
            - Only RESET when state is NOT_PLAYED or GAME_OVER.
            - Pick from available_actions when given.
            - Final output: a single JSON object. No markdown.

            # FRAME
            game_id: {game_id}
            state: {state}
            levels_completed: {levels}
            win_levels: {win_levels}
            available_actions: {available}

            # GRID (hex)
            {grid}
            """
        ).strip().format(
            game_id=latest_frame.game_id,
            state=latest_frame.state.name,
            levels=latest_frame.levels_completed,
            win_levels=latest_frame.win_levels,
            available=self._action_names(latest_frame.available_actions),
            grid=self._render_grid(latest_frame.frame),
        )

    def _action_names(self, actions: Optional[list[Any]]) -> list[str]:
        # GameAction is a plain Enum (not IntEnum), so value->member must be
        # done by scanning members rather than via GameAction(value).
        out: list[str] = []
        for a in actions or []:
            if isinstance(a, GameAction):
                out.append(a.name)
                continue
            try:
                wanted = int(a)
                matched = next((m for m in GameAction if m.value == wanted), None)
                out.append(matched.name if matched else str(a))
            except (TypeError, ValueError):
                out.append(str(a))
        return out

    def _render_grid(self, grid_3d: Optional[list[list[list[int]]]]) -> str:
        if not grid_3d:
            return "(no grid)"
        lines: list[str] = []
        for i, plane in enumerate(grid_3d):
            lines.append(f"Grid {i}:")
            for row in plane:
                lines.append("  " + "".join(f"{c:x}" for c in row))
        return "\n".join(lines)

    def cleanup(self, *args: Any, **kwargs: Any) -> None:
        if self._cleanup and hasattr(self, "recorder") and not self.is_playback:
            self.recorder.record(
                {
                    "openclaw_model": self.model,
                    "openclaw_total_tokens": self.token_counter,
                }
            )
        super().cleanup(*args, **kwargs)
