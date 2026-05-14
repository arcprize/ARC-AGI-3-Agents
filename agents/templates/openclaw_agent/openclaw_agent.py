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
import uuid
from typing import Any, Optional

import openai
from arcengine import FrameData, GameAction, GameState
from openai import OpenAI as OpenAIClient

from ...agent import Agent

logger = logging.getLogger()

# One run-id per Python process: every fresh `uv run main.py` gets a new
# value, so OpenClaw's server-side session memory starts blank each run.
# Multiple games inside one Swarm run share the same suffix, which is fine —
# their card_id/game_id still keep the per-game sessions distinct. Override
# with OPENCLAW_RUN_ID=<name> to pin a stable, resumable session.
_RUN_ID = os.environ.get("OPENCLAW_RUN_ID") or uuid.uuid4().hex[:8]


class OpenClaw(Agent):
    """An agent that uses an OpenClaw Gateway to play games."""

    MAX_ACTIONS: int = 80

    DEFAULT_BASE_URL = "http://127.0.0.1:18789/v1"
    DEFAULT_AGENT = "openclaw/default"

    token_counter: int

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Parent __init__ uses self.name, which reads self.model and the
        # model override — set both before super().__init__.
        self.model = os.environ.get("OPENCLAW_AGENT", self.DEFAULT_AGENT)
        # OPENCLAW_MODEL overrides the gateway's configured default model
        # per-request (sent as x-openclaw-model on each call) so callers
        # can compare providers without editing ~/.openclaw/openclaw.json.
        self._model_override = os.environ.get("OPENCLAW_MODEL") or None
        super().__init__(*args, **kwargs)
        self.token_counter = 0
        base_url = os.environ.get("OPENCLAW_BASE_URL", self.DEFAULT_BASE_URL)
        token = os.environ.get("OPENCLAW_GATEWAY_TOKEN", "")
        # OpenClaw is stateless per request by default. The session key
        # below pins all turns of one game to a persistent agent session
        # so OpenClaw retains conversation history server-side — that's
        # why choose_action only sends the new user message each turn.
        # The _RUN_ID suffix rotates the key every process so a new
        # `uv run main.py` starts with blank server-side memory.
        self._session_key = f"arc:{self.card_id}:{self.game_id}:{_RUN_ID}"
        self._client = OpenAIClient(
            base_url=base_url,
            api_key=token or "no-auth",  # required by SDK; ignored when auth.mode=none
            default_headers={"x-openclaw-session-key": self._session_key},
        )
        logger.info(
            f"OpenClaw agent for {self.game_id} -> {base_url} model={self.model} "
            f"model_override={self._model_override or '(none)'} "
            f"session={self._session_key}"
        )

    @property
    def name(self) -> str:
        parts = [self.model]
        if self._model_override:
            parts.append(self._model_override)
        sanitized = ".".join(p.replace("/", "-").replace(":", "-") for p in parts)
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
        # `model` here is the OpenClaw agent slug; the underlying provider
        # model is whatever that agent's primary is in ~/.openclaw/openclaw.json.
        # OPENCLAW_MODEL, if set, overrides the provider model per request via
        # the documented x-openclaw-model header.
        extra_headers = (
            {"x-openclaw-model": self._model_override} if self._model_override else None
        )
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                extra_headers=extra_headers,
            )
        except openai.BadRequestError as e:
            logger.error(f"OpenClaw 400: {e}")
            logger.error(f"prompt: {prompt[:500]}")
            raise

        msg = response.choices[0].message

        if response.usage:
            self._track_tokens(response.usage.total_tokens, msg.content or "")

        blob = self._parse_blob(msg)
        action = self._action_from_blob(blob)
        action.reasoning = self._extract_reasoning(blob)
        return action

    def _parse_action(self, msg: Any, latest_frame: FrameData) -> GameAction:
        return self._action_from_blob(self._parse_blob(msg))

    _JSON_BLOB = re.compile(r"\{[^{}]*\"action\"[^{}]*\}", re.DOTALL)

    def _parse_blob(self, msg: Any) -> Optional[dict[str, Any]]:
        text = (msg.content or "").strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        match = self._JSON_BLOB.search(text)
        if match is None:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _action_from_blob(self, blob: Optional[dict[str, Any]]) -> GameAction:
        if not isinstance(blob, dict) or "action" not in blob:
            logger.warning(
                "OpenClaw reply did not parse to action JSON; falling back to ACTION5."
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

    # Mirror of arcengine.enums.MAX_REASONING_BYTES. Structured-field caps
    # below are defensive; the final size enforcement is what guarantees fit.
    _MAX_REASONING_BYTES = 16 * 1024
    _MAX_ALT_CHARS = 200
    _MAX_ALTS = 5

    @classmethod
    def _extract_reasoning(cls, blob: Optional[dict[str, Any]]) -> dict[str, Any]:
        # reasoning_tokens stays 0: OpenClaw v2026.5.7's normalizeUsage has
        # no reasoning slot. Read the real field here if OpenClaw forwards it.
        parsed = isinstance(blob, dict)
        # Re-stating the isinstance on this line (instead of using `parsed`)
        # lets mypy narrow `src` to dict[str, Any] for the .get() calls below.
        src: dict[str, Any] = blob if isinstance(blob, dict) else {}

        thought_raw = src.get("thought")
        thought = str(thought_raw).strip() if thought_raw else ""
        if not thought:
            thought = "(no thought provided)" if parsed else "(parse failed)"

        try:
            confidence = float(src.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        raw_alts = src.get("alternatives_considered")
        if isinstance(raw_alts, list):
            alternatives = [
                str(a)[: cls._MAX_ALT_CHARS] for a in raw_alts[: cls._MAX_ALTS]
            ]
        else:
            alternatives = []

        return cls._enforce_size(
            {
                "thought": thought,
                "confidence": confidence,
                "alternatives_considered": alternatives,
                "reasoning_tokens": 0,
            }
        )

    @classmethod
    def _enforce_size(cls, payload: dict[str, Any]) -> dict[str, Any]:
        # Trim `thought` (the largest, least-structured field) until the JSON
        # payload fits under arcengine's MAX_REASONING_BYTES cap. The doc
        # emphasizes keeping the justification, so trimming it is preferable
        # to letting the server reject the whole step.
        while True:
            raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            excess = len(raw) - cls._MAX_REASONING_BYTES
            if excess <= 0:
                return payload
            thought = payload.get("thought", "")
            if not thought:
                return payload  # nothing left to trim
            # Drop excess chars plus a small margin to absorb JSON-escape growth.
            new_len = max(0, len(thought) - excess - 16)
            if new_len >= len(thought):
                return payload  # no progress possible
            payload["thought"] = thought[:new_len]

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
        return (
            textwrap.dedent(
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
            containing the action AND a brief reasoning trace. No prose
            around it, no markdown fence.

            Required keys:
              "action": one of "RESET", "ACTION1".."ACTION5", "ACTION7",
                       or "ACTION6" with extra integer "x","y" in [0,63].
              "thought": one short sentence (<=200 chars) on why this action.
              "confidence": number in [0,1] (use 0.5 if unsure).
              "alternatives_considered": array of up to 4 short strings
                                         describing other actions weighed
                                         (use [] if none).

            Examples (use these literal string values for "action"):
              {{"action":"ACTION1","thought":"Player is below the door; moving up should advance.","confidence":0.8,"alternatives_considered":["ACTION4 to test right wall"]}}
              {{"action":"ACTION6","x":12,"y":34,"thought":"Click the lone red square.","confidence":0.6,"alternatives_considered":[]}}
              {{"action":"RESET","thought":"State is GAME_OVER; restart.","confidence":1.0,"alternatives_considered":[]}}

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
            )
            .strip()
            .format(
                game_id=latest_frame.game_id,
                state=latest_frame.state.name,
                levels=latest_frame.levels_completed,
                win_levels=latest_frame.win_levels,
                available=self._action_names(latest_frame.available_actions),
                grid=self._render_grid(latest_frame.frame),
            )
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
                    "openclaw_model_override": self._model_override,
                    "openclaw_session_key": self._session_key,
                    "openclaw_total_tokens": self.token_counter,
                }
            )
        super().cleanup(*args, **kwargs)
