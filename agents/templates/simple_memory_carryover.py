import json
import logging
import os
import re
import textwrap
import time
import uuid
from typing import Any

from arcengine import FrameData, GameAction, GameState
from openai import AuthenticationError, BadRequestError
from openai import OpenAI as OpenAIClient

from ..agent import Agent

logger = logging.getLogger(__name__)

MAX_REASONING_BYTES = 16 * 1024


class SimpleMemoryCarryover(Agent):
    """LLM agent with N-turn history and explicit memory carryover."""

    MAX_ACTIONS = 40
    MAX_ATTEMPTS = 3
    HISTORY_LENGTH = 1
    MAX_ANIMATION_FRAMES = 7
    ACTION_MAPPINGS = {
        "RESET": "RESET - Reset current level",
        "ACTION1": "ACTION1 - Up arrow key, W",
        "ACTION2": "ACTION2 - Down arrow key, S",
        "ACTION3": "ACTION3 - Left arrow key, A",
        "ACTION4": "ACTION4 - Right arrow key, D",
        "ACTION5": "ACTION5 - Spacebar",
        "ACTION6": "ACTION6 - Click",
        "ACTION7": "ACTION7 (special action, Undo) - Z",
    }

    RECORDINGS_DIR_ENV = "RECORDINGS_DIR"
    BASE_URL = "https://openrouter.ai/api/v1"
    MODEL = "anthropic/claude-opus-4.6"

    ENABLE_CHAT_LOG = True
    ENABLE_TURN_LOG = True
    DETERMINISTIC_FALLBACK = True
    # Only relevant when DETERMINISTIC_FALLBACK = True
    MAX_CONSECUTIVE_FALLBACKS = 2

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing required env var: OPENROUTER_API_KEY")
        self.base_url = self.BASE_URL
        self.model = self.MODEL
        self.carryover_memory = ""
        self._pending_reasoning: dict[str, Any] | None = None
        self.consecutive_parse_failures = 0
        self._consecutive_fallbacks = 0
        self._force_exit = False
        self._turn_history: list[tuple[list[list[list[Any]]], str, str]] = []
        self._log_conversation = self.ENABLE_CHAT_LOG
        self._log_turns = self.ENABLE_TURN_LOG
        self._chat_log_path: str | None = None
        self._turn_log_path: str | None = None
        self._conversation_log_session_id = uuid.uuid4().hex
        self._conversation_log_announced = False
        self._turn_log_announced = False
        self.client = OpenAIClient(
            api_key=api_key,
            base_url=self.base_url,
        )
        super().__init__(*args, **kwargs)

    @property
    def name(self) -> str:
        model_name = getattr(self, "model", self.MODEL)
        sanitized_model_name = model_name.replace("/", "-").replace(":", "-")
        return f"{super().name}.{sanitized_model_name}.memory-carryover"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN or self._force_exit

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state is GameState.GAME_OVER:
            return GameAction.RESET
        if self._force_exit:
            return GameAction.RESET
        turn_start = time.time()
        available_actions = self._available_actions(latest_frame)
        fallback_action = self._deterministic_fallback_action(latest_frame)
        max_attempts = self.MAX_ATTEMPTS
        system_prompt = self._build_system_prompt(
            available_actions, latest_state=latest_frame.state
        )
        user_prompt = self._build_user_prompt(latest_frame)
        response: Any | None = None
        request_payload: dict[str, Any] | None = None
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        request_payload = create_kwargs

        llm_request_sent: float | None = None
        llm_response_received: float | None = None
        last_error: str | None = None
        for attempt_index in range(max_attempts):
            try:
                llm_request_sent = time.time()
                response = self.client.chat.completions.create(**create_kwargs)
                llm_response_received = time.time()
                args = self._extract_action_and_memory_from_response(response)

                action, action_note = self._action_from_arguments(
                    args=args,
                    fallback_action=fallback_action,
                    available_actions=available_actions,
                )
                retry_error = self._retry_error_from_action_note(action_note)
                if retry_error is not None:
                    if attempt_index < (max_attempts - 1):
                        raise ValueError(retry_error)
                    raise ValueError(f"{retry_error} after retries.")

                self.carryover_memory = self._extract_memory(args)
                self.consecutive_parse_failures = 0
                self._consecutive_fallbacks = 0
                timestamps: dict[str, float | None] = {
                    "turn_start": turn_start,
                    "llm_request_sent": llm_request_sent,
                    "llm_response_received": llm_response_received,
                    "turn_end": time.time(),
                }
                self._pending_reasoning = self._build_action_reasoning(
                    response=response,
                    parsed_args=args,
                )
                self._log_turn(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response=response,
                    parsed_args=args,
                    action=action,
                    error=None,
                    memory_after=self.carryover_memory,
                    request_payload=request_payload,
                )
                self._log_turn_jsonl(
                    latest_frame=latest_frame,
                    action=action,
                    action_note=action_note,
                    parse_status="ok",
                    memory_after=self.carryover_memory,
                    timestamps=timestamps,
                    response=response,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    error=None,
                )
                self._store_turn_history(latest_frame, action)
                return action
            except AuthenticationError as e:
                error = f"AuthenticationError: {e}"
                self._log_turn(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response=response,
                    parsed_args=None,
                    action=fallback_action,
                    error=error,
                    memory_after=self.carryover_memory,
                    request_payload=request_payload,
                )
                raise RuntimeError(
                    "OpenRouter authentication failed. Check OPENROUTER_API_KEY."
                ) from e
            except (
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
                BadRequestError,
            ) as e:
                last_error = f"{type(e).__name__}: {e}"
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"

            if attempt_index < (max_attempts - 1):
                logger.warning(
                    "Action selection failed; retrying with the same prompt (attempt %s/%s). Error: %s",
                    attempt_index + 1,
                    max_attempts,
                    last_error,
                )
                continue

        error = last_error or "Action selection failed after retries."

        if not self.DETERMINISTIC_FALLBACK:
            logger.warning(
                "All %s LLM attempts failed and DETERMINISTIC_FALLBACK is off. Ending game.",
                max_attempts,
            )
            self._force_exit = True
            action = GameAction.RESET
        else:
            action = self._handle_parse_failure(
                fallback_action,
                error,
                response=response,
            )
            self._consecutive_fallbacks += 1
            if self._consecutive_fallbacks >= self.MAX_CONSECUTIVE_FALLBACKS:
                logger.warning(
                    "Hit %s consecutive fallbacks. Ending game.",
                    self._consecutive_fallbacks,
                )
                self._force_exit = True
                action = GameAction.RESET

        self._log_turn(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=response,
            parsed_args=None,
            action=action,
            error=error,
            memory_after=self.carryover_memory,
            request_payload=request_payload,
        )
        fallback_timestamps = None
        if turn_start is not None:
            fallback_timestamps = {
                "turn_start": turn_start,
                "llm_request_sent": llm_request_sent,
                "llm_response_received": llm_response_received,
                "turn_end": time.time(),
            }
        self._log_turn_jsonl(
            latest_frame=latest_frame,
            action=action,
            action_note="fallback_after_parse_failure",
            parse_status="failed",
            memory_after=self.carryover_memory,
            timestamps=fallback_timestamps,
            response=response,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            error=error,
        )
        self._store_turn_history(latest_frame, action)
        return action

    def _store_turn_history(self, latest_frame: FrameData, action: GameAction) -> None:
        action_label = action.name
        if action.is_complex():
            data = action.action_data.model_dump()
            action_label = f"{action.name} x={data.get('x')} y={data.get('y')}"
        self._turn_history.append(
            (latest_frame.frame, action_label, self.carryover_memory)
        )
        max_keep = self.HISTORY_LENGTH * 2
        if len(self._turn_history) > max_keep:
            self._turn_history = self._turn_history[-self.HISTORY_LENGTH :]

    def _build_system_prompt(
        self, available_actions: list[GameAction], latest_state: GameState
    ) -> str:
        available_names = [action.name for action in available_actions]
        available_actions_inline = ", ".join(available_names) or "<none>"
        non_click_action_names = [
            action_name
            for action_name in available_names
            if action_name not in (GameAction.ACTION6.name, GameAction.RESET.name)
        ]
        non_click_actions_inline = ", ".join(non_click_action_names) or "<none>"
        action_mappings = self._available_action_mappings(available_actions)
        action6_available = GameAction.ACTION6 in available_actions
        game_state = (
            latest_state.name
            if isinstance(latest_state, GameState)
            else str(latest_state)
        )
        action6_rule = ""
        action6_format_rule = ""
        non_click_format_rule = ""
        if non_click_action_names:
            non_click_format_rule = f"- For non-click actions, use `<action>ACTION_NAME</action>` where ACTION_NAME is one of ({non_click_actions_inline})\n"
        if action6_available:
            action6_rule = "- If you choose ACTION6, you must also provide integer x and y in [0,63].\n"
            action6_format_rule = "- For ACTION6, provide coordinates as `<action>ACTION6 x=<int> y=<int></action>`\n"

        example_parts: list[str] = []
        if non_click_action_names:
            example_parts.append(
                "Example for a simple action:\n"
                "  <action>ACTION1</action>\n"
                "  <notes>Level 1: walls are color 5. Moving up changed row 3.</notes>"
            )
        if action6_available:
            example_parts.append(
                "Example for ACTION6 (click):\n"
                "  <action>ACTION6 x=32 y=12</action>\n"
                "  <notes>Clicked at (32,12). The blue cell moved right.</notes>"
            )
        examples_section = ""
        if example_parts:
            examples_section = "\n\n## Examples\n\n" + "\n\n".join(example_parts)

        return textwrap.dedent(
            """
You are a turn-based game-playing agent. Your task is to effiently complete the game.

You do NOT have persistent hidden memory. Your notes work as follows:
- You may carry forward notes to the next turn by including <notes>...</notes> in your response.
- When you write `<notes>...</notes>`, it COMPLETELY REPLACES all previous notes
- Your notes will be shown back to you in the HISTORY section on the next turn, alongside the frame you saw and the action you took
- Any information not included in your `<notes>` output will be permanently lost
- Use notes to record observations, hypotheses, and plans

## Turn History

You will see the last few turns in the HISTORY section. Each entry shows:
1. The frame you saw
2. The action you took
3. The notes you carried forward

Use this to track patterns and understand the effects of your actions. On the first turn there is no history.

## Your Task

1. Analyze the current frame
2. Review your history (previous frames, actions, and notes)
3. Choose exactly one action from the available options
4. Write notes you want to carry forward
5. Submit your chosen action and notes

## Available Actions

Choose exactly ONE action from ({available_actions_inline}).
Available action mappings:
{action_mappings}
Current game state: {game_state}
{action6_rule}
## Instructions for Your Response

Think through your reasoning, then provide your chosen action and notes.

Your output must contain exactly:
1) one <action>...</action> block
2) one <notes>...</notes> block

Formatting rules:
{non_click_format_rule}
{action6_format_rule}
- Do not output pseudo-code or plain text action descriptions outside these tags
{examples_section}
            """.format(
                available_actions_inline=available_actions_inline,
                action_mappings=action_mappings,
                game_state=game_state,
                action6_rule=action6_rule,
                non_click_format_rule=non_click_format_rule,
                action6_format_rule=action6_format_rule,
                examples_section=examples_section,
            )
        ).strip()

    def _build_user_prompt(self, latest_frame: FrameData) -> str:
        turn_data_text = self._build_turn_data_text(latest_frame)
        history_text = self._build_history_text()
        animation_frames = self._interpolate_frames(latest_frame.frame)

        sections = ["TURN_DATA:", turn_data_text, ""]

        if history_text:
            sections.extend(["HISTORY:", history_text, ""])

        sections.extend(
            [
                "CURRENT_FRAME:",
                self._pretty_print_3d(animation_frames),
            ]
        )

        return "\n".join(sections).strip()

    def _build_history_text(self) -> str:
        if not self._turn_history:
            return ""

        recent = self._turn_history[-self.HISTORY_LENGTH :]
        total_history = len(self._turn_history)
        start_index = total_history - len(recent)

        parts: list[str] = []
        for offset, (frame_data, action_name, notes) in enumerate(recent):
            turn_number = start_index + offset
            interpolated = self._interpolate_frames(frame_data)
            frame_text = self._pretty_print_3d(interpolated)
            parts.append(f"--- Turn {turn_number} ---")
            parts.append(frame_text)
            parts.append(f"Action taken: {action_name}")
            parts.append(f"Notes: {notes if notes else '[none]'}")
            parts.append("")

        return "\n".join(parts).strip()

    def _interpolate_frames(
        self, frames: list[list[list[Any]]]
    ) -> list[list[list[Any]]]:
        n = len(frames)
        max_frames = self.MAX_ANIMATION_FRAMES
        if n <= max_frames:
            return frames
        if max_frames <= 1:
            return [frames[0]]
        if max_frames == 2:
            return [frames[0], frames[-1]]
        indices = [round(i * (n - 1) / (max_frames - 1)) for i in range(max_frames)]
        return [frames[idx] for idx in indices]

    def _build_turn_data_text(self, latest_frame: FrameData) -> str:
        state_name = self._frame_state_name(latest_frame)
        levels_completed = self._coerce_optional_int(
            getattr(latest_frame, "levels_completed", None)
        )
        win_levels = self._coerce_optional_int(
            getattr(latest_frame, "win_levels", None)
        )
        current_level_index = self._current_level_index(levels_completed, win_levels)

        lines = [
            f"- game_state: {state_name}",
            f"- current_level: {self._format_optional_int(current_level_index)}",
        ]
        return "\n".join(lines)

    def _frame_state_name(self, frame: FrameData) -> str:
        state = getattr(frame, "state", None)
        if isinstance(state, GameState):
            return str(state.name)
        return str(state)

    def _coerce_optional_int(self, value: Any) -> int | None:
        try:
            if value is None:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    def _format_optional_int(self, value: int | None) -> str:
        if value is None:
            return "unknown"
        return str(value)

    def _current_level_index(
        self, levels_completed: int | None, win_levels: int | None
    ) -> int | None:
        if levels_completed is None:
            return None
        candidate = levels_completed + 1
        if win_levels is not None and win_levels > 0:
            return min(candidate, win_levels)
        return candidate

    def _parse_action_token(self, action_token: str) -> GameAction:
        token = action_token.strip().upper()
        if not token:
            raise ValueError("missing action_name")
        return GameAction.from_name(token)

    def _available_action_mappings(self, available_actions: list[GameAction]) -> str:
        lines: list[str] = []
        for action in available_actions:
            action_name = action.name
            label = self.ACTION_MAPPINGS.get(action_name)
            if label is None:
                lines.append(f"- {action_name}")
            else:
                lines.append(f"- {label}")
        if not lines:
            return "- <none>"
        return "\n".join(lines)

    def _extract_action_and_memory_from_response(self, response: Any) -> dict[str, Any]:
        response_text = self._extract_response_text(response)
        action_text = self._extract_tag_content(response_text, "action")
        try:
            notes_text = self._extract_tag_content(response_text, "notes")
        except ValueError:
            notes_text = self._extract_tag_content(response_text, "memory")
        args = self._parse_action_text(action_text)
        args["memory_for_next_turn"] = notes_text
        return args

    def _extract_response_text(self, response: Any) -> str:
        if not response.choices:
            raise ValueError("No choices returned from model.")
        message = response.choices[0].message
        content = getattr(message, "content", None)
        if content is None:
            raise ValueError("Model response content was empty.")

        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts: list[str] = []
            for item in content:
                item_text: str | None = None
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        raw_text = item.get("text")
                        item_text = str(raw_text) if raw_text is not None else None
                    elif item.get("text") is not None:
                        item_text = str(item.get("text"))
                else:
                    raw_text = getattr(item, "text", None)
                    if raw_text is not None:
                        item_text = str(raw_text)
                if item_text:
                    parts.append(item_text)
            text = "\n".join(parts)
        else:
            text = str(content)

        text = text.strip()
        if not text:
            raise ValueError("Model response text was empty.")
        return text

    def _extract_tag_content(self, response_text: str, tag_name: str) -> str:
        pattern = re.compile(
            rf"<{tag_name}>(.*?)</{tag_name}>",
            re.IGNORECASE | re.DOTALL,
        )
        matches = pattern.findall(response_text)
        if not matches:
            raise ValueError(
                f"Missing <{tag_name}>...</{tag_name}> block in model output."
            )
        return str(matches[-1]).strip()

    def _parse_action_text(self, action_text: str) -> dict[str, Any]:
        payload = action_text.strip()
        if not payload:
            raise ValueError("empty <action> tag")

        if payload.startswith("{"):
            parsed = json.loads(payload)
            if not isinstance(parsed, dict):
                raise ValueError("<action> JSON must decode to an object.")
            return parsed

        action_match = re.search(r"\b(ACTION[1-7]|RESET)\b", payload, re.IGNORECASE)
        if action_match is None:
            raise ValueError(f"Could not parse action name from <action>: {payload!r}")
        action_name = action_match.group(1).upper()
        parsed_action: dict[str, Any] = {"action_name": action_name}

        x_match = re.search(r"\bx\s*[:=]\s*(-?\d+)", payload, re.IGNORECASE)
        y_match = re.search(r"\by\s*[:=]\s*(-?\d+)", payload, re.IGNORECASE)
        if x_match is not None and y_match is not None:
            parsed_action["x"] = int(x_match.group(1))
            parsed_action["y"] = int(y_match.group(1))
            return parsed_action

        if action_name == GameAction.ACTION6.name:
            remainder = payload[action_match.end() :]
            coords = re.findall(r"-?\d+", remainder)
            if len(coords) >= 2:
                parsed_action["x"] = int(coords[0])
                parsed_action["y"] = int(coords[1])
        return parsed_action

    def _extract_memory(self, args: dict[str, Any]) -> str:
        memory = args.get("memory_for_next_turn", "")
        if isinstance(memory, str):
            return memory.strip()
        return json.dumps(memory, ensure_ascii=True).strip()

    def _action_from_arguments(
        self,
        args: dict[str, Any],
        fallback_action: GameAction,
        available_actions: list[GameAction],
    ) -> tuple[GameAction, str]:
        action_name = str(
            args.get("action_name", args.get("action", args.get("name", "")))
        ).strip()
        try:
            requested_action = self._parse_action_token(action_name)
        except ValueError:
            return fallback_action, f"invalid_action_name={action_name}"

        if requested_action not in available_actions:
            return fallback_action, f"action_not_available={requested_action.name}"

        if requested_action.is_complex():
            x, y = self._parse_coordinates(args)
            if x is None or y is None:
                return fallback_action, "invalid_coordinates_for_action6"
            requested_action.set_data({"x": x, "y": y})
            return requested_action, "ok_action6"

        return requested_action, "ok_simple"

    def _retry_error_from_action_note(self, action_note: str) -> str | None:
        if action_note == "invalid_coordinates_for_action6":
            return "ACTION6 was selected without valid coordinates"
        if action_note.startswith("invalid_action_name="):
            return f"Model selected an invalid action token ({action_note})"
        if action_note.startswith("action_not_available="):
            return f"Model selected an unavailable action ({action_note})"
        return None

    def _parse_coordinates(self, args: dict[str, Any]) -> tuple[int | None, int | None]:
        try:
            x_raw = args.get("x")
            y_raw = args.get("y")
            if x_raw is None or y_raw is None:
                return None, None
            x = int(x_raw)
            y = int(y_raw)
        except (TypeError, ValueError):
            return None, None
        if not (0 <= x <= 63 and 0 <= y <= 63):
            return None, None
        return x, y

    def _handle_parse_failure(
        self,
        fallback_action: GameAction,
        error: str,
        response: Any | None = None,
    ) -> GameAction:
        self.consecutive_parse_failures += 1

        self._pending_reasoning = self._build_action_reasoning(
            response=response,
            parsed_args=None,
            error=error,
        )
        return fallback_action

    def do_action_request(self, action: GameAction) -> FrameData:
        data = action.action_data.model_dump()
        reasoning_payload = self._pending_reasoning
        try:
            raw = self.arc_env.step(
                action,
                data=data,
                reasoning=reasoning_payload,
            )
            action_input = getattr(raw, "action_input", None)
            if (
                reasoning_payload is not None
                and action_input is not None
                and getattr(action_input, "reasoning", None) is None
            ):
                action_input.reasoning = reasoning_payload
            return self._convert_raw_frame_data(raw)
        finally:
            self._pending_reasoning = None

    def _build_action_reasoning(
        self,
        *,
        response: Any | None,
        parsed_args: dict[str, Any] | None,
        error: str | None = None,
    ) -> dict[str, Any]:
        response_text = None
        if response is not None:
            try:
                response_text = self._extract_response_text(response)
            except (ValueError, AttributeError):
                pass

        payload: dict[str, Any] = {
            "model": self.model,
            "content": response_text,
            "parsed_output": parsed_args,
        }

        if error is not None:
            payload["error"] = error

        self._enforce_reasoning_size_limit(payload)
        return payload

    def _enforce_reasoning_size_limit(self, payload: dict[str, Any]) -> None:
        serialized = json.dumps(payload, ensure_ascii=True, default=str)
        if len(serialized.encode("utf-8")) <= MAX_REASONING_BYTES:
            return

        content = payload.get("content")
        if isinstance(content, str) and len(content) > 500:
            payload["content"] = content[:500] + "...(truncated)"
            serialized = json.dumps(payload, ensure_ascii=True, default=str)
            if len(serialized.encode("utf-8")) <= MAX_REASONING_BYTES:
                return

        payload["content"] = None

    def _extract_token_usage(self, response: Any) -> dict[str, Any] | None:
        if response is None:
            return None
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }

    def _available_actions(self, latest_frame: FrameData) -> list[GameAction]:
        if latest_frame.state is GameState.GAME_OVER:
            return [GameAction.RESET]
        raw_actions = getattr(latest_frame, "available_actions", None) or []
        parsed: list[GameAction] = []
        for raw in raw_actions:
            action = self._coerce_action(raw)
            if action is not None:
                parsed.append(action)
        if GameAction.RESET not in parsed:
            parsed.append(GameAction.RESET)
        return parsed

    def _deterministic_fallback_action(self, latest_frame: FrameData) -> GameAction:
        available_actions = self._available_actions(latest_frame)
        non_reset_actions = [a for a in available_actions if a != GameAction.RESET]
        for action in non_reset_actions:
            if action.is_simple():
                return action
        for action in non_reset_actions:
            if action.is_complex():
                action.set_data({"x": 0, "y": 0})
                return action
        return GameAction.RESET

    def _pretty_print_3d(self, array_3d: list[list[list[Any]]]) -> str:
        lines = []
        single = len(array_3d) == 1
        for i, block in enumerate(array_3d):
            if not single:
                lines.append(f"Frame Step {i + 1}:")
            for row in block:
                lines.append(" ".join(str(value) for value in row))
            lines.append("")
        return "\n".join(lines).strip()

    def _coerce_action(self, raw: Any) -> GameAction | None:
        if isinstance(raw, GameAction):
            return raw
        if isinstance(raw, str):
            try:
                return GameAction.from_name(raw)
            except ValueError:
                return None
        if isinstance(raw, int):
            try:
                return GameAction.from_id(raw)
            except ValueError:
                return None
        return None

    def _resolve_chat_log_path(self) -> str:
        directory = os.getenv(self.RECORDINGS_DIR_ENV, "").strip()
        if not directory:
            directory = "recordings"
        os.makedirs(directory, exist_ok=True)

        filename = (
            f"{self.name}.{self._conversation_log_session_id}.conversation.chat.md"
        )
        return os.path.join(directory, filename)

    def _resolve_turn_log_path(self) -> str:
        directory = os.getenv(self.RECORDINGS_DIR_ENV, "").strip()
        if not directory:
            directory = "recordings"
        os.makedirs(directory, exist_ok=True)

        filename = f"{self.name}.{self._conversation_log_session_id}.turnlog.jsonl"
        return os.path.join(directory, filename)

    def _log_turn_jsonl(
        self,
        *,
        latest_frame: FrameData,
        action: GameAction,
        action_note: str,
        parse_status: str,
        memory_after: str,
        timestamps: dict[str, float | None] | None,
        response: Any | None,
        system_prompt: str,
        user_prompt: str,
        error: str | None,
    ) -> None:
        if not self._log_turns:
            return

        if self._turn_log_path is None:
            self._turn_log_path = self._resolve_turn_log_path()
            if not self._turn_log_announced:
                logger.info(
                    "SimpleMemoryCarryover turn log: %s",
                    self._turn_log_path,
                )
                self._turn_log_announced = True

        state_name = self._frame_state_name(latest_frame)
        levels_completed = self._coerce_optional_int(
            getattr(latest_frame, "levels_completed", None)
        )
        win_levels = self._coerce_optional_int(
            getattr(latest_frame, "win_levels", None)
        )
        current_level = self._current_level_index(levels_completed, win_levels)

        record: dict[str, Any] = {
            "turn": self.action_counter,
            "game_id": self.game_id,
            "game_state": state_name,
            "current_level": current_level,
            "action": action.name,
            "action_note": action_note,
            "parse_status": parse_status,
            "memory_chars": len(memory_after),
            "memory": memory_after,
        }

        if action.is_complex():
            data = action.action_data.model_dump()
            record["action_data"] = {"x": data.get("x"), "y": data.get("y")}

        if timestamps:
            record["timestamps"] = timestamps

        token_usage = self._extract_token_usage(response)
        if token_usage:
            record["token_usage"] = token_usage

        record["system_prompt"] = system_prompt
        record["user_prompt"] = user_prompt

        response_text = None
        if response is not None:
            try:
                response_text = self._extract_response_text(response)
            except (ValueError, AttributeError):
                pass
        record["response_text"] = response_text

        if error is not None:
            record["error"] = error

        try:
            line = json.dumps(record, ensure_ascii=True, default=str)
            with open(self._turn_log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as exc:
            logger.warning("Failed to write turn log: %s", exc)

    def _format_chat_turn(
        self,
        *,
        turn_index: int,
        system_prompt: str,
        user_prompt: str,
        parsed_args: dict[str, Any] | None,
        response: Any | None,
        action: GameAction,
        error: str | None,
        memory_after: str,
        request_payload: dict[str, Any] | None,
    ) -> str:
        response_payload = self._raw_response_payload(response)
        action_data = action.action_data.model_dump()
        action6_lines: list[str] = []
        if action.name == "ACTION6":
            action6_lines = [
                f"- ACTION6 coordinates: x={action_data.get('x')}, y={action_data.get('y')}",
            ]
        lines = [
            f"## Turn {turn_index}",
            "",
            "### System Prompt",
            "```text",
            system_prompt,
            "```",
            "",
            "### User Prompt",
            "```text",
            user_prompt,
            "```",
            "",
            "### API Request (Raw)",
            "```json",
            json.dumps(request_payload, ensure_ascii=True, indent=2),
            "```",
            "",
            "### API Response (Raw)",
            "```json",
            json.dumps(response_payload, ensure_ascii=True, indent=2),
            "```",
            "",
            "### Parsed Output (Derived)",
            "```json",
            json.dumps(parsed_args, ensure_ascii=True, indent=2),
            "```",
            "",
            "### Decision",
            f"- selected_action: {action.name} ({action.value})",
            *action6_lines,
            "Memory for next turn:",
            "```text",
            memory_after if memory_after else "<empty>",
            "```",
            f"- parse_failures_in_row: {self.consecutive_parse_failures}",
            f"- error: {error or '<none>'}",
            "",
            "---",
            "",
        ]
        return "\n".join(lines)

    def _raw_response_payload(self, response: Any | None) -> Any:
        if response is None:
            return None
        if hasattr(response, "model_dump"):
            try:
                return self._redact_thought_signatures(response.model_dump())
            except Exception:
                pass
        if hasattr(response, "to_dict"):
            try:
                return self._redact_thought_signatures(response.to_dict())
            except Exception:
                pass
        return {"repr": repr(response)}

    def _redact_thought_signatures(self, payload: Any) -> Any:
        if isinstance(payload, dict):
            if payload.get("type") == "reasoning.encrypted":
                return None

            redacted: dict[str, Any] = {}
            for key, value in payload.items():
                if key == "thought_signature":
                    redacted[key] = "(hidden for brevity)"
                else:
                    sanitized_value = self._redact_thought_signatures(value)
                    if sanitized_value is not None:
                        redacted[key] = sanitized_value
            return redacted
        if isinstance(payload, list):
            sanitized_items = []
            for item in payload:
                sanitized = self._redact_thought_signatures(item)
                if sanitized is not None:
                    sanitized_items.append(sanitized)
            return sanitized_items
        return payload

    def _log_turn(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response: Any | None,
        parsed_args: dict[str, Any] | None,
        action: GameAction,
        error: str | None,
        memory_after: str,
        request_payload: dict[str, Any] | None,
    ) -> None:
        if not self._log_conversation:
            return

        if self._chat_log_path is None:
            self._chat_log_path = self._resolve_chat_log_path()
            if not self._conversation_log_announced:
                logger.info(
                    "SimpleMemoryCarryover chat log: %s",
                    self._chat_log_path,
                )
                self._conversation_log_announced = True

        chat_turn = self._format_chat_turn(
            turn_index=self.action_counter,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            parsed_args=parsed_args,
            response=response,
            action=action,
            error=error,
            memory_after=memory_after,
            request_payload=request_payload,
        )
        with open(self._chat_log_path, "a", encoding="utf-8") as f:
            f.write(chat_turn)
