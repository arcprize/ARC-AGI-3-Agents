import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from arc_agi import EnvironmentWrapper
from arc_agi.scorecard import EnvironmentScorecard
from arcengine import FrameData, FrameDataRaw, GameAction, GameState
from pydantic import ValidationError

from .recorder import Recorder
from .tracing import trace_agent_session

logger = logging.getLogger()


class Agent(ABC):
    """Interface for an agent that plays one ARC-AGI-3 game."""

    MAX_ACTIONS: int = 80  # to avoid looping forever if agent doesnt exit
    ROOT_URL: str

    action_counter: int = 0
    REASONING_MAX_BYTES: int = 16 * 1024
    REASONING_PREVIEW_CHARS: int = 1200

    timer: float = 0
    agent_name: str
    card_id: str
    game_id: str
    guid: str
    frames: list[FrameData]

    recorder: Recorder
    headers: dict[str, str]
    arc_env: EnvironmentWrapper

    # AgentOps tracing attributes
    trace: Any = None
    tags: list[str]

    def __init__(
        self,
        card_id: str,
        game_id: str,
        agent_name: str,
        ROOT_URL: str,
        record: bool,
        arc_env: EnvironmentWrapper,
        tags: Optional[list[str]] = None,
    ) -> None:
        self.ROOT_URL = ROOT_URL
        self.card_id = card_id
        self.game_id = game_id
        self.guid = ""
        self.agent_name = agent_name
        self.tags = tags or []
        self.frames = [FrameData(levels_completed=0)]
        self._cleanup = True
        if record:
            self.start_recording()
        self.headers = {
            "X-API-Key": os.getenv("ARC_API_KEY", ""),
            "Accept": "application/json",
        }
        self.arc_env = arc_env

    @trace_agent_session
    def main(self) -> None:
        """The main agent loop. Play the game_id until finished, then exits."""
        self.timer = time.time()
        while (
            not self.is_done(self.frames, self.frames[-1])
            and self.action_counter <= self.MAX_ACTIONS
        ):
            action = self.choose_action(
                self.frames,
                self._convert_raw_frame_data(
                    self.arc_env.observation_space if self.arc_env else None
                ),
            )
            if frame := self.take_action(action):
                self.append_frame(frame)
                logger.info(
                    f"{self.game_id} - {action.name}: count {self.action_counter}, levels completed {frame.levels_completed}, avg fps {self.fps})"
                )
            self.action_counter += 1

        self.cleanup()

    @property
    def state(self) -> GameState:
        return self.frames[-1].state

    @property
    def levels_completed(self) -> int:
        return self.frames[-1].levels_completed  # type: ignore[no-any-return]

    @property
    def seconds(self) -> float:
        return (time.time() - self.timer) * 100 // 1 / 100

    @property
    def fps(self) -> float:
        if self.action_counter == 0:
            return 0.0
        elapsed_time = max(self.seconds, 0.1)
        return round(self.action_counter / elapsed_time, 2)

    @property
    def is_playback(self) -> bool:
        return type(self) is Playback

    @property
    def name(self) -> str:
        n = self.__class__.__name__.lower()
        return f"{self.game_id}.{n}"

    def start_recording(self) -> None:
        filename = self.agent_name if self.is_playback else None
        self.recorder = Recorder(prefix=self.name, filename=filename)
        logger.info(
            f"created new recording for {self.name} into {self.recorder.filename}"
        )

    def append_frame(self, frame: FrameData) -> None:
        self.frames.append(frame)
        if frame.guid:
            self.guid = frame.guid
        if hasattr(self, "recorder") and not self.is_playback:
            self.recorder.record(json.loads(frame.model_dump_json()))

    def do_action_request(self, action: GameAction) -> FrameData:
        data = action.action_data.model_dump()
        reasoning_payload = self._extract_reasoning_payload(action)
        if reasoning_payload is not None:
            encoded = self._encode_json_bytes(reasoning_payload)
            size = len(encoded) if encoded is not None else -1
            logger.debug(
                "Submitting %s with reasoning payload bytes=%d",
                action.name,
                size,
            )
        raw = self.arc_env.step(
            action,
            data=data,
            reasoning=reasoning_payload,
        )
        return self._convert_raw_frame_data(raw)

    def _extract_reasoning_payload(self, action: GameAction) -> Optional[dict[str, Any]]:
        """Normalize any attached action reasoning into the env step payload."""
        reasoning = getattr(action, "reasoning", None)
        if reasoning is None:
            return None
        if isinstance(reasoning, dict):
            payload: dict[str, Any] = reasoning
        elif isinstance(reasoning, str):
            payload = {"text": reasoning}
        else:
            payload = {"value": str(reasoning)}
        return self._compact_reasoning_payload(payload)

    def _compact_reasoning_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        compact_payload = self._trim_json_value(payload)
        encoded = self._encode_json_bytes(compact_payload)
        if encoded is not None and len(encoded) <= self.REASONING_MAX_BYTES:
            return compact_payload

        fallback: dict[str, Any] = {
            "truncated": True,
            "original_size_bytes": len(encoded) if encoded is not None else None,
        }
        for key in [
            "agent",
            "model",
            "action",
            "selected_action",
            "mode",
            "state_sig",
            "reasoning_tokens",
            "total_reasoning_tokens",
        ]:
            if key in compact_payload:
                fallback[key] = self._trim_json_value(compact_payload[key], max_depth=1)
        preview = str(compact_payload)
        fallback["preview"] = preview[: self.REASONING_PREVIEW_CHARS]

        encoded_fallback = self._encode_json_bytes(fallback)
        if encoded_fallback is None or len(encoded_fallback) > self.REASONING_MAX_BYTES:
            fallback = {
                "truncated": True,
                "preview": preview[:300],
            }
        logger.warning(
            "Reasoning payload exceeded %d bytes and was compacted for env.step",
            self.REASONING_MAX_BYTES,
        )
        return fallback

    def _trim_json_value(
        self,
        value: Any,
        max_depth: int = 3,
        max_items: int = 12,
        max_string: int = 600,
    ) -> Any:
        if max_depth <= 0:
            return str(value)[:max_string]
        if isinstance(value, dict):
            trimmed: dict[str, Any] = {}
            for idx, (key, item) in enumerate(value.items()):
                if idx >= max_items:
                    trimmed["..."] = f"truncated {len(value) - max_items} keys"
                    break
                trimmed[str(key)[:80]] = self._trim_json_value(
                    item,
                    max_depth=max_depth - 1,
                    max_items=max_items,
                    max_string=max_string,
                )
            return trimmed
        if isinstance(value, list):
            out = [
                self._trim_json_value(
                    item,
                    max_depth=max_depth - 1,
                    max_items=max_items,
                    max_string=max_string,
                )
                for item in value[:max_items]
            ]
            if len(value) > max_items:
                out.append(f"... truncated {len(value) - max_items} items")
            return out
        if isinstance(value, tuple):
            return self._trim_json_value(
                list(value),
                max_depth=max_depth,
                max_items=max_items,
                max_string=max_string,
            )
        if isinstance(value, str):
            return value[:max_string]
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        return str(value)[:max_string]

    def _encode_json_bytes(self, value: Any) -> Optional[bytes]:
        try:
            return json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode(
                "utf-8"
            )
        except (TypeError, ValueError):
            return None

    def _convert_raw_frame_data(self, raw: FrameDataRaw | None) -> FrameData:
        if raw is None:
            raise ValueError("Received None frame data from environment")
        out = FrameData(
            game_id=raw.game_id,
            frame=[arr.tolist() for arr in raw.frame],
            state=raw.state,
            levels_completed=raw.levels_completed,
            win_levels=raw.win_levels,
            action_input=raw.action_input,
            guid=raw.guid,
            full_reset=raw.full_reset,
            available_actions=raw.available_actions,
        )
        return out

    def take_action(self, action: GameAction) -> Optional[FrameData]:
        """Submits the specific action and gets the next frame."""
        frame_data = self.do_action_request(action)
        try:
            frame = FrameData.model_validate(frame_data)
        except ValidationError as e:
            logger.warning(f"Incoming frame data did not validate: {e}")
            return None
        return frame

    def cleanup(self, scorecard: Optional[EnvironmentScorecard] = None) -> None:
        """Called after main loop is finished."""
        if self._cleanup:
            self._cleanup = False  # only cleanup once per agent
            if hasattr(self, "recorder") and not self.is_playback:
                if scorecard:
                    self.recorder.record(scorecard.get(self.game_id))
                logger.info(
                    f"recording for {self.name} is available in {self.recorder.filename}"
                )
            if self.action_counter >= self.MAX_ACTIONS:
                logger.info(
                    f"Exiting: agent reached MAX_ACTIONS of {self.MAX_ACTIONS}, took {self.seconds} seconds ({self.fps} average fps)"
                )
            else:
                logger.info(
                    f"Finishing: agent took {self.action_counter} actions, took {self.seconds} seconds ({self.fps} average fps)"
                )

    @abstractmethod
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing or not."""
        raise NotImplementedError

    @abstractmethod
    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Choose which action the Agent should take, fill in any arguments, and return it."""
        raise NotImplementedError


class Playback(Agent):
    """An agent that plays back from a recorded session from another agent."""

    MAX_ACTIONS = 1000000
    PLAYBACK_FPS = 5

    recorded_actions: list[dict[str, Any]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.recorder = Recorder(
            prefix=Recorder.get_prefix(self.agent_name),
            guid=Recorder.get_guid(self.agent_name),
        )
        self.recorded_actions = []
        if self.agent_name in Recorder.list():
            try:
                self.recorded_actions = self.filter_actions()
                logger.info(
                    f"Loaded {len(self.recorded_actions)} actions from {self.agent_name}"
                )
            except Exception as e:
                logger.error(f"Failed to load recording {self.agent_name}: {e}")
                self.recorded_actions = []
        else:
            logger.warning(
                f"Recording {self.agent_name} not found in available recordings"
            )

    def filter_actions(self) -> list[dict[str, Any]]:
        return [
            a
            for a in self.recorder.get()
            if "data" in a and "action_input" in a["data"]
        ]

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return bool(self.action_counter >= len(self.recorded_actions))

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        loop_start_time = time.time()

        if self.action_counter >= len(self.recorded_actions):
            logger.warning(
                f"No more recorded actions available (counter: {self.action_counter}, total: {len(self.recorded_actions)})"
            )
            return GameAction.RESET

        recorded_data = self.recorded_actions[self.action_counter]["data"]
        action_input = recorded_data["action_input"]

        action = GameAction.from_id(action_input["id"])
        data = action_input["data"].copy()
        data["game_id"] = self.game_id
        action.set_data(data)
        if "reasoning" in action_input and action_input["reasoning"] is not None:
            action.reasoning = action_input["reasoning"]

        logger.debug(
            f"Playback action {self.action_counter}: {action.name} with data {data}"
        )

        target_frame_time = 1.0 / getattr(self, "PLAYBACK_FPS", 5)
        elapsed_time = time.time() - loop_start_time
        sleep_time = max(0, target_frame_time - elapsed_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

        return action

    def append_frame(self, frame: FrameData) -> None:
        # overwrite append_frame to not double record
        self.frames.append(frame)
        if frame.guid:
            self.guid = frame.guid
