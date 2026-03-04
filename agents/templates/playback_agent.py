import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

from arcengine import FrameData, GameAction

from ..agent import Agent

logger = logging.getLogger()


class PlaybackAgent(Agent):
    """An agent that plays back from a recorded sessions data/playback_agent folder"""

    MAX_ACTIONS = 1000000
    PLAYBACK_FPS = 30
    recorded_actions: list[dict[str, Any]]

    def get_recording_path(self) -> Optional[str]:
        recordings_dir = Path(
            os.environ.get("PLAYBACK_AGENT_RECORDINGS", "data/playback_agent")
        )

        # try full game_id
        path = next(recordings_dir.glob(f"{self.game_id}*.jsonl"), None)
        if path:
            return str(path)

        # fallback to prefix
        prefix = self.game_id.split("-")[0]
        path = next(recordings_dir.glob(f"{prefix}*.jsonl"), None)

        return str(path) if path else None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        recording_path = self.get_recording_path()
        logger.info(f"Recording path: {recording_path}")

        if not recording_path:
            logger.error(f"No recording found for game_id={self.game_id}")
            self.recorded_actions = []
            return

        try:
            # Load + filter once
            self.recorded_actions = self.load_recorded_actions(recording_path)
            logger.info(
                f"Loaded {len(self.recorded_actions)} actions from recording {recording_path}"
            )
        except Exception as e:
            logger.exception(f"Failed to load recording {recording_path}: {e}")
            self.recorded_actions = []

    def load_recorded_actions(self, recording_path: str) -> list[dict[str, Any]]:
        """
        Load events from a JSONL recording file and return only events
        """
        actions: list[dict[str, Any]] = []

        with open(recording_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                event = json.loads(line)
                data = event.get("data")
                if isinstance(data, dict) and "action_input" in data:
                    actions.append(event)

        return actions

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return bool(self.action_counter >= len(self.recorded_actions))

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
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

        target_frame_time = 1.0 / getattr(self, "PLAYBACK_FPS", 5)
        elapsed_time = time.time() - loop_start_time
        sleep_time = max(0, target_frame_time - elapsed_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

        return action