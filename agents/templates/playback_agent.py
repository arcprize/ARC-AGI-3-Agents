import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional
import threading
from arcengine import FrameData, GameAction, SimpleAction
from ..agent import Agent

logger = logging.getLogger()

class PlaybackAgent(Agent):
    """An agent that plays back from a recorded sessions data/playback_agent folder"""
    MAX_ACTIONS = 1000000
    PLAYBACK_FPS = 10
    
    # lock to protect shared GameAction enum modifications
    _action_lock = threading.Lock()
    
    _recorded_actions: list[dict[str, Any]]
    _current_action_config: Optional[dict[str, Any]]
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.action_counter = 0
        self._current_action_config = None
        
        recording_path = self._get_recording_path()
        logger.info(f"Recording path: {recording_path}")
        
        if not recording_path:
            logger.error(f"No recording found for game_id={self.game_id}")
            self._recorded_actions = []
            return
        
        try:
            self._recorded_actions = self._load_recorded_actions(recording_path)
            logger.info(
                f"Loaded {len(self._recorded_actions)} actions from recording {recording_path}"
            )
        except Exception as e:
            logger.exception(f"Failed to load recording {recording_path}: {e}")
            self._recorded_actions = []
    
    def _get_recording_path(self) -> Optional[str]:
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
    
    def _load_recorded_actions(self, recording_path: str) -> list[dict[str, Any]]:
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
        return bool(self.action_counter >= len(self._recorded_actions))
    
    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        start_time = time.time()
        
        recorded_data = self._recorded_actions[self.action_counter]["data"]
        action_input = recorded_data["action_input"]
        
        # store data in instance variable (thread-safe per instance)
        self._current_action_config = {
            "action_id": action_input["id"],
            "data": action_input["data"].copy(),
        }
        self._current_action_config["data"]["game_id"] = self.game_id
        
        # return the action without modifying the shared enum
        action = GameAction.from_id(action_input["id"])
        
        target_frame_time = 1.0 / self.PLAYBACK_FPS
        elapsed_time = time.time() - start_time
        sleep_time = max(0, target_frame_time - elapsed_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        return action
    
    def take_action(self, action: GameAction) -> Optional[FrameData]:
        """Override to apply action configuration under lock before using it."""
        with self._action_lock:
            # Apply the stored configuration to the shared enum under lock
            if self._current_action_config:
                action.set_data(self._current_action_config["data"])
                
            return super().take_action(action)
    
    def append_frame(self, frame: FrameData) -> None:
        pass