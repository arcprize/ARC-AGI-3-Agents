import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional
import threading
from arcengine import FrameData, GameAction
from ..agent import Agent

logger = logging.getLogger()


class PlaybackAgent(Agent):
    """Plays back recorded actions"""
    
    MAX_ACTIONS = 1000000
    ACTIONS_PER_SECOND = 5  
    
    _rate_lock = threading.Lock()
    _last_action_time = 0.0
    
    _recorded_actions: list[dict[str, Any]]
    _current_action_data: Optional[dict[str, Any]]
 
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.action_counter = 0
        self._current_action_data = None
        
        recording_path = self._get_recording_path()
        if not recording_path:
            logger.error(f"No recording found for game_id={self.game_id}")
            self._recorded_actions = []
            return
        
        try:
            self._recorded_actions = self._load_actions(recording_path)
            logger.info(f"Loaded {len(self._recorded_actions)} actions from {recording_path}")
        except Exception as e:
            logger.exception(f"Failed to load recording: {e}")
            self._recorded_actions = []
    
    def _get_recording_path(self) -> Optional[str]:
        recordings_dir = Path(os.environ.get("PLAYBACK_AGENT_RECORDINGS", "data/playback_agent"))
        path = next(recordings_dir.glob(f"{self.game_id}*.jsonl"), None)
        if path:
            return str(path)
        prefix = self.game_id.split("-")[0]
        path = next(recordings_dir.glob(f"{prefix}*.jsonl"), None)
        return str(path) if path else None
    
    def _load_actions(self, path: str) -> list[dict[str, Any]]:
        actions = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line := line.strip():
                    event = json.loads(line)
                    if isinstance(data := event.get("data"), dict) and "action_input" in data:
                        actions.append(event)
        return actions
    
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return self.action_counter >= len(self._recorded_actions)
    
    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        # extract action from recording
        recorded_data = self._recorded_actions[self.action_counter]["data"]
        action_input = recorded_data["action_input"]
        
        # store in instance variables
        self._current_action_data = action_input["data"].copy()
        if self._current_action_data:
            self._current_action_data["game_id"] = self.game_id
 
        # rate limiting wait until it's our turn
        with self._rate_lock:
            now = time.time()
            time_since_last = now - PlaybackAgent._last_action_time
            min_interval = 1.0 / self.ACTIONS_PER_SECOND
            
            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                time.sleep(sleep_time)
            
            PlaybackAgent._last_action_time = time.time()
        
        return GameAction.from_id(action_input["id"])
    
    def do_action_request(self, action: GameAction) -> FrameData:
        """Execute action using stored data."""
        data = self._current_action_data or {"game_id": self.game_id}
        raw = self.arc_env.step(action, data=data)
        return self._convert_raw_frame_data(raw)
    
    def append_frame(self, frame: FrameData) -> None:
        pass