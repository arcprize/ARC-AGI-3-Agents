from __future__ import annotations

from abc import ABC
from typing import Any, Optional

from .structs import FrameData, GameAction


class Agent(ABC):
    MAX_ACTIONS: int
    game_id: str
    action_counter: int
    frames: list[FrameData]

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool: ...

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction: ...

    def append_frame(self, frame: FrameData) -> None: ...

    @property
    def name(self) -> str: ...

    def cleanup(self, scorecard: Optional[Any] = ...) -> None: ...
