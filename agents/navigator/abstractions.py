"""Shared scaffolding for defining grid abstractions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .types import Frame, FrameHash


AbstractionDetector = Callable[[Frame], Optional[Any]]


@dataclass
class FrameAbstraction:
    """In-memory snapshot of a frame enhanced with higher-level abstractions."""

    frame_hash: FrameHash
    frame: Frame
    abstractions: dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, value: Any) -> None:
        self.abstractions[name] = value

    def get(self, name: str) -> Any:
        return self.abstractions.get(name)


USER_ABSTRACTIONS: list[tuple[str, AbstractionDetector]] = []


def register_abstraction(name: str, detector: AbstractionDetector) -> None:
    USER_ABSTRACTIONS.append((name, detector))

