"""Shared type definitions for navigator modules."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Mapping, NewType, Tuple

from ..structs import GameAction

logger = logging.getLogger(__name__)

FrameHash = NewType("FrameHash", int)

Grid = list[list[int]]

@dataclass(frozen=True)
class EnergyHudMeasurement:
    """Structured representation of the HUD energy bar."""

    filled_blocks: int
    capacity: int
    rect: Tuple[int, int, int, int]

    @property
    def empty_blocks(self) -> int:
        return max(self.capacity - self.filled_blocks, 0)

    @property
    def fill_ratio(self) -> float:
        if self.capacity == 0:
            return 0.0
        return self.filled_blocks / self.capacity


@dataclass
class TransitionMap:
    """Outgoing transitions observed from a single frame hash."""

    transitions: Dict[GameAction, FrameHash] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, int]:
        return {action.name: int(target) for action, target in self.transitions.items()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, int]) -> "TransitionMap":
        transitions: Dict[GameAction, FrameHash] = {}
        for action_name, raw in payload.items():
            if action_name not in GameAction.__members__:
                logger.warning("Skipping unknown action in memory payload: %s", action_name)
                continue
            try:
                transitions[GameAction[action_name]] = FrameHash(int(raw))
            except (TypeError, ValueError):
                logger.warning(
                    "Skipping transition for action %s due to invalid target %s",
                    action_name,
                    raw,
                )
                continue
        return cls(transitions)


STATE_GRAPH = Dict[FrameHash, TransitionMap]


@dataclass
class Memory:
    """Persistent navigation memory retained across runs."""

    state_graph: STATE_GRAPH = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Dict[str, int]]:
        return {str(state_hash): record.to_dict() for state_hash, record in self.state_graph.items()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Mapping[str, int]]) -> "Memory":
        memory = cls()
        for hash_str, transitions in payload.items():
            try:
                state_hash = FrameHash(int(hash_str))
            except (TypeError, ValueError):
                logger.warning("Skipping invalid state id in memory payload: %s", hash_str)
                continue
            if not isinstance(transitions, Mapping):
                logger.warning(
                    "Skipping state %s because transitions map is missing or malformed",
                    hash_str,
                )
                continue
            memory.state_graph[state_hash] = TransitionMap.from_dict(transitions)
        return memory
