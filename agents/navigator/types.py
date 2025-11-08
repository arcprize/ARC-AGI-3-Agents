"""Shared type definitions and helpers for navigator modules."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, NewType, Optional, Sequence, Tuple

from ..structs import GameAction

logger = logging.getLogger(__name__)

FrameHash = NewType("FrameHash", str)

Frame = list[list[int]] # 64x64


@dataclass(frozen=True)
class EnergyHudMeasurement:
    """Structured representation of an energy HUD value.

    The energy is represented as a non-negative integer `value` with an
    optional `capacity` upper bound and the HUD mask geometry. If the HUD spans
    disjoint regions, include every rectangle that belongs to the HUD.
    """

    value: int
    capacity: int
    mask: Sequence[Tuple[int, int, int, int]]
    """Rectangles describing every pixel that belongs to the energy HUD.

    This mask is considered the canonical HUD geometry for the current frame and
    is reused for frame hashing so that HUD redraws do not affect state identity.
    """

    @property
    def empty_blocks(self) -> int:
        return max(self.capacity - self.value, 0)

    @property
    def fill_ratio(self) -> float:
        if self.capacity == 0:
            return 0.0
        return self.value / self.capacity

    # Backwards compatibility: expose `filled_blocks` as an alias for `value`.
    @property
    def filled_blocks(self) -> int:
        return self.value


@dataclass
class TransitionMap:
    """Outgoing transitions observed from a single frame hash."""

    transitions: Dict[GameAction, FrameHash] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, str]:
        return {action.name: str(target) for action, target in self.transitions.items()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, str]) -> "TransitionMap":
        transitions: Dict[GameAction, FrameHash] = {}
        for action_name, raw in payload.items():
            if action_name not in GameAction.__members__:
                logger.warning("Skipping unknown action in memory payload: %s", action_name)
                continue
            try:
                transitions[GameAction[action_name]] = FrameHash(str(raw))
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
    level_terminal_states: Dict[int, FrameHash] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Dict[str, object]]:
        return {
            "state_graph": {
                str(state_hash): record.to_dict()
                for state_hash, record in self.state_graph.items()
            },
            "terminal_states": {
                str(level): str(state_hash)
                for level, state_hash in self.level_terminal_states.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "Memory":
        memory = cls()
        state_graph_payload = payload.get("state_graph", payload)
        if isinstance(state_graph_payload, Mapping):
            for hash_str, transitions in state_graph_payload.items():
                try:
                    state_hash = FrameHash(str(hash_str))
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
        terminal_payload = payload.get("terminal_states", {})
        if isinstance(terminal_payload, Mapping):
            for level_str, raw_state in terminal_payload.items():
                try:
                    level = int(level_str)
                    memory.level_terminal_states[level] = FrameHash(str(raw_state))
                except (TypeError, ValueError):
                    logger.warning(
                        "Skipping invalid terminal entry %s -> %s",
                        level_str,
                        raw_state,
                    )
                    continue
        return memory


def load_memory(path: Path, *, logger_prefix: Optional[str] = None) -> Memory:
    if not path.exists():
        return Memory()
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        logger.warning(
            "%s could not load memory file %s; starting fresh",
            logger_prefix or "navigator",
            path,
        )
        return Memory()
    if isinstance(raw, dict):
        return Memory.from_dict(raw)
    logger.warning(
        "%s memory file %s was not a dict; starting fresh",
        logger_prefix or "navigator",
        path,
    )
    return Memory()


def save_memory(memory: Memory, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(memory.to_dict(), indent=2))


def persist_metrics(
    *,
    recorder: Any,
    game_id: str,
    agent_name: str,
    known_states_total: int,
    energy_capacity: Optional[int],
) -> None:
    if not recorder or not getattr(recorder, "filename", None):
        return
    target_path = Path(recorder.filename).with_suffix(".tracking.json")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "game_id": game_id,
        "agent": agent_name,
        "known_states_total": known_states_total,
        "energy_capacity": energy_capacity,
    }
    with target_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
