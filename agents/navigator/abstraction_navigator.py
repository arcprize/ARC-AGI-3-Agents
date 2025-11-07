from __future__ import annotations

"""AbstractionNavigator: iterate fast, capture insights.

Update cycle:
  1. Run `uv run main.py --agent=abstractionnavigator --game=<id>`.
  2. Inspect the logs + `recordings/*.tracking.json` for new behaviour.
  3. Edit abstractions or heuristics in this file (generic section first, game block last).
  4. Re-run, rinse, repeat—commit once a new abstraction proves useful.
"""

import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from ..agent import Agent
from ..structs import FrameData, GameAction, GameState
from .nfr_planner import NearFrontierPlanner
from .types import (
    EnergyHudMeasurement,
    Frame,
    FrameHash,
    Memory,
    TransitionMap,
    load_memory,
    persist_metrics,
    save_memory,
)

logger = logging.getLogger()

MEMORY_PATH = Path(__file__).resolve().parent / "memory" / "memory.json"

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


AbstractionDetector = Callable[[Frame], Optional[Any]]

MaskRect = tuple[int, int, int, int]
FrameMask = Sequence[MaskRect]

# Game-specific detectors extend this list near the bottom of the file.
USER_ABSTRACTIONS: list[tuple[str, AbstractionDetector]] = []


def hash_frame(frame: Frame, *, mask: Optional[FrameMask] = None) -> FrameHash:
    """Return a hash of the frame with the masked regions zeroed out."""

    if not frame:
        return FrameHash(0)

    if mask:
        height = len(frame)
        mask_ranges_by_row: dict[int, list[tuple[int, int]]] = {}
        for raw_y0, raw_y1, raw_x0, raw_x1 in mask:
            if height == 0:
                break
            y0 = max(0, min(height - 1, raw_y0))
            y1 = max(0, min(height - 1, raw_y1))
            if y1 < y0:
                continue
            for y in range(y0, y1 + 1):
                row_len = len(frame[y])
                if row_len == 0:
                    continue
                x0 = max(0, min(row_len - 1, raw_x0))
                x1 = max(0, min(row_len - 1, raw_x1))
                if x1 < x0:
                    continue
                mask_ranges_by_row.setdefault(y, []).append((x0, x1))
    else:
        mask_ranges_by_row = {}

    normalized: list[tuple[int, ...]] = []
    for y, row in enumerate(frame):
        ranges = mask_ranges_by_row.get(y)
        normalized_row = []
        if ranges:
            for x, cell in enumerate(row):
                if any(x0 <= x <= x1 for x0, x1 in ranges):
                    normalized_row.append(0)
                else:
                    normalized_row.append(cell)
        else:
            normalized_row.extend(row)
        normalized.append(tuple(normalized_row))

    return FrameHash(hash(tuple(normalized)))


@dataclass(frozen=True)
class NavigatorSnapshot:
    """Immutable view of a single observation step."""

    frame: FrameData
    abstraction: FrameAbstraction
    frame_hash: FrameHash
    score: int
    level: int
    energy: Optional[EnergyHudMeasurement]
    energy_capacity: Optional[int]
    level_start_state: FrameHash
    available_actions: list[GameAction]
    game_state: GameState


class AbstractionNavigator(Agent):
    """Exploration-focused agent that will grow into an abstraction navigator."""

    MAX_ACTIONS = 100
    ARROW_ACTIONS = [
        GameAction.ACTION1,  # Up
        GameAction.ACTION2,  # Down
        GameAction.ACTION3,  # Left
        GameAction.ACTION4,  # Right,
    ]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1_000_000) ^ hash(self.game_id)
        self.rng = random.Random(seed)
        self.last_action: Optional[GameAction] = None

        self.memory: Memory = load_memory(MEMORY_PATH, logger_prefix=self.game_id)
        self._nfr_planner = NearFrontierPlanner(
            arrow_actions=self.ARROW_ACTIONS,
            state_graph=self.memory.state_graph,
        )
        self._snapshots: deque[NavigatorSnapshot] = deque(maxlen=3)

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return any(
            [
                latest_frame.state is GameState.WIN,
                latest_frame.state is GameState.GAME_OVER,
                self.action_counter >= self.MAX_ACTIONS,
            ]
        )

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._reset_tracking()
            action = GameAction.RESET
            action.reasoning = "resetting before exploration"
            return action

        # Package raw frame into a snapshot with derived abstractions/state info.
        snapshot = self._create_navigator_snapshot(latest_frame)

        terminal_target = self.memory.level_terminal_states.get(snapshot.level)
        nfr_action = self._nfr_planner.next_action(
            current_state=snapshot.frame_hash,
            available_actions=snapshot.available_actions,
            level_start_state=snapshot.level_start_state,
            target_state=terminal_target,
        )
        if nfr_action is None:
            action = GameAction.RESET
            action.reasoning = "nfr-fallback-reset"
            self.last_action = None
            return action

        if nfr_action in self.ARROW_ACTIONS:
            self.last_action = nfr_action
        elif nfr_action is GameAction.RESET:
            self.last_action = None
        return nfr_action

    def _reset_tracking(self) -> None:
        self.last_action = None
        self._snapshots.clear()

    def _create_navigator_snapshot(self, frame_data: FrameData) -> NavigatorSnapshot:

        prev_snapshot = self._snapshots[-1] if self._snapshots else None
        prev_prev_snapshot = self._snapshots[-2] if len(self._snapshots) >= 2 else None
        frame = frame_data.frame[0]

        capacity_hint = prev_snapshot.energy_capacity if prev_snapshot else None
        energy_measurement = measure_energy_blocks(frame, capacity_hint=capacity_hint)
        frame_hash = hash_frame(frame, mask=FRAME_HASH_MASK)
        abstraction = FrameAbstraction(frame_hash=frame_hash, frame=frame)
        if energy_measurement is not None:
            abstraction.add("energy", energy_measurement)
        energy_capacity = energy_measurement.capacity if energy_measurement else None

        for name, detector in USER_ABSTRACTIONS:
            try:
                result = detector(frame)
            except Exception:
                logger.exception(
                    "%s abstraction %s failed",
                    self.game_id,
                    getattr(detector, "__name__", repr(detector)),
                )
                continue
            if result is not None:
                abstraction.add(name, result)

        level, level_start_state = self._infer_level(
            prev_snapshot,
            prev_prev_snapshot,
            score=frame_data.score,
            frame_hash=frame_hash,
        )

        snapshot = NavigatorSnapshot(
            frame=frame_data,
            abstraction=abstraction,
            frame_hash=frame_hash,
            score=frame_data.score,
            level=level,
            energy=energy_measurement,
            energy_capacity=energy_capacity,
            level_start_state=level_start_state,
            available_actions=frame_data.available_actions,
            game_state=frame_data.state,
        )

        self._snapshots.append(snapshot)
        self._update_level_state(prev_snapshot, snapshot)
        self._track_state_graph(prev_snapshot, snapshot)
        return snapshot

    def cleanup(self, scorecard: Optional[Any] = None) -> None:
        known_states_total = len(self.memory.state_graph)
        logger.info(
            "%s known states total=%d",
            self.game_id,
            known_states_total,
        )

        save_memory(self.memory, MEMORY_PATH)

        persist_metrics(
            recorder=getattr(self, "recorder", None),
            game_id=self.game_id,
            agent_name=self.name,
            known_states_total=known_states_total,
            energy_capacity=self._snapshots[-1].energy_capacity if self._snapshots else None,
        )

        super().cleanup(scorecard)

    def _update_level_state(
        self,
        prev_snapshot: Optional[NavigatorSnapshot],
        snapshot: NavigatorSnapshot,
    ) -> None:
        level_changed = (
            prev_snapshot is not None and snapshot.level != prev_snapshot.level
        )

        if level_changed and prev_snapshot is not None:
            self._handle_level_change(prev_snapshot, snapshot)

    def _track_state_graph(
        self,
        prev_snapshot: Optional[NavigatorSnapshot],
        snapshot: NavigatorSnapshot,
    ) -> None:
        self._record_state_visit(snapshot.frame_hash)

        previous_state_hash = prev_snapshot.frame_hash if prev_snapshot else None
        if (
            previous_state_hash is None
            or not self.last_action
            or self.last_action is GameAction.RESET
        ):
            return
        self._record_state_transition(
            previous_state_hash, self.last_action, snapshot.frame_hash
        )

    def _record_state_visit(self, frame_hash: FrameHash) -> None:
        state_graph = self.memory.state_graph
        record = state_graph.get(frame_hash)
        if record is None:
            record = TransitionMap()
            state_graph[frame_hash] = record

    def _record_state_transition(
        self,
        previous_hash: FrameHash,
        action: GameAction,
        next_hash: FrameHash,
    ) -> None:
        state_graph = self.memory.state_graph
        transition_map = state_graph.get(previous_hash)
        if transition_map is None:
            transition_map = TransitionMap()
            state_graph[previous_hash] = transition_map

        if next_hash not in state_graph:
            state_graph[next_hash] = TransitionMap()
        existing = transition_map.transitions.get(action)
        if existing is None:
            transition_map.transitions[action] = next_hash
            return
        if existing != next_hash:
            message = (
                f"Non-deterministic transition: state={previous_hash}, "
                f"action={action.name}, existing_target={existing}, new_target={next_hash}"
            )
            logger.error(message)
            raise ValueError(message)

    def _handle_level_change(
        self, prev_snapshot: NavigatorSnapshot, snapshot: NavigatorSnapshot
    ) -> None:
        level_completed = prev_snapshot.level
        terminal_hash = prev_snapshot.frame_hash
        self.memory.level_terminal_states[level_completed] = terminal_hash
        logger.info(
            "%s level advanced to %d at step %d",
            self.game_id,
            snapshot.level,
            self.action_counter,
        )
        logger.info(
            "%s level start confirmed at hash=%s",
            self.game_id,
            snapshot.level_start_state,
        )
        logger.info(
            "%s recorded terminal state for level %d: %s",
            self.game_id,
            level_completed,
            terminal_hash,
        )

    def _infer_level(
        self,
        prev_snapshot: Optional[NavigatorSnapshot],
        prev_prev_snapshot: Optional[NavigatorSnapshot],
        *,
        score: int,
        frame_hash: FrameHash,
    ) -> tuple[int, FrameHash]:
        if prev_snapshot is None:
            return 1, frame_hash

        level = prev_snapshot.level
        level_start_state = prev_snapshot.level_start_state
        if (
            prev_prev_snapshot is not None
            and prev_snapshot.score != prev_prev_snapshot.score
            and score == prev_snapshot.score
            and frame_hash != prev_snapshot.frame_hash
        ):
            level = prev_snapshot.level + 1
            level_start_state = frame_hash
        return level, level_start_state


# ---------------------------------------------------------------------------
# Game-specific abstractions (ls20)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundingBox:
    min_y: int
    max_y: int
    min_x: int
    max_x: int

    @property
    def height(self) -> int:
        return self.max_y - self.min_y + 1

    @property
    def width(self) -> int:
        return self.max_x - self.min_x + 1


@dataclass(frozen=True)
class PlayerDetection:
    """Detect the ls20 avatar sprite footprint."""

    center: tuple[float, float]
    pixel_count: int
    bbox: BoundingBox


def detect_player(frame_cells: Frame) -> Optional[PlayerDetection]:
    positions: list[tuple[int, int]] = []
    for y, row in enumerate(frame_cells):
        for x, cell in enumerate(row):
            if cell == 12:
                positions.append((y, x))

    if not positions or len(positions) > 256:
        return None

    min_y = min(y for y, _ in positions)
    max_y = max(y for y, _ in positions)
    min_x = min(x for _, x in positions)
    max_x = max(x for _, x in positions)

    height = max_y - min_y + 1
    width = max_x - min_x + 1
    if height > 16 or width > 16:
        return None

    total_y = sum(y for y, _ in positions)
    total_x = sum(x for _, x in positions)
    count = len(positions)
    center = (total_y / count, total_x / count)

    bbox = BoundingBox(
        min_y=min_y,
        max_y=max_y,
        min_x=min_x,
        max_x=max_x,
    )
    return PlayerDetection(center=center, pixel_count=count, bbox=bbox)


ENERGY_HUD_MASK: tuple[MaskRect, ...] = ((1, 2, 2, 45),)
FRAME_HASH_MASK: FrameMask = ENERGY_HUD_MASK


def measure_energy_blocks(
    frame: Frame, *, capacity_hint: Optional[int] = None
) -> Optional[EnergyHudMeasurement]:
    row_index = 2
    if len(frame) <= row_index or not frame[row_index]:
        return None

    row = frame[row_index]
    _, _, x0, x1 = ENERGY_HUD_MASK[0]
    row_len = len(row)
    if row_len <= x0:
        return None
    upper_x = min(row_len - 1, x1)

    blocks: list[int] = []
    for x in range(x0, upper_x + 1, 2):
        value = row[x]
        if value in (3, 15):
            blocks.append(value)
        elif blocks:
            break

    total = len(blocks)
    if total < 6:
        values = {row[x] for x in range(x0, upper_x + 1, 2)}
        if values == {8} and capacity_hint:
            return EnergyHudMeasurement(
                filled_blocks=0,
                capacity=capacity_hint,
                rect=ENERGY_HUD_MASK[0],
            )
        return None

    if any(v not in (3, 15) for v in blocks):
        return None

    filled = sum(1 for v in blocks if v == 15)
    return EnergyHudMeasurement(
        filled_blocks=filled,
        capacity=total,
        rect=ENERGY_HUD_MASK[0],
    )


USER_ABSTRACTIONS.extend(
    [
        # Add new abstractions here: provide a (name, detector) pair. Detectors must
        # accept the frame and return either a structured result or None.
        ("player", detect_player),
    ]
)
