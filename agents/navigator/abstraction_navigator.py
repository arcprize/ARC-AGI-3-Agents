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
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from ..agent import Agent
from ..structs import FrameData, GameAction, GameState
from .nfr_planner import NearFrontierPlanner
from .types import (
    EnergyHudMeasurement,
    FrameHash,
    Grid,
    LevelEvent,
    Memory,
    TransitionMap,
    load_memory,
    persist_metrics,
    save_memory,
)

logger = logging.getLogger()

MEMORY_PATH = Path(__file__).resolve().parent / "memory" / "memory.json"

class NavigatorMode(str, Enum):
    """Operating modes for the abstraction navigator."""

    EXPLORE = "explore"
    TODO = "todo"


@dataclass
class FrameAbstraction:
    """In-memory snapshot of a frame enhanced with higher-level abstractions."""

    frame_hash: FrameHash
    grid: Grid
    abstractions: dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, value: Any) -> None:
        self.abstractions[name] = value

    def get(self, name: str) -> Any:
        return self.abstractions.get(name)


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
    """Example of a user-defined abstraction."""

    center: tuple[float, float]
    pixel_count: int
    bbox: BoundingBox


GridAbstraction = Callable[[Grid], Optional[Any]]


def detect_player(grid: Grid) -> Optional[PlayerDetection]:
    positions: list[tuple[int, int]] = []
    for y, row in enumerate(grid):
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


ENERGY_HUD_RECT = (1, 2, 2, 45)  # (y0, y1, x0, x1); tuned for ls20 energy bar


def measure_energy_blocks(
    grid: Grid, *, capacity_hint: Optional[int] = None
) -> Optional[EnergyHudMeasurement]:
    row_index = 2
    if len(grid) <= row_index or not grid[row_index]:
        return None
    row = grid[row_index]
    _, _, x0, x1 = ENERGY_HUD_RECT
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
                rect=ENERGY_HUD_RECT,
            )
        return None

    if any(v not in (3, 15) for v in blocks):
        return None

    filled = sum(1 for v in blocks if v == 15)
    return EnergyHudMeasurement(
        filled_blocks=filled,
        capacity=total,
        rect=ENERGY_HUD_RECT,
    )


USER_ABSTRACTIONS: list[tuple[str, GridAbstraction]] = [
    # Add new abstractions here: provide a (name, detector) pair. Detectors must
    # accept the grid and return either a structured result or None.
    ("player", detect_player),
]


@dataclass(frozen=True)
class NavigatorSnapshot:
    """Immutable view of a single observation step."""

    frame: FrameData
    abstraction: FrameAbstraction
    frame_hash: FrameHash
    score: int
    energy: Optional[EnergyHudMeasurement]
    available_actions: tuple[GameAction, ...]
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
        mode_value = kwargs.pop("mode", NavigatorMode.EXPLORE)
        self.mode = NavigatorMode(mode_value)
        super().__init__(*args, **kwargs)
        self._level_start_state: Optional[FrameHash] = None
        seed = int(time.time() * 1_000_000) ^ hash(self.game_id)
        self.rng = random.Random(seed)
        self.last_action: Optional[GameAction] = None

        self.memory: Memory = load_memory(MEMORY_PATH, logger_prefix=self.game_id)
        self.unique_states_this_run: set[FrameHash] = set()
        self._nfr_planner = NearFrontierPlanner(
            arrow_actions=self.ARROW_ACTIONS,
            state_graph=self.memory.state_graph,
        )
        self._snapshot: Optional[NavigatorSnapshot] = None
        self._prev_snapshot: Optional[NavigatorSnapshot] = None
        self._pending_level_score: Optional[int] = None
        self._pending_level_hash: Optional[FrameHash] = None

        self.energy_capacity: Optional[int] = None
        self.current_level = 0
        self.level_events: list[LevelEvent] = []

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

        if self.mode is NavigatorMode.TODO:
            logger.info("%s TODO mode: planning logic not yet implemented", self.game_id)
            raise NotImplementedError("Navigator TODO mode is a placeholder")

        # Observe and update discovered graph / level / HUD.
        self._observe(latest_frame)

        snapshot = self._snapshot
        if snapshot is None:
            action = GameAction.RESET
            action.reasoning = "no-state-reset"
            self.last_action = None
            return action

        current_state = snapshot.frame_hash
        available_actions = list(snapshot.available_actions)

        if self._level_start_state is None:
            self._level_start_state = current_state

        nfr_action = self._nfr_planner.next_action(
            current_state=current_state,
            available_actions=available_actions,
            level_start_state=self._level_start_state,
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
        self.unique_states_this_run.clear()
        self.energy_capacity = None
        self.current_level = 0
        self.level_events = []
        self._level_start_state = None
        self._snapshot = None
        self._prev_snapshot = None
        self._pending_level_score = None
        self._pending_level_hash = None

    def _observe(self, frame: FrameData) -> None:
        if frame.is_empty() or not frame.frame:
            return

        prev_snapshot = self._snapshot

        grid = frame.frame[0]
        energy_measurement = measure_energy_blocks(
            grid, capacity_hint=self.energy_capacity
        )
        frame_hash = self._hash_grid(grid)
        abstraction = FrameAbstraction(frame_hash=frame_hash, grid=grid)
        if energy_measurement is not None:
            self.energy_capacity = energy_measurement.capacity
            abstraction.add("energy", energy_measurement)

        for name, detector in USER_ABSTRACTIONS:
            try:
                result = detector(grid)
            except Exception:
                logger.exception(
                    "%s abstraction %s failed",
                    self.game_id,
                    getattr(detector, "__name__", repr(detector)),
                )
                continue
            if result is not None:
                abstraction.add(name, result)

        snapshot = NavigatorSnapshot(
            frame=frame,
            abstraction=abstraction,
            frame_hash=frame_hash,
            score=frame.score,
            energy=energy_measurement,
            available_actions=tuple(frame.available_actions or ()),
            game_state=frame.state,
        )

        self._prev_snapshot = prev_snapshot
        self._snapshot = snapshot

        if self._level_start_state is None:
            self._level_start_state = snapshot.frame_hash

        level_changed = (
            prev_snapshot is not None and snapshot.score != prev_snapshot.score
        )

        if level_changed:
            self._handle_level_change(snapshot)
        elif (
            self._pending_level_score is not None
            and snapshot.score == self._pending_level_score
            and snapshot.frame_hash != self._pending_level_hash
        ):
            self._level_start_state = snapshot.frame_hash
            self._pending_level_score = None
            self._pending_level_hash = None
            logger.info("%s level start confirmed at hash=%s", self.game_id, frame_hash)
        elif (
            self._pending_level_score is not None
            and snapshot.score != self._pending_level_score
        ):
            # Score shifted again before confirmation; drop the pending record.
            self._pending_level_score = None
            self._pending_level_hash = None

        abstraction.add("level", snapshot.score)
        self.current_level = snapshot.score

        self._record_state_visit(snapshot.frame_hash)

        previous_state_hash = prev_snapshot.frame_hash if prev_snapshot else None
        if (
            previous_state_hash is not None
            and self.last_action
            and self.last_action is not GameAction.RESET
        ):
            self._record_state_transition(
                previous_state_hash, self.last_action, snapshot.frame_hash
            )

    def cleanup(self, scorecard: Optional[Any] = None) -> None:
        states_visited_run = len(self.unique_states_this_run)
        known_states_total = len(self.memory.state_graph)
        logger.info(
            "%s states visited this run: %d (known total=%d)",
            self.game_id,
            states_visited_run,
            known_states_total,
        )

        if self.level_events:
            logger.info(
                "%s level transitions: %s",
                self.game_id,
                self.level_events,
            )

        save_memory(self.memory, MEMORY_PATH)

        persist_metrics(
            recorder=getattr(self, "recorder", None),
            game_id=self.game_id,
            agent_name=self.name,
            states_visited_run=states_visited_run,
            known_states_total=known_states_total,
            energy_capacity=self.energy_capacity,
            level_events=self.level_events,
        )

        super().cleanup(scorecard)

    def _record_state_visit(self, frame_hash: FrameHash) -> None:
        state_graph = self.memory.state_graph
        record = state_graph.get(frame_hash)
        if record is None:
            record = TransitionMap()
            state_graph[frame_hash] = record
        self.unique_states_this_run.add(frame_hash)

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

    def _hash_grid(self, grid: Grid) -> FrameHash:
        """
        Return a hash of the grid ignoring the HUD rectangle.

        The HUD is treated as adversarial noise: masking it collapses states that only
        differ by score rendering into a single identifier, keeping the graph stable.
        """
        if not grid:
            return FrameHash(0)
        normalized = []
        raw_y0, raw_y1, raw_x0, raw_x1 = ENERGY_HUD_RECT
        max_y = len(grid) - 1
        if max_y >= 0:
            y0 = max(0, min(max_y, raw_y0))
            y1 = max(0, min(max_y, raw_y1))
        else:
            y0 = y1 = 0
        for y, row in enumerate(grid):
            normalized_row = []
            row_len = len(row)
            if row_len > 0:
                x0 = max(0, min(row_len - 1, raw_x0))
                x1 = max(0, min(row_len - 1, raw_x1))
            else:
                x0 = 0
                x1 = -1
            for x, cell in enumerate(row):
                if y0 <= y <= y1 and x0 <= x <= x1:
                    normalized_row.append(0)
                else:
                    normalized_row.append(cell)
            normalized.append(tuple(normalized_row))
        return FrameHash(hash(tuple(normalized)))


    def _handle_level_change(self, snapshot: NavigatorSnapshot) -> None:
        self.current_level = snapshot.score
        self._pending_level_score = snapshot.score
        self._pending_level_hash = snapshot.frame_hash

        event: LevelEvent = {
            "level": self.current_level,
            "step": self.action_counter,
            "state_hash": snapshot.frame_hash,
            "energy": snapshot.energy.filled_blocks if snapshot.energy else 0,
            "timestamp": time.time(),
        }

        self.level_events.append(event)
        snapshot.abstraction.add("level_transition", event)
        logger.info(
            "%s level advanced to %d at step %d",
            self.game_id,
            self.current_level,
            self.action_counter,
        )
