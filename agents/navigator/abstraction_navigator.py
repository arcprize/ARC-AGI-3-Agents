"""AbstractionNavigator: iterate fast, capture insights.

Update cycle:
  1. Run `uv run main.py --agent=abstractionnavigator --game=<id>`.
  2. Inspect the logs + `recordings/*.tracking.json` for new behaviour.
  3. Edit abstractions or heuristics in this file (generic section first, game block last).
  4. Re-run, rinse, repeat—commit once a new abstraction proves useful.
"""

import json
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, TypedDict

from ..agent import Agent
from ..structs import FrameData, GameAction, GameState
from .nfr_planner import FrameHash, NearFrontierPlanner, PlannerContext, STATE_GRAPH, TransitionMap

logger = logging.getLogger()

MEMORY_PATH = Path(__file__).resolve().parent / "memory" / "state_graph.json"

# === Generic infrastructure ==================================================

class LevelEvent(TypedDict):
    level: int
    step: int
    state_hash: int
    energy: int
    timestamp: float


class NavigatorMode(str, Enum):
    """Operating modes for the abstraction navigator."""

    EXPLORE = "explore"
    TODO = "todo"


@dataclass
class FrameAbstraction:
    """In-memory snapshot of a frame enhanced with higher-level abstractions."""

    frame_hash: FrameHash
    grid: list[list[Any]]
    abstractions: dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, value: Any) -> None:
        self.abstractions[name] = value

    def get(self, name: str) -> Any:
        return self.abstractions.get(name)


@dataclass(frozen=True)
class NavigatorSnapshot:
    """Immutable view of a single observation step."""

    frame: FrameData
    abstraction: FrameAbstraction
    frame_hash: FrameHash
    score: int
    energy_blocks: Optional[int]
    energy_capacity: Optional[int]
    available_actions: tuple[GameAction, ...]
    game_state: GameState


@dataclass
class Memory:
    """Persistent navigation memories retained across runs."""

    state_graph: STATE_GRAPH = field(default_factory=dict)


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
        # --- NFR planning state ---
        self._level_start_state: Optional[FrameHash] = None  # s0 of current level
        seed = int(time.time() * 1_000_000) ^ hash(self.game_id)
        self.rng = random.Random(seed)
        self.last_action: Optional[GameAction] = None

        self.memory: Memory = self._load_memory()
        self.unique_states_this_run: set[FrameHash] = set()
        planner_context = PlannerContext(
            arrow_actions=self.ARROW_ACTIONS,
            state_graph=self.memory.state_graph,
        )
        self._nfr_planner = NearFrontierPlanner(planner_context)
        self._snapshot: Optional[NavigatorSnapshot] = None
        self._prev_snapshot: Optional[NavigatorSnapshot] = None
        self._pending_level_score: Optional[int] = None
        self._pending_level_hash: Optional[FrameHash] = None

        self.energy_capacity: Optional[int] = None
        self.current_level = 0
        self.level_events: list[LevelEvent] = []

        self.abstraction_builders: list[
            Callable[[FrameAbstraction, dict[str, Any]], None]
        ] = []
        self.last_snapshot: Optional[FrameAbstraction] = None
        self._register_game_abstractions()

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
        # Reset the environment if it's not in a playable state
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._reset_tracking()
            action = GameAction.RESET
            action.reasoning = "resetting before exploration"
            return action

        if self.mode is NavigatorMode.TODO:
            logger.info("%s TODO mode: planning logic not yet implemented", self.game_id)
            raise NotImplementedError("Navigator TODO mode is a placeholder")

        # Observe and update discovered graph / level / HUD
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
        if nfr_action is not None and nfr_action in self.ARROW_ACTIONS:
            self.last_action = nfr_action
        elif nfr_action is GameAction.RESET:
            self.last_action = None
        logger.info(
            "%s planner_step: state=%s level_start=%s action=%s reason=%s available=%s",
            self.game_id,
            current_state,
            self._level_start_state,
            nfr_action.name if nfr_action else None,
            getattr(nfr_action, "reasoning", None) if nfr_action else None,
            [action.name for action in available_actions],
        )
        return nfr_action


    def _reset_tracking(self) -> None:
        self.last_action = None
        self.unique_states_this_run.clear()
        self.energy_capacity = None
        self.current_level = 0
        self.level_events = []
        self.last_snapshot = None
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
        energy_measurement = self._measure_energy_blocks(grid)
        frame_hash = self._hash_grid(grid)
        abstraction = FrameAbstraction(frame_hash=frame_hash, grid=grid)
        context = {
            "frame": frame,
            "previous": self.last_snapshot,
            "energy_measurement": energy_measurement,
        }
        for builder in self.abstraction_builders:
            try:
                builder(abstraction, context)
            except Exception:
                logger.exception(
                    "%s abstraction builder %s failed",
                    self.game_id,
                    builder.__name__,
                )

        energy_blocks = None
        if energy_measurement:
            energy_blocks, capacity, _ = energy_measurement
            if capacity:
                self.energy_capacity = capacity

        snapshot = NavigatorSnapshot(
            frame=frame,
            abstraction=abstraction,
            frame_hash=frame_hash,
            score=frame.score,
            energy_blocks=energy_blocks,
            energy_capacity=self.energy_capacity,
            available_actions=tuple(frame.available_actions or ()),
            game_state=frame.state,
        )

        self._prev_snapshot = prev_snapshot
        self._snapshot = snapshot
        self.last_snapshot = abstraction

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

        self._save_memory()

        self._persist_metrics(
            states_visited_run=states_visited_run,
            known_states_total=known_states_total,
            level_events=self.level_events,
        )

        super().cleanup(scorecard)

    # --- Persistence ------------------------------------------------------

    def _persist_metrics(
        self,
        *,
        states_visited_run: int,
        known_states_total: int,
        level_events: list[LevelEvent],
    ) -> None:
        recorder = getattr(self, "recorder", None)
        if not recorder or not getattr(recorder, "filename", None):
            return

        target_path = Path(recorder.filename).with_suffix(".tracking.json")
        target_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "game_id": self.game_id,
            "agent": self.name,
            "states_visited_run": states_visited_run,
            "known_states_total": known_states_total,
            "energy_capacity": self.energy_capacity,
            "level_events": level_events,
        }

        with target_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def _load_memory(self) -> Memory:
        memory = Memory()
        state_graph = memory.state_graph
        if not MEMORY_PATH.exists():
            return memory

        try:
            data = json.loads(MEMORY_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning(
                "%s could not load memory file %s; starting fresh",
                self.game_id,
                MEMORY_PATH,
            )
            return memory

        states_section = data.get("states")
        if isinstance(states_section, dict):
            for hash_str, info in states_section.items():
                try:
                    state_hash = FrameHash(int(hash_str))
                except (TypeError, ValueError):
                    continue
                transitions = (
                    info.get("transitions") if isinstance(info, dict) else None
                )
                if not isinstance(transitions, dict):
                    continue
                record = state_graph.setdefault(state_hash, TransitionMap())
                for action_name, target_value in transitions.items():
                    if action_name not in GameAction.__members__:
                        continue
                    try:
                        target_hash = FrameHash(int(target_value))
                    except (TypeError, ValueError):
                        continue
                    record.transitions[GameAction[action_name]] = target_hash
        return memory

    def _save_memory(self) -> None:
        MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "states": {
                str(state_hash): {
                    "transitions": {
                        action.name: int(target)
                        for action, target in record.transitions.items()
                    },
                }
                for state_hash, record in self.memory.state_graph.items()
            }
        }
        MEMORY_PATH.write_text(json.dumps(payload, indent=2))

    # --- State tracking ---------------------------------------------------

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

    
    # --- Hashing: HUD-masked grid hashing ---------------------------------
    def _hash_grid(self, grid: list[list[Any]]) -> FrameHash:
        """
        Return a hash of the grid **ignoring HUD cells** if a HUD rect is present.
        We treat HUD as adversarially chosen noise: masking it collapses states
        that differ only by score rendering into a single identifier.
        """
        if not grid:
            return FrameHash(0)
        normalized = []
        raw_y0, raw_y1, raw_x0, raw_x1 = self.ENERGY_HUD_RECT
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
                    normalized_row.append(self._cell_value(cell))
            normalized.append(tuple(normalized_row))
        return FrameHash(hash(tuple(normalized)))

    # --- NFR helpers: graph building and shortest paths --------------------
    @staticmethod
    def _hash_frame(grid: list[list[Any]]) -> FrameHash:
        if not grid:
            return FrameHash(0)
        normalized = [
            tuple(AbstractionNavigator._cell_value(cell) for cell in row)
            for row in grid
        ]
        return FrameHash(hash(tuple(normalized)))

    @staticmethod
    def _cell_value(cell: Any) -> int:
        if isinstance(cell, int):
            return cell
        if isinstance(cell, list) and cell:
            first = cell[0]
            return first if isinstance(first, int) else 0
        return 0

    # === Game-specific abstractions (ls20) =================================

    ENERGY_HUD_RECT = (1, 2, 2, 45)  # (y0, y1, x0, x1); tuned for ls20 energy bar

    def _register_game_abstractions(self) -> None:
        self.abstraction_builders.extend(
            [
                self._build_player_abstraction,
                self._build_energy_abstraction,
            ]
        )

    def _build_player_abstraction(
        self, snapshot: FrameAbstraction, context: dict[str, Any]
    ) -> None:
        detection = self._detect_player(snapshot.grid)
        if detection:
            snapshot.add("player", detection)

    def _build_energy_abstraction(
        self, snapshot: FrameAbstraction, context: dict[str, Any]
    ) -> None:
        measurement = context.get("energy_measurement")
        if not measurement:
            return
        blocks_filled, capacity, _ = measurement
        if capacity:
            self.energy_capacity = capacity
        snapshot.add(
            "energy",
            {
                "blocks": blocks_filled,
                "capacity": capacity,
            },
        )

    def _handle_level_change(self, snapshot: NavigatorSnapshot) -> None:
        self.current_level = snapshot.score
        self._pending_level_score = snapshot.score
        self._pending_level_hash = snapshot.frame_hash

        event: LevelEvent = {
            "level": self.current_level,
            "step": self.action_counter,
            "state_hash": snapshot.frame_hash,
            "energy": 0,
            "timestamp": time.time(),
        }

        if snapshot.energy_blocks is not None:
            event["energy"] = snapshot.energy_blocks

        self.level_events.append(event)
        snapshot.abstraction.add("level_transition", event)
        logger.info(
            "%s level advanced to %d at step %d",
            self.game_id,
            self.current_level,
            self.action_counter,
        )

    def _detect_player(self, grid: list[list[Any]]) -> Optional[dict[str, Any]]:
        positions: list[tuple[int, int]] = []
        for y, row in enumerate(grid):
            for x, cell in enumerate(row):
                if self._cell_value(cell) == 12:
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

        return {
            "center": center,
            "pixel_count": count,
            "bbox": {
                "min_y": min_y,
                "max_y": max_y,
                "min_x": min_x,
                "max_x": max_x,
            },
        }

    def _measure_energy_blocks(
        self, grid: list[list[Any]]
    ) -> Optional[tuple[int, int, tuple[int, int, int, int]]]:
        row_index = 2
        if len(grid) <= row_index or not grid[row_index]:
            return None
        row = grid[row_index]
        _, _, x0, x1 = self.ENERGY_HUD_RECT
        row_len = len(row)
        if row_len <= x0:
            return None
        upper_x = min(row_len - 1, x1)

        blocks: list[int] = []
        for x in range(x0, upper_x + 1, 2):
            value = self._cell_value(row[x])
            if value in (3, 15):
                blocks.append(value)
            elif blocks:
                break

        total = len(blocks)
        if total < 6:
            values = {self._cell_value(row[x]) for x in range(x0, upper_x + 1, 2)}
            if values == {8} and self.energy_capacity:
                return 0, self.energy_capacity, self.ENERGY_HUD_RECT
            return None

        if any(v not in (3, 15) for v in blocks):
            return None

        filled = sum(1 for v in blocks if v == 15)
        return filled, total, self.ENERGY_HUD_RECT
