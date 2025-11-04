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
from pathlib import Path
from typing import Any, Callable, Optional, TypedDict

from ..agent import Agent
from ..structs import FrameData, GameAction, GameState

logger = logging.getLogger()

# === Generic infrastructure ==================================================

class LevelEvent(TypedDict):
    level: int
    step: int
    state_hash: int
    energy: int
    timestamp: float


@dataclass
class FrameAbstraction:
    """In-memory snapshot of a frame enhanced with higher-level abstractions."""

    frame_hash: int
    grid: list[list[Any]]
    abstractions: dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, value: Any) -> None:
        self.abstractions[name] = value

    def get(self, name: str) -> Any:
        return self.abstractions.get(name)


class AbstractionNavigator(Agent):
    """Exploration-focused agent that will grow into an abstraction navigator."""

    MAX_ACTIONS = 30
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
        self.last_state_hash: Optional[int] = None

        self.knowledge_path = Path("agents/knowledge/random_blocks.json")
        self.state_visit_counts, self.state_transition_counts = self._load_knowledge()
        self.unique_states_this_run: set[int] = set()

        self.energy_capacity: Optional[int] = None
        self.current_level = 0
        self.level_events: list[LevelEvent] = []

        self._knowledge_dirty = False
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
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._reset_tracking()
            action = GameAction.RESET
            action.reasoning = "resetting before exploration"
            return action

        self._observe(latest_frame)

        available_actions = list(latest_frame.available_actions or [])
        if not available_actions:
            available_actions = [GameAction.RESET]

        arrow_actions = [a for a in available_actions if a in self.ARROW_ACTIONS]
        if not arrow_actions:
            action = self.rng.choice(available_actions)
            action.reasoning = f"following available action ({action.name})"
            self.last_action = action if action in self.ARROW_ACTIONS else None
            return action

        chosen = self._select_exploratory_action(arrow_actions)
        chosen.reasoning = f"exploring with {chosen.name}"
        self.last_action = chosen
        return chosen

    def _select_exploratory_action(self, candidates: list[GameAction]) -> GameAction:
        current_state = self.last_state_hash
        if current_state is None:
            return self.rng.choice(candidates)

        unknown_actions: list[GameAction] = []
        scored_actions: list[tuple[int, GameAction]] = []

        for action in candidates:
            key = (self.game_id, current_state, action)
            transition_counts = self.state_transition_counts.get(key)
            if not transition_counts:
                unknown_actions.append(action)
                continue
            total = sum(transition_counts.values())
            scored_actions.append((total, action))

        if unknown_actions:
            return self.rng.choice(unknown_actions)

        if scored_actions:
            min_total = min(total for total, _ in scored_actions)
            best = [action for total, action in scored_actions if total == min_total]
            return self.rng.choice(best)

        return self.rng.choice(candidates)

    def _reset_tracking(self) -> None:
        self.last_action = None
        self.last_state_hash = None
        self.unique_states_this_run.clear()
        self.energy_capacity = None
        self.current_level = 0
        self.level_events = []
        self.last_snapshot = None

    def _observe(self, frame: FrameData) -> None:
        if frame.is_empty() or not frame.frame:
            return

        grid = frame.frame[0]
        frame_hash = self._hash_frame(grid)
        snapshot = FrameAbstraction(frame_hash=frame_hash, grid=grid)
        context = {"frame": frame, "previous": self.last_snapshot}
        for builder in self.abstraction_builders:
            try:
                builder(snapshot, context)
            except Exception:
                logger.exception(
                    "%s abstraction builder %s failed",
                    self.game_id,
                    builder.__name__,
                )

        previous_state_hash = self.last_state_hash
        self._record_state_visit(frame_hash)
        if (
            previous_state_hash is not None
            and self.last_action
            and self.last_action is not GameAction.RESET
        ):
            self._record_state_transition(previous_state_hash, self.last_action, frame_hash)

        self.last_state_hash = frame_hash
        self.last_snapshot = snapshot

    def cleanup(self, scorecard: Optional[Any] = None) -> None:
        states_visited_run = len(self.unique_states_this_run)
        known_states_total = len(self.state_visit_counts.get(self.game_id, {}))
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

        if self._knowledge_dirty:
            self._save_knowledge()

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

    def _load_knowledge(
        self,
    ) -> tuple[
        dict[str, dict[int, int]],
        dict[tuple[str, int, GameAction], dict[int, int]],
    ]:
        state_visits: dict[str, dict[int, int]] = {}
        state_transitions: dict[
            tuple[str, int, GameAction], dict[int, int]
        ] = {}

        if not self.knowledge_path.exists():
            return state_visits, state_transitions

        try:
            data = json.loads(self.knowledge_path.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning(
                "%s could not load knowledge file %s; starting fresh",
                self.game_id,
                self.knowledge_path,
            )
            return state_visits, state_transitions

        states_section = data.get("states", {})
        if isinstance(states_section, dict):
            for game_id, info in states_section.items():
                visit_counts = info.get("visit_counts")
                if not isinstance(game_id, str) or not isinstance(visit_counts, dict):
                    continue
                visits: dict[int, int] = {}
                for hash_str, count in visit_counts.items():
                    try:
                        visits[int(hash_str)] = int(count)
                    except (TypeError, ValueError):
                        continue
                if visits:
                    state_visits[game_id] = visits

        transitions_section = data.get("state_transitions", [])
        if isinstance(transitions_section, list):
            for item in transitions_section:
                if not isinstance(item, dict):
                    continue
                game_id = item.get("game_id")
                source = item.get("source")
                action_name = item.get("action")
                targets = item.get("targets", {})
                if (
                    not isinstance(game_id, str)
                    or not isinstance(source, int)
                    or not isinstance(action_name, str)
                    or action_name not in GameAction.__members__
                    or not isinstance(targets, dict)
                ):
                    continue
                action = GameAction[action_name]
                counts: dict[int, int] = {}
                for target_hash_str, count in targets.items():
                    try:
                        counts[int(target_hash_str)] = int(count)
                    except (TypeError, ValueError):
                        continue
                if counts:
                    state_transitions[(game_id, source, action)] = counts

        return state_visits, state_transitions

    def _save_knowledge(self) -> None:
        self.knowledge_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "states": {
                game_id: {
                    "visit_counts": {str(state_hash): count for state_hash, count in visits.items()}
                }
                for game_id, visits in self.state_visit_counts.items()
                if visits
            },
            "state_transitions": [
                {
                    "game_id": game_id,
                    "source": source_hash,
                    "action": action.name,
                    "targets": {str(target_hash): count for target_hash, count in targets.items()},
                }
                for (game_id, source_hash, action), targets in self.state_transition_counts.items()
                if targets
            ],
        }
        self.knowledge_path.write_text(json.dumps(payload, indent=2))
        self._knowledge_dirty = False

    # --- State tracking ---------------------------------------------------

    def _record_state_visit(self, frame_hash: int) -> None:
        visits = self.state_visit_counts.setdefault(self.game_id, {})
        visits[frame_hash] = visits.get(frame_hash, 0) + 1
        self.unique_states_this_run.add(frame_hash)
        self._knowledge_dirty = True

    def _record_state_transition(
        self,
        previous_hash: int,
        action: GameAction,
        next_hash: int,
    ) -> None:
        key = (self.game_id, previous_hash, action)
        targets = self.state_transition_counts.setdefault(key, {})
        targets[next_hash] = targets.get(next_hash, 0) + 1
        self._knowledge_dirty = True

    @staticmethod
    def _hash_frame(grid: list[list[Any]]) -> int:
        if not grid:
            return 0
        normalized = [
            tuple(AbstractionNavigator._cell_value(cell) for cell in row)
            for row in grid
        ]
        return hash(tuple(normalized))

    @staticmethod
    def _cell_value(cell: Any) -> int:
        if isinstance(cell, int):
            return cell
        if isinstance(cell, list) and cell:
            first = cell[0]
            return first if isinstance(first, int) else 0
        return 0

    # === Game-specific abstractions (ls20) =================================

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
        measurement = self._measure_energy_blocks(snapshot.grid)
        if measurement is None:
            return
        blocks_filled, capacity = measurement
        snapshot.add(
            "energy",
            {
                "blocks": blocks_filled,
                # expose legacy field for downstream compatibility
                "segments": blocks_filled,
                "capacity": capacity,
            },
        )
        self._detect_level_transition(snapshot, blocks_filled, capacity)

    def _detect_level_transition(
        self, snapshot: FrameAbstraction, blocks: int, capacity: int
    ) -> None:
        previous_snapshot = self.last_snapshot
        if previous_snapshot is None:
            self.energy_capacity = capacity
            snapshot.add("level", self.current_level)
            return

        prev_info = previous_snapshot.get("energy") or {}
        prev_blocks = prev_info.get("blocks", prev_info.get("segments"))

        if not self._energy_refilled(prev_blocks, blocks, capacity):
            snapshot.add("level", self.current_level)
            return

        if snapshot.frame_hash == previous_snapshot.frame_hash:
            snapshot.add("level", self.current_level)
            return

        if capacity > (self.energy_capacity or 0):
            self.energy_capacity = capacity
        self.current_level += 1
        event: LevelEvent = {
            "level": self.current_level,
            "step": self.action_counter,
            "state_hash": snapshot.frame_hash,
            "energy": blocks,
            "timestamp": time.time(),
        }
        self.level_events.append(event)
        snapshot.add("level_transition", event)
        logger.info(
            "%s detected new level %d at step %d",
            self.game_id,
            self.current_level,
            self.action_counter,
        )

        snapshot.add("level", self.current_level)

    @staticmethod
    def _energy_refilled(
        previous_blocks: Optional[int], current_blocks: int, capacity: int
    ) -> bool:
        if current_blocks <= 0:
            return False
        if previous_blocks == 0:
            return True
        if previous_blocks is None and capacity > 0:
            return current_blocks >= max(1, capacity - 1)
        return False

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
    ) -> Optional[tuple[int, int]]:
        row_index = 2
        if len(grid) <= row_index or not grid[row_index]:
            return None
        row = grid[row_index]

        blocks: list[int] = []
        for x in range(2, len(row), 2):
            value = self._cell_value(row[x])
            if value in (3, 15):
                blocks.append(value)
            elif blocks:
                break

        total = len(blocks)
        if total < 6:
            values = {self._cell_value(row[x]) for x in range(2, len(row), 2)}
            if values == {8} and self.energy_capacity:
                return 0, self.energy_capacity
            return None

        if any(v not in (3, 15) for v in blocks):
            return None

        filled = sum(1 for v in blocks if v == 15)
        return filled, total
