from __future__ import annotations

"""BaseAbstractionNavigator: generic infrastructure for abstraction-driven exploration.

This module contains only game-agnostic logic. Provide game-specific pieces
through constructor arguments from a thin wrapper (see `abstraction_navigator.py`).
"""

import logging
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from ..agent import Agent
from ..structs import FrameData, GameAction, GameState
from .abstractions import FrameAbstraction, AbstractionDetector
from .grid_hash import FrameMask, hash_frame
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


class BaseAbstractionNavigator(Agent):
    """Exploration-focused agent with pluggable, game-specific abstractions.

    Provide the following game-specific hooks when constructing:
      - user_abstractions: list[(name, detector)] where detector(frame) -> Any | None
      - hash_mask: FrameMask used to hash frames while ignoring HUD areas etc.
      - measure_energy: callable (frame) -> EnergyHudMeasurement | None

    Notes on energy measurement:
      - By default, wrappers wire a concrete HUD extractor. If your game does not
        expose an energy HUD, pass a no-op function that returns None.
    """

    MAX_ACTIONS = 60
    ARROW_ACTIONS = [
        GameAction.ACTION1,  # Up
        GameAction.ACTION2,  # Down
        GameAction.ACTION3,  # Left
        GameAction.ACTION4,  # Right,
    ]

    def __init__(
        self,
        *args: Any,
        user_abstractions: Sequence[tuple[str, AbstractionDetector]],
        hash_mask: FrameMask,
        measure_energy: Callable[[Frame], Optional[EnergyHudMeasurement]],
        **kwargs: Any,
    ) -> None:
        # initialise known attributes for type-checker; real values set in Agent.__init__
        self.game_id = getattr(self, "game_id", "")
        self.action_counter = getattr(self, "action_counter", 0)
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1_000_000) ^ hash(self.game_id)
        self.rng = random.Random(seed)
        self.last_action: Optional[GameAction] = None

        self._user_abstractions = list(user_abstractions)
        self._hash_mask = hash_mask
        self._measure_energy = measure_energy

        self.memory: Memory = load_memory(MEMORY_PATH, logger_prefix=self.game_id)
        self._nfr_planner = NearFrontierPlanner(
            arrow_actions=self.ARROW_ACTIONS,
            state_graph=self.memory.state_graph,
        )
        self._snapshots: deque[NavigatorSnapshot] = deque(maxlen=3)

    # Hints for the type checker; values are initialised in Agent.__init__
    game_id: str
    action_counter: int

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

        prev_snapshot = self._snapshots[-2] if len(self._snapshots) >= 2 else None

        if self._should_reset_for_apparent_restart(prev_snapshot, snapshot):
            logger.info(
                "%s apparent restart screen detected: state=%s score=%s",
                self.game_id,
                snapshot.game_state.name,
                snapshot.score,
            )
            action = GameAction.RESET
            action.reasoning = "apparent-restart-reset"
            self.last_action = None
            return action
        else:
            self._track_state_graph(prev_snapshot, snapshot)

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

        energy_measurement = self._measure_energy(frame)
        frame_hash = hash_frame(frame, mask=self._hash_mask)
        abstraction = FrameAbstraction(frame_hash=frame_hash, frame=frame)
        if energy_measurement is not None:
            abstraction.add("energy", energy_measurement)
        energy_capacity = energy_measurement.capacity if energy_measurement else None

        for name, detector in self._user_abstractions:
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
        return snapshot

    def _should_reset_for_apparent_restart(
        self,
        prev_snapshot: Optional[NavigatorSnapshot],
        snapshot: NavigatorSnapshot,
    ) -> bool:
        if not self._looks_like_transition_screen(snapshot):
            return False

        confirmed_progress = (
            prev_snapshot is not None and snapshot.score > prev_snapshot.score
        )
        if confirmed_progress:
            return False

        return True

    def _looks_like_transition_screen(
        self, snapshot: NavigatorSnapshot
    ) -> bool:
        frame_layers = getattr(snapshot.frame, "frame", None)
        if not isinstance(frame_layers, list):
            return False

        layer_count = len(frame_layers)
        if layer_count <= 1:
            return False

        return True

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
            logger.warning(
                "Non-deterministic transition: state=%s, action=%s, existing_target=%s, new_target=%s",
                previous_hash,
                action.name,
                existing,
                next_hash,
            )

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


