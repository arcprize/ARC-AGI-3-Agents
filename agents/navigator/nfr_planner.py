"""Planner utilities for Near-Frontier with Reset (NFR) exploration."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..structs import GameAction
from .types import FrameHash, STATE_GRAPH, TransitionMap


logger = logging.getLogger(__name__)

class NearFrontierPlanner:
    """Compute plans that minimise distance to the nearest unexplored frontier."""

    def __init__(
        self,
        *,
        arrow_actions: Sequence[GameAction],
        state_graph: STATE_GRAPH,
        unstable_states: Optional[set[FrameHash]] = None,
    ) -> None:
        self._arrow_actions = list(arrow_actions)
        self._state_graph = state_graph
        self._unstable: set[FrameHash] = set(unstable_states or set())

    def next_action(
        self,
        *,
        current_state: FrameHash,
        available_actions: Sequence[GameAction],
        level_start_state: FrameHash,
        target_state: Optional[FrameHash] = None,
    ) -> Optional[GameAction]:

        available_set = set(available_actions)
        if not available_set:
            return None

        adj = self._build_adj()
        s0 = level_start_state

        dist_c, prev_c = self._bfs(adj, current_state)
        dist_s0, prev_s0 = self._bfs(adj, s0)

        if target_state is not None and target_state in dist_c:
            terminal_path = self._recover_path(prev_c, current_state, target_state)
            if terminal_path:
                action = terminal_path[0]
                action.reasoning = f"nfr-terminal:{action.name.lower()}"
                return action

        INF = 10**9
        best: Optional[Tuple[int, int, FrameHash]] = None
        for state in self._frontier_states():
            d_current = dist_c.get(state, INF)
            d_reset = dist_s0.get(state, INF)
            frontier_cost = min(d_current, 1 + d_reset)
            if frontier_cost >= INF:
                continue
            key = (frontier_cost, d_reset, state)
            if best is None or key < best:
                best = key

        if best is None:
            return None

        # best stores (cost, reset_distance, frontier_state)
        _, d_reset, target_state = best
        d_current = dist_c.get(target_state, INF)

        navigation: List[GameAction] = []
        if d_current <= 1 + d_reset:
            navigation = self._recover_path(prev_c, current_state, target_state)
        else:
            logger.info(
                "nfr-reset decision: current=%s s0=%s target=%s d_current=%s d_reset=%s available=%s",
                current_state,
                s0,
                target_state,
                d_current,
                d_reset,
                [action.name for action in available_actions],
            )
            action = GameAction.RESET
            action.reasoning = "nfr-reset"
            return action

        if navigation:
            action = navigation[0]
            action.reasoning = f"nfr-nav:{action.name.lower()}"
            return action

        # Already at the chosen frontier: probe the first unseen arrow action that is available.
        for action in self._arrow_actions:
            if action not in available_set:
                continue
            if self._is_action_known(current_state, action):
                continue
            action.reasoning = f"nfr-probe:{action.name.lower()}"
            return action

        return None

    def _discovered_states(self) -> set[FrameHash]:
        states: set[FrameHash] = set(self._state_graph.keys())
        for transition_map in self._state_graph.values():
            states.update(transition_map.transitions.values())
        return {s for s in states if s not in self._unstable}

    def _build_adj(self) -> Dict[FrameHash, List[Tuple[FrameHash, GameAction]]]:
        adjacency: Dict[FrameHash, List[Tuple[FrameHash, GameAction]]] = {}
        for state, transition_map in self._state_graph.items():
            if state in self._unstable:
                continue
            for action, target in transition_map.transitions.items():
                if action not in self._arrow_actions:
                    continue
                if target in self._unstable:
                    continue
                adjacency.setdefault(state, []).append((target, action))
        return adjacency

    def _bfs(
        self,
        adjacency: Dict[FrameHash, List[Tuple[FrameHash, GameAction]]],
        start: FrameHash,
    ) -> Tuple[Dict[FrameHash, int], Dict[FrameHash, Tuple[FrameHash, GameAction]]]:
        distances: Dict[FrameHash, int] = {start: 0}
        predecessors: Dict[FrameHash, Tuple[FrameHash, GameAction]] = {}
        queue: deque[FrameHash] = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor, action in adjacency.get(node, []):
                if neighbor in distances:
                    continue
                distances[neighbor] = distances[node] + 1
                predecessors[neighbor] = (node, action)
                queue.append(neighbor)
        return distances, predecessors

    def _recover_path(
        self,
        predecessors: Dict[FrameHash, Tuple[FrameHash, GameAction]],
        start: FrameHash,
        goal: FrameHash,
    ) -> List[GameAction]:
        actions: List[GameAction] = []
        node = goal
        while node != start:
            previous = predecessors.get(node)
            if previous is None:
                return []
            parent, action = previous
            actions.append(action)
            node = parent
        actions.reverse()
        return actions

    def _frontier_states(self) -> Iterable[FrameHash]:
        for state in self._discovered_states():
            for action in self._arrow_actions:
                if not self._is_action_known(state, action):
                    yield state
                    break

    def _is_action_known(self, state: FrameHash, action: GameAction) -> bool:
        transition_map = self._state_graph.get(state)
        if transition_map is None:
            return False
        return action in transition_map.transitions
