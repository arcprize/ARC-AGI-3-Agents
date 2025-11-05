"""Planner utilities for Near-Frontier with Reset (NFR) exploration."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, NewType, Optional, Sequence, Tuple

from ..structs import GameAction


FrameHash = NewType("FrameHash", int)


@dataclass
class StateKnowledge:
    transitions: Dict[GameAction, FrameHash] = field(default_factory=dict)


@dataclass
class PlannerContext:
    """Lightweight view of the navigator knowledge for a single game."""

    arrow_actions: Sequence[GameAction]
    state_knowledge: Dict[FrameHash, StateKnowledge]


class NearFrontierPlanner:
    """Compute plans that minimise distance to the nearest unexplored frontier."""

    def __init__(self, context: PlannerContext) -> None:
        self._context = context
        self._knowledge = context.state_knowledge

    def next_action(
        self,
        *,
        current_state: FrameHash,
        available_actions: Sequence[GameAction],
        level_start_state: FrameHash,
    ) -> Optional[GameAction]:

        available_set = set(available_actions)
        if not available_set:
            return None

        adj = self._build_adj()
        s0 = level_start_state

        dist_c, prev_c = self._bfs(adj, current_state)
        dist_s0, prev_s0 = self._bfs(adj, s0)

        INF = 10**9
        best: Optional[Tuple[int, int, FrameHash]] = None
        for state in self._frontier_states():
            d_current = dist_c.get(state, INF)
            d_reset = dist_s0.get(state, INF)
            d_plus = min(d_current, 1 + d_reset)
            if d_plus >= INF:
                continue
            key = (d_plus, d_reset, state)
            if best is None or key < best:
                best = key

        if best is None:
            return None

        _, d_reset, target_state = best
        d_current = dist_c.get(target_state, INF)

        navigation: List[GameAction] = []
        if d_current <= 1 + d_reset:
            navigation = self._extract_path_actions(prev_c, current_state, target_state)
        else:
            action = GameAction.RESET
            action.reasoning = "nfr-reset"
            return action

        if navigation:
            action = navigation[0]
            action.reasoning = f"nfr-nav:{action.name.lower()}"
            return action

        # Already at the chosen frontier: probe the first unseen action that is available.
        for action in self._context.arrow_actions:
            if action not in available_set:
                continue
            if self._is_action_known(current_state, action):
                continue
            action.reasoning = f"nfr-probe:{action.name.lower()}"
            return action

        return None

    def _discovered_states(self) -> set[FrameHash]:
        states: set[FrameHash] = set(self._knowledge.keys())
        for record in self._knowledge.values():
            states.update(record.transitions.values())
        return states

    def _build_adj(self) -> Dict[FrameHash, List[Tuple[FrameHash, GameAction]]]:
        adjacency: Dict[FrameHash, List[Tuple[FrameHash, GameAction]]] = {}
        for state, record in self._knowledge.items():
            for action, target in record.transitions.items():
                if action not in self._context.arrow_actions:
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

    def _extract_path_actions(
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
            for action in self._context.arrow_actions:
                if not self._is_action_known(state, action):
                    yield state
                    break

    def _is_action_known(self, state: FrameHash, action: GameAction) -> bool:
        record = self._knowledge.get(state)
        if record is None:
            return False
        return action in record.transitions
