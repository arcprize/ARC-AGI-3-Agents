"""
RLM Agent — Recursive Language Model agent for ARC-AGI-3.

Uses the `rlms` library to recursively reason about game frames via a Python
REPL environment, with OpenRouter as the default LLM backend.

Designed to be general-purpose across ALL ARC-AGI-3 games (LS20, FT09, VC33, etc.)
by discovering game mechanics through exploration rather than hardcoding
game-specific knowledge.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import textwrap
from typing import Any

from arcengine import FrameData, GameAction, GameState

from ..agent import Agent

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

VALID_ACTIONS = [
    "RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4",
    "ACTION5", "ACTION6", "ACTION7",
]

COLOR_MAP = {
    0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
    5: "gray", 6: "magenta", 7: "orange", 8: "light_blue",
    9: "purple", 10: "brown", 11: "cyan", 12: "dark_red",
    13: "dark_green", 14: "dark_blue", 15: "white",
}

# ──────────────────────────────────────────────────────────────────────
# General-Purpose Grid Analysis Utilities
# (No game-specific assumptions – works for any ARC-AGI-3 game)
# ──────────────────────────────────────────────────────────────────────


def color_name(val: int) -> str:
    """Human-readable name for a cell value."""
    return COLOR_MAP.get(val, f"color_{val}")


def summarize_grid(grid: list[list[int]], max_objects: int = 64) -> str:
    """Produce a compact summary of the grid state."""
    if not grid or not grid[0]:
        return "Empty grid"

    rows, cols = len(grid), len(grid[0])
    color_counts: dict[int, int] = {}
    non_zero = 0

    for row in grid:
        for cell in row:
            color_counts[cell] = color_counts.get(cell, 0) + 1
            if cell != 0:
                non_zero += 1

    top_colors = sorted(color_counts.items(), key=lambda x: -x[1])[:8]
    color_info = ", ".join(f"{color_name(c)}={n}" for c, n in top_colors)

    return (
        f"Grid size: {rows}x{cols} | "
        f"Non-zero cells: {non_zero}/{rows * cols} | "
        f"Unique values: {len(color_counts)} | "
        f"Colors: [{color_info}]"
    )


def diff_summary(prev_grid: list[list[int]] | None, curr_grid: list[list[int]]) -> str:
    """Compact change-detection between two grids."""
    if prev_grid is None:
        return "First frame — no previous grid to compare"
    if not prev_grid or not curr_grid:
        return "Cannot compare — missing grid data"

    rows = min(len(prev_grid), len(curr_grid))
    cols = min(len(prev_grid[0]), len(curr_grid[0])) if rows else 0
    changes: list[str] = []
    pixel_changes = 0

    for r in range(rows):
        for c in range(cols):
            if prev_grid[r][c] != curr_grid[r][c]:
                pixel_changes += 1
                if len(changes) < 5:
                    changes.append(
                        f"({r},{c}): {color_name(prev_grid[r][c])}→{color_name(curr_grid[r][c])}"
                    )

    if pixel_changes == 0:
        return "No changes detected"
    detail = "; ".join(changes)
    if pixel_changes > 5:
        detail += f" … and {pixel_changes - 5} more"
    return f"{pixel_changes} pixels changed: {detail}"


def find_objects(grid: list[list[int]]) -> list[dict[str, Any]]:
    """Find connected-component objects in the grid (flood-fill, ignoring color 0)."""
    if not grid or not grid[0]:
        return []

    rows, cols = len(grid), len(grid[0])
    visited: set[tuple[int, int]] = set()
    objects: list[dict[str, Any]] = []

    for y in range(rows):
        for x in range(cols):
            if grid[y][x] != 0 and (x, y) not in visited:
                color = grid[y][x]
                queue = [(x, y)]
                visited.add((x, y))
                cells: list[tuple[int, int]] = []

                while queue:
                    cx, cy = queue.pop()
                    cells.append((cx, cy))
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nx, ny = cx + dx, cy + dy
                        if (0 <= nx < cols and 0 <= ny < rows
                                and grid[ny][nx] == color and (nx, ny) not in visited):
                            visited.add((nx, ny))
                            queue.append((nx, ny))

                cx = sum(x for x, _ in cells) // len(cells)
                cy = sum(y for _, y in cells) // len(cells)
                xs = [x for x, _ in cells]
                ys = [y for _, y in cells]
                objects.append({
                    "color": color_name(color),
                    "color_id": color,
                    "size": len(cells),
                    "center": (cx, cy),
                    "bbox": (min(xs), min(ys), max(xs), max(ys)),
                })

    return objects


def grid_region(grid: list[list[int]], x1: int, y1: int, x2: int, y2: int) -> list[list[int]]:
    """Extract a rectangular sub-region from the grid."""
    return [row[x1:x2 + 1] for row in grid[y1:y2 + 1]]


# ──────────────────────────────────────────────────────────────────────
# System prompt for the RLM — fully general, no game-specific logic
# ──────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert AI game agent playing ARC-AGI-3 interactive reasoning games.
ARC-AGI-3 games are turn-based 2D grid puzzles with NO instructions provided.
You must discover the rules, goals, and mechanics through exploration.

Each game is unique — do NOT assume any specific game mechanics.

## Environment
- `grid`: current 64×64 game grid (list[list[int]], values 0-15)
- `prev_grid`: previous grid (or None on first turn)
- `memory`: list of past observations
- `hypothesis`: your current theory about the game rules
- `turn_number`: current turn count

## Helper functions available
- `summarize_grid(grid)` → compact text summary of grid state
- `diff_summary(prev_grid, grid)` → what changed between frames
- `find_objects(grid)` → list of connected-component objects with color, size, center, bbox
- `grid_region(grid, x1, y1, x2, y2)` → extract a sub-region
- `color_name(val)` → human-readable color name for a cell value

## Actions
- RESET: restart the game / level
- ACTION1: usually mapped to Up / W
- ACTION2: usually mapped to Down / S
- ACTION3: usually mapped to Left / A
- ACTION4: usually mapped to Right / D
- ACTION5: interact / select / execute (Space)
- ACTION6: click at coordinates — requires {"x": int, "y": int} in action_data
- ACTION7: undo

**Actions are semantically mapped but may behave differently per game.**

## Your task
Analyze the grid, compare with previous state, form hypotheses, and choose an action.
Output MUST be a `result` dict:

```python
result = {
    "action": "ACTION1",
    "reasoning": "why I chose this action",
    "hypothesis": "my theory about the game rules",
    "observation": "what I noticed this turn"
}
```

## Strategy for unknown games
1. On turn 1, summarize the grid and identify objects
2. Try each directional action and observe what changes
3. Look for patterns: did something move? appear? disappear?
4. Form a hypothesis about the goal (reach something? arrange something? avoid something?)
5. Refine your hypothesis with each observation
6. Use ACTION5 when you suspect interaction is needed
7. Use ACTION6 (click) if objects seem clickable
8. RESET if stuck or if you want to retry with new knowledge
""")


# ──────────────────────────────────────────────────────────────────────
# RLM Agent Implementation
# ──────────────────────────────────────────────────────────────────────


class RLMAgent(Agent):
    """
    Recursive Language Model agent for ARC-AGI-3.

    General-purpose agent that works across all ARC-AGI-3 games by
    discovering mechanics through exploration rather than relying on
    game-specific hardcoded logic.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # RLM configuration from environment
        self.BACKEND = os.getenv("RLM_BACKEND", "openrouter")
        self.MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
        self.ENVIRONMENT = os.getenv("RLM_ENVIRONMENT", "local")
        self.VERBOSE = os.getenv("RLM_VERBOSE", "false").lower() == "true"

        # Agent state
        self.hypothesis: str = "Unknown game. Need to explore and discover the rules."
        self.memory: list[dict[str, Any]] = []
        self.turn_number: int = 0
        self._total_rlm_calls: int = 0
        self._consecutive_no_change: int = 0
        self._last_grid_hash: int | None = None

        # Initialize RLM client
        self._rlm_client = self._create_rlm_client()

        logger.info(
            "RLMAgent initialized: backend=%s, model=%s, env=%s",
            self.BACKEND, self.MODEL, self.ENVIRONMENT,
        )

    # ── RLM client setup ────────────────────────────────────────────

    def _create_rlm_client(self) -> Any:
        """Create and configure the RLM client."""
        try:
            import rlms
            return rlms.RLM(**self._build_backend_kwargs())
        except ImportError as exc:
            logger.error("Failed to import rlms: %s", exc)
            raise ImportError(
                "RLM agent requires the 'rlms' package. Install with: pip install rlms"
            ) from exc

    def _build_backend_kwargs(self) -> dict[str, Any]:
        """Build backend-specific configuration for RLM."""
        env_keys = {
            "openrouter": ("OPENROUTER_API_KEY", self.MODEL),
            "openai": ("OPENAI_API_KEY", os.getenv("OPENAI_MODEL", "gpt-4")),
            "anthropic": ("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")),
        }

        if self.BACKEND not in env_keys:
            raise ValueError(f"Unsupported backend: {self.BACKEND}")

        key_var, model = env_keys[self.BACKEND]
        api_key = os.getenv(key_var)
        if not api_key:
            raise ValueError(f"{key_var} environment variable required for {self.BACKEND} backend")

        return {
            "backend": self.BACKEND,
            "model": model,
            "api_key": api_key,
            "environment": self.ENVIRONMENT,
            "verbose": self.VERBOSE,
        }

    # ── Core agent interface ────────────────────────────────────────

    @property
    def MAX_ACTIONS(self) -> int:
        return 80

    def choose_action(self, frame: FrameData) -> GameAction:
        """Choose the next action using recursive reasoning."""
        self.turn_number += 1

        # Handle NOT_PLAYED state (game not started yet)
        if frame.state == GameState.NOT_PLAYED:
            self._handle_game_reset()
            return GameAction.RESET

        # Stuck detection → explore randomly
        if self._is_stuck(frame):
            logger.info("Agent appears stuck, taking exploratory action")
            return self._exploratory_action()

        # Build prompt and call RLM
        prompt = self._build_prompt(frame)
        try:
            self._total_rlm_calls += 1
            logger.info(
                "RLM call #%d | turn %d | model=%s",
                self._total_rlm_calls, self.turn_number, self.MODEL,
            )
            rlm_result = self._rlm_client.completion(prompt)
            action, meta = self._parse_rlm_result(rlm_result, frame)
            logger.info("Chosen action: %s | %s", action.name, meta.get("reasoning", "")[:100])
            return action
        except Exception as exc:
            logger.error("RLM call failed: %s", exc)
            return self._fallback_action()

    def is_done(self) -> bool:
        return self.action_counter >= self.MAX_ACTIONS

    # ── Prompt building ─────────────────────────────────────────────

    def _build_prompt(self, frame: FrameData) -> str:
        """Build the complete prompt for the RLM."""
        prev_grid = self.memory[-1].get("grid") if self.memory else None
        # frame.frame is list[list[list[int]]] (list of 2D grids); use the first one
        grid = frame.frame[0] if frame.frame else []

        # Pre-compute analysis for context
        grid_summary = summarize_grid(grid)
        change_summary = diff_summary(prev_grid, grid)
        objects = find_objects(grid)
        obj_summary = "; ".join(
            f"{o['color']}(size={o['size']}, center={o['center']})"
            for o in objects[:15]
        ) or "No objects found"

        # Recent memory
        recent = self.memory[-5:] if self.memory else []
        memory_text = "\n".join(
            f"  Turn {m['turn']}: {m['action']} → {m['observation'][:80]}"
            for m in recent
        ) or "  (no prior observations)"

        prompt = f"""{SYSTEM_PROMPT}

{'=' * 60}
## Current State — Turn {self.turn_number}
Game ID: {frame.game_id}
Game State: {frame.state}
Levels Completed: {getattr(frame, 'levels_completed', 0)}

Grid Summary: {grid_summary}
Changes: {change_summary}
Objects: {obj_summary}

## Hypothesis
{self.hypothesis}

## Recent Memory
{memory_text}

{'=' * 60}
Analyze the grid and choose your next action.
Set result = {{"action": "...", "reasoning": "...", "hypothesis": "...", "observation": "..."}}
"""
        return prompt

    # ── Response parsing ────────────────────────────────────────────

    def _parse_rlm_result(
        self, rlm_result: Any, frame: FrameData
    ) -> tuple[GameAction, dict[str, Any]]:
        """Parse the RLM response and extract the chosen action."""
        response_text = str(rlm_result.response) if rlm_result else ""
        logger.debug("RLM response (first 500 chars): %s", response_text[:500])

        parsed = self._extract_result_dict(response_text)

        if parsed and "action" in parsed:
            action_name = parsed["action"].upper().strip()

            if action_name not in VALID_ACTIONS:
                logger.warning("Invalid action '%s', falling back", action_name)
                action = self._fallback_action()
            else:
                action = GameAction.from_name(action_name)

            # ACTION6 coordinate handling
            if action_name == "ACTION6" and "action_data" in parsed:
                try:
                    action.set_data(parsed["action_data"])
                except Exception as exc:
                    logger.warning("Invalid ACTION6 data: %s", exc)
                    action = self._fallback_action()

            hypothesis = parsed.get("hypothesis", self.hypothesis)
            reasoning = parsed.get("reasoning", "")
            observation = parsed.get("observation", "")

            self.hypothesis = hypothesis
            grid_2d = frame.frame[0] if frame.frame else []
            self._record_observation(action.name, observation, reasoning, grid_2d)

            return action, self._build_meta(action, reasoning, frame)

        # Fallback: scan for action keyword in raw text
        logger.warning("Structured parse failed, scanning text for action keyword")
        action = self._action_from_text(response_text)
        grid_2d = frame.frame[0] if frame.frame else []
        self._record_observation(action.name, "Fallback parse", "", grid_2d)
        return action, self._build_meta(action, "Fallback text extraction", frame)

    def _extract_result_dict(self, text: str) -> dict[str, Any] | None:
        """Try to extract a result dict from the RLM response."""
        # Strategy 1: regex for result = {...} or bare JSON with "action"
        patterns = [
            r'result\s*=\s*(\{[^}]+\})',
            r'```(?:json)?\s*(\{[^}]+\})\s*```',
            r'(\{"action":\s*"[^"]+?"[^}]*\})',
        ]
        for pat in patterns:
            for m in re.findall(pat, text, re.DOTALL | re.IGNORECASE):
                try:
                    cleaned = m.replace("'", '"')
                    cleaned = re.sub(r',\s*}', '}', cleaned)
                    cleaned = re.sub(r',\s*]', ']', cleaned)
                    parsed = json.loads(cleaned)
                    if isinstance(parsed, dict) and "action" in parsed:
                        return parsed
                except (json.JSONDecodeError, ValueError):
                    continue

        # Strategy 2: key=value style
        action_match = re.search(
            r'["\']?action["\']?\s*[:=]\s*["\']?(ACTION\d|RESET)["\']?',
            text, re.IGNORECASE,
        )
        if action_match:
            result: dict[str, Any] = {"action": action_match.group(1).upper()}
            for key in ("reasoning", "hypothesis", "observation"):
                km = re.search(
                    rf'["\']?{key}["\']?\s*[:=]\s*["\']([^"\']+)["\']',
                    text, re.IGNORECASE,
                )
                if km:
                    result[key] = km.group(1)
            return result

        # Strategy 3: any bare ACTION mention
        mention = re.search(r'\b(ACTION\d|RESET)\b', text, re.IGNORECASE)
        if mention:
            return {"action": mention.group(1).upper(), "reasoning": "Extracted from text"}

        return None

    def _action_from_text(self, text: str) -> GameAction:
        """Last-resort: find any action keyword in raw text."""
        m = re.search(r'\b(ACTION\d|RESET)\b', text, re.IGNORECASE)
        if m and m.group(1).upper() in VALID_ACTIONS:
            return GameAction.from_name(m.group(1).upper())
        return self._fallback_action()

    # ── State management helpers ────────────────────────────────────

    def _handle_game_reset(self) -> None:
        """Clear memory on game reset."""
        logger.info("Game reset — clearing memory")
        self.memory.clear()
        self.hypothesis = "Game reset. Starting fresh exploration."
        self.turn_number = 0
        self._consecutive_no_change = 0
        self._last_grid_hash = None

    def _is_stuck(self, frame: FrameData) -> bool:
        """Detect if the agent is stuck (grid unchanged for several turns)."""
        h = hash(str(frame.frame))
        if h == self._last_grid_hash:
            self._consecutive_no_change += 1
        else:
            self._consecutive_no_change = 0
            self._last_grid_hash = h
        return self._consecutive_no_change >= 5

    def _exploratory_action(self) -> GameAction:
        """Random exploration to escape stuck states."""
        self._consecutive_no_change = 0
        return random.choice([
            GameAction.ACTION1, GameAction.ACTION2,
            GameAction.ACTION3, GameAction.ACTION4,
            GameAction.ACTION5,
        ])

    def _fallback_action(self) -> GameAction:
        """Safe fallback action when parsing fails."""
        logger.warning("Using fallback action (ACTION5)")
        return GameAction.ACTION5

    def _record_observation(
        self, action: str, observation: str, reasoning: str,
        grid: list[list[int]] | None = None,
    ) -> None:
        """Append an observation to episodic memory."""
        self.memory.append({
            "turn": self.turn_number,
            "action": action,
            "observation": observation,
            "reasoning": reasoning,
            "hypothesis": self.hypothesis,
            "grid": grid,
        })
        if len(self.memory) > 50:
            self.memory = self.memory[-50:]

    def _build_meta(
        self, action: GameAction, reasoning: str, frame: FrameData
    ) -> dict[str, Any]:
        """Build reasoning metadata dict."""
        return {
            "model": self.MODEL,
            "backend": self.BACKEND,
            "agent_type": "rlm_agent",
            "action_chosen": action.name,
            "reasoning": reasoning,
            "hypothesis": self.hypothesis,
            "turn": self.turn_number,
            "rlm_calls_total": self._total_rlm_calls,
            "game_context": {
                "levels_completed": getattr(frame, "levels_completed", 0),
                "state": frame.state.name,
                "action_counter": self.action_counter,
            },
        }
