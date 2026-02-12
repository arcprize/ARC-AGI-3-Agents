"""
RLM Agent — Recursive Language Model agent for ARC-AGI-3.

Implements the RLM paradigm from Zhang, Kraska & Khattab (MIT, 2025)
  arxiv.org/abs/2512.24601  ·  github.com/alexzhang13/rlm

Core idea: instead of stuffing the entire game state into one giant
prompt, the LLM interacts with a persistent Python REPL that holds the
game grid as a *variable*.  The LLM writes code to inspect, transform,
and reason about the grid, sees the printed output, and iterates until
it emits FINAL({...}) with its chosen action.

No external ``rlms`` dependency — the REPL loop is implemented here
using the OpenAI-compatible chat API (works with OpenRouter, OpenAI,
Anthropic via their OpenAI-compat endpoints, or any local vLLM server).

Designed to be fully general across ALL ARC-AGI-3 games.
"""

from __future__ import annotations

import io
import json
import logging
import os
import random
import re
import textwrap
import traceback
from contextlib import redirect_stdout, redirect_stderr
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

COLOR_MAP: dict[int, str] = {
    0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
    5: "gray", 6: "magenta", 7: "orange", 8: "light_blue",
    9: "purple", 10: "brown", 11: "cyan", 12: "dark_red",
    13: "dark_green", 14: "dark_blue", 15: "white",
}

MAX_REPL_ITERATIONS_EXPLORE = 2   # REPL turns during exploration phase
MAX_REPL_ITERATIONS_EXPLOIT = 3   # REPL turns once hypothesis is formed
MAX_OUTPUT_CHARS    = 1_500       # truncate long REPL stdout
MAX_MEMORY_ENTRIES  = 20
EXPLORATION_TURNS   = 5           # systematic exploration for first N turns

# ──────────────────────────────────────────────────────────────────────
# General-purpose grid helpers (pre-loaded into the REPL namespace)
# ──────────────────────────────────────────────────────────────────────


def color_name(val: int) -> str:
    return COLOR_MAP.get(val, f"color_{val}")


def summarize_grid(grid: list[list[int]], max_objects: int = 64) -> str:
    if not grid or not grid[0]:
        return "Empty grid"
    rows, cols = len(grid), len(grid[0])
    counts: dict[int, int] = {}
    nz = 0
    for row in grid:
        for c in row:
            counts[c] = counts.get(c, 0) + 1
            if c != 0:
                nz += 1
    top = sorted(counts.items(), key=lambda x: -x[1])[:8]
    info = ", ".join(f"{color_name(c)}={n}" for c, n in top)
    return f"Grid {rows}x{cols} | non-zero={nz}/{rows*cols} | colors=[{info}]"


def diff_summary(prev: list[list[int]] | None, curr: list[list[int]]) -> str:
    if prev is None:
        return "First frame — no previous grid"
    if not prev or not curr:
        return "Missing grid data"
    R, C = min(len(prev), len(curr)), min(len(prev[0]), len(curr[0]))
    changes: list[str] = []
    n = 0
    for r in range(R):
        for c in range(C):
            if prev[r][c] != curr[r][c]:
                n += 1
                if len(changes) < 5:
                    changes.append(
                        f"({r},{c}):{color_name(prev[r][c])}→{color_name(curr[r][c])}"
                    )
    if n == 0:
        return "No changes"
    detail = "; ".join(changes)
    if n > 5:
        detail += f" … +{n - 5} more"
    return f"{n} pixels changed: {detail}"


def find_objects(grid: list[list[int]]) -> list[dict[str, Any]]:
    """Flood-fill connected components (ignoring colour 0)."""
    if not grid or not grid[0]:
        return []
    R, C = len(grid), len(grid[0])
    vis: set[tuple[int, int]] = set()
    objs: list[dict[str, Any]] = []
    for y in range(R):
        for x in range(C):
            if grid[y][x] != 0 and (x, y) not in vis:
                col = grid[y][x]
                stk = [(x, y)]
                vis.add((x, y))
                cells: list[tuple[int, int]] = []
                while stk:
                    cx, cy = stk.pop()
                    cells.append((cx, cy))
                    for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < C and 0 <= ny < R and grid[ny][nx] == col and (nx, ny) not in vis:
                            vis.add((nx, ny))
                            stk.append((nx, ny))
                xs = [p[0] for p in cells]
                ys = [p[1] for p in cells]
                objs.append({
                    "color": color_name(col), "color_id": col,
                    "size": len(cells),
                    "center": (sum(xs) // len(xs), sum(ys) // len(ys)),
                    "bbox": (min(xs), min(ys), max(xs), max(ys)),
                })
    return objs


def grid_region(grid: list[list[int]], x1: int, y1: int, x2: int, y2: int) -> list[list[int]]:
    return [row[x1:x2 + 1] for row in grid[y1:y2 + 1]]


def detect_moving_object(
    prev_grid: list[list[int]] | None, curr_grid: list[list[int]],
) -> dict[str, Any]:
    """Compare two frames and return info about what moved (i.e. the player).
    General-purpose: works for any game where a character moves on the grid."""
    if prev_grid is None or not curr_grid:
        return {"found": False, "reason": "no previous frame"}
    R, C = len(curr_grid), len(curr_grid[0])
    appeared: list[tuple[int, int, int]] = []   # (x, y, color)
    disappeared: list[tuple[int, int, int]] = []
    for y in range(min(R, len(prev_grid))):
        for x in range(min(C, len(prev_grid[0]))):
            if prev_grid[y][x] != curr_grid[y][x]:
                disappeared.append((x, y, prev_grid[y][x]))
                appeared.append((x, y, curr_grid[y][x]))
    if not appeared:
        return {"found": False, "reason": "no changes"}
    # The "player" is likely the NEW color that appeared in the changed cells
    from collections import Counter
    new_colors = Counter(c for _, _, c in appeared)
    bg_colors = Counter(c for _, _, c in disappeared)
    # Player colors are those that appeared but aren't dominant background
    player_colors = set(new_colors.keys()) - {0}  # exclude black
    player_cells = [(x, y) for x, y, c in appeared if c in player_colors and c not in bg_colors]
    if not player_cells:
        # Fallback: just use all appeared cells
        player_cells = [(x, y) for x, y, c in appeared if c != 0]
    if not player_cells:
        return {"found": False, "reason": "only background changes"}
    xs = [p[0] for p in player_cells]
    ys = [p[1] for p in player_cells]
    cx, cy = sum(xs) // len(xs), sum(ys) // len(ys)
    return {
        "found": True,
        "center": (cx, cy),
        "bbox": (min(xs), min(ys), max(xs), max(ys)),
        "n_changed_pixels": len(appeared),
        "player_colors": list(player_colors),
    }


def find_unique_objects(grid: list[list[int]], bg_color: int | None = None) -> list[dict[str, Any]]:
    """Find distinct colored regions, ignoring background colors.
    Auto-detects background as the 2 most common colors."""
    if not grid or not grid[0]:
        return []
    R, C = len(grid), len(grid[0])
    from collections import Counter
    color_counts = Counter(c for row in grid for c in row)
    total = R * C
    # Auto-detect background: top 2 most common colors
    if bg_color is not None:
        bg_colors = {bg_color, 0}
    else:
        top2 = [c for c, _ in color_counts.most_common(3)]
        bg_colors = set(top2) | {0}
    # Rare colors (< 2% of grid) are likely interactive objects
    objects = []
    for color, count in color_counts.items():
        if color in bg_colors:
            continue
        if count < total * 0.02:  # rare = interesting
            cells = [(x, y) for y in range(R) for x in range(C) if grid[y][x] == color]
            xs = [p[0] for p in cells]
            ys = [p[1] for p in cells]
            objects.append({
                "color": color_name(color), "color_id": color,
                "count": count, "rarity": round(count / total, 4),
                "center": (sum(xs) // len(xs), sum(ys) // len(ys)),
                "bbox": (min(xs), min(ys), max(xs), max(ys)),
            })
    return sorted(objects, key=lambda o: o["count"])


def suggest_direction(player_pos: tuple[int, int], target_pos: tuple[int, int]) -> str:
    """Suggest ACTION1-4 to move player toward target.
    ACTION1=Up, ACTION2=Down, ACTION3=Left, ACTION4=Right."""
    px, py = player_pos
    tx, ty = target_pos
    dx, dy = tx - px, ty - py
    if abs(dx) > abs(dy):
        return "ACTION4" if dx > 0 else "ACTION3"
    else:
        return "ACTION2" if dy > 0 else "ACTION1"


# ──────────────────────────────────────────────────────────────────────
# Sandboxed REPL  (exec-based, like the official RLM implementation)
# ──────────────────────────────────────────────────────────────────────


class GameREPL:
    """
    A persistent Python REPL sandbox that holds game state as variables
    and exposes helper functions.  Based on the REPL design from
    Zhang et al. §3 — the LLM writes code cells, we ``exec`` them, and
    feed stdout back into the conversation.
    """

    def __init__(self, grid: list[list[int]], prev_grid: list[list[int]] | None,
                 memory: list[dict[str, Any]], hypothesis: str, turn: int) -> None:
        self._globals: dict[str, Any] = {
            "__builtins__": __builtins__,
            # game state variables
            "grid": grid,
            "prev_grid": prev_grid,
            "memory": memory,
            "hypothesis": hypothesis,
            "turn_number": turn,
            # helper functions
            "summarize_grid": summarize_grid,
            "diff_summary": diff_summary,
            "find_objects": find_objects,
            "grid_region": grid_region,
            "color_name": color_name,
            "color_map": COLOR_MAP,
            "detect_moving_object": detect_moving_object,
            "find_unique_objects": find_unique_objects,
            "suggest_direction": suggest_direction,
        }
        self._locals: dict[str, Any] = {}

    def execute(self, code: str) -> str:
        """Run *code* in the sandbox; return captured stdout + stderr."""
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(code, self._globals, self._locals)  # noqa: S102
        except Exception:
            stderr_buf.write(traceback.format_exc())

        out = stdout_buf.getvalue() + stderr_buf.getvalue()
        # Truncate if needed
        if len(out) > MAX_OUTPUT_CHARS:
            out = out[:MAX_OUTPUT_CHARS] + f"\n… [truncated to {MAX_OUTPUT_CHARS} chars]"
        return out

    def get_var(self, name: str) -> Any:
        return self._locals.get(name, self._globals.get(name))


# ──────────────────────────────────────────────────────────────────────
# System prompt  — instructs the root LM how to use the REPL
# ──────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_EXPLORE = textwrap.dedent("""\
You are an expert AI agent playing ARC-AGI-3 games.
ARC-AGI-3 games are turn-based 2D grid puzzles with NO instructions.
You must DISCOVER the rules through exploration.

You have a persistent Python REPL.  Variables already loaded:
  grid        – current 64×64 game grid (list[list[int]], values 0-15)
  prev_grid   – previous grid (None on first turn)
  memory      – list of past observations (list[dict])
  hypothesis  – your current theory (str)
  turn_number – current turn (int)

Helper functions:
  summarize_grid(grid)                          → text summary
  diff_summary(prev_grid, grid)                 → what changed
  find_objects(grid)                            → list of objects {color, size, center, bbox}
  detect_moving_object(prev_grid, grid)         → find what moved (the player!)
  find_unique_objects(grid, bg_color=4)          → rare objects (goals, keys)
  suggest_direction(player_pos, target_pos)     → best ACTION1-4 toward target
  grid_region(grid, x1, y1, x2, y2)            → sub-grid
  color_name(val)                               → human color name

Available game actions:
  ACTION1=Up  ACTION2=Down  ACTION3=Left  ACTION4=Right
  ACTION5=interact  ACTION6=click(x,y)  ACTION7=undo  RESET=restart

## Strategy
1. Use detect_moving_object(prev_grid, grid) to find the player position.
2. Use find_unique_objects(grid) to find goals/targets (rare colored objects).
3. Use suggest_direction(player, target) to pick the right direction.
4. Emit FINAL({...}) IMMEDIATELY — do NOT over-analyse.

FINAL({"action": "ACTION1", "reasoning": "...", "hypothesis": "...", "observation": "..."})

Rules:
- You MUST emit FINAL(...) within 1-2 iterations. Be FAST.
- If nothing changed after an action, try a DIFFERENT direction.
""")

SYSTEM_PROMPT_EXPLOIT = textwrap.dedent("""\
You are an expert AI agent playing ARC-AGI-3 games.
You have been exploring and now have a hypothesis about the game.

You have a persistent Python REPL.  Variables already loaded:
  grid, prev_grid, memory, hypothesis, turn_number
  + All helper functions from exploration phase.

KEY helpers for navigation:
  detect_moving_object(prev_grid, grid) → {found, center, bbox}
  find_unique_objects(grid)             → rare objects (likely goals)
  suggest_direction(player, target)     → ACTION1-4 toward target

Available actions: ACTION1=Up, ACTION2=Down, ACTION3=Left, ACTION4=Right,
  ACTION5=interact, ACTION6=click(x,y), ACTION7=undo, RESET=restart.

## Strategy
1. Find player with detect_moving_object(prev_grid, grid).
2. Find target with find_unique_objects(grid).
3. Use suggest_direction() to navigate.
4. Emit FINAL({...}) with your action.

Be strategic and FAST. Navigate toward the goal.
""")

# ──────────────────────────────────────────────────────────────────────
# LLM client  — thin OpenAI-compatible wrapper
# ──────────────────────────────────────────────────────────────────────


def _build_openai_client() -> Any:
    """Return an ``openai.OpenAI`` client configured from env vars."""
    from openai import OpenAI

    backend = os.getenv("RLM_BACKEND", "openrouter")

    if backend == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY required")
        return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    if backend == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required")
        return OpenAI(api_key=api_key)

    if backend == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY required")
        return OpenAI(api_key=api_key, base_url="https://api.anthropic.com/v1")

    # Fallback: treat as base_url
    return OpenAI(api_key=os.getenv("LLM_API_KEY", "none"), base_url=backend)


# ──────────────────────────────────────────────────────────────────────
# RLM Agent
# ──────────────────────────────────────────────────────────────────────


class RLMAgent(Agent):
    """
    Recursive Language Model agent for ARC-AGI-3.

    Implements the RLM paradigm (Zhang et al., 2025): the LLM
    interacts with a persistent Python REPL that holds the game grid
    as a variable.  The LLM writes code to analyse it, sees the
    printed output, iterates, and emits FINAL({...}) when ready.

    General-purpose — works across all ARC-AGI-3 games by discovering
    mechanics through exploration.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.MODEL: str = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
        self.VERBOSE: bool = os.getenv("RLM_VERBOSE", "false").lower() == "true"

        # Agent state
        self.hypothesis: str = "Unknown game. Need to explore and discover the rules."
        self.memory: list[dict[str, Any]] = []
        self.turn_number: int = 0
        self._total_llm_calls: int = 0
        self._consecutive_no_change: int = 0
        self._last_grid_hash: int | None = None

        # Programmatic navigation state
        self._player_pos: tuple[int, int] | None = None
        self._goal_pos: tuple[int, int] | None = None
        self._action_effects: dict[str, tuple[int, int]] = {}  # action -> (dx, dy)
        self._tried_actions: list[str] = []
        self._nav_mode: bool = False  # True once we have player + goal

        # LLM client (lazy — created on first use so tests can patch)
        self._client: Any | None = None

        logger.info("RLMAgent init: model=%s", self.MODEL)

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = _build_openai_client()
        return self._client

    # ── Core interface ──────────────────────────────────────────────

    @property
    def MAX_ACTIONS(self) -> int:
        return 80

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return self.action_counter >= self.MAX_ACTIONS or latest_frame.state == GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        frame = latest_frame
        self.turn_number += 1
        logger.info("RLM call #%d | turn %d | model=%s", self._total_llm_calls + 1, self.turn_number, self.MODEL)

        if frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._handle_reset()
            return GameAction.RESET

        # Track which actions this game supports
        self._available_actions = set(frame.available_actions or [1, 2, 3, 4])

        grid = frame.frame[0] if frame.frame else []
        prev_grid = self.memory[-1].get("grid") if self.memory else None

        # Detect if last action had no effect and learn action effects
        if prev_grid and grid == prev_grid:
            self._consecutive_no_change += 1
            logger.info("No change after last action (%d consecutive)", self._consecutive_no_change)
        else:
            if self._consecutive_no_change > 0:
                logger.info("Grid changed! Resetting no-change counter.")
            self._consecutive_no_change = 0
            self._last_grid_hash = hash(str(grid))
            # Learn: track player position from frame diffs
            self._update_navigation(prev_grid, grid)

        # Phase 1: Systematic exploration to learn action effects
        if self.turn_number <= EXPLORATION_TURNS and self._consecutive_no_change < 3:
            action = self._systematic_explore(grid, prev_grid, frame)
            return action

        # If stuck for 3+ turns, try random different action
        if self._consecutive_no_change >= 3:
            logger.info("Stuck for %d turns — trying random action", self._consecutive_no_change)
            action = self._exploratory_action()
            self._record(action.name, {"reasoning": "stuck — random exploration"}, grid)
            return action

        # Phase 3: Fall back to LLM REPL for complex reasoning
        try:
            action, meta = self._rlm_loop(grid, prev_grid, frame)
            logger.info("Action: %s | %s", action.name, meta.get("reasoning", "")[:100])
            self._record(action.name, meta, grid)
            return action
        except Exception as exc:
            logger.error("RLM loop failed: %s", exc)
            return self._fallback()

    def _update_navigation(self, prev_grid: list[list[int]] | None, grid: list[list[int]]) -> None:
        """Learn action effects from frame diffs. Compute displacement vectors directly."""
        if not prev_grid or not grid or not self._tried_actions:
            # First frame: find goal from rare objects
            if not self._goal_pos and grid:
                self._find_goal(grid, set())
            return

        last_action = self._tried_actions[-1]

        # Compute displacement: find where things appeared vs disappeared
        R, C = len(grid), len(grid[0])
        appeared_xy: list[tuple[int, int]] = []
        disappeared_xy: list[tuple[int, int]] = []
        for y in range(min(R, len(prev_grid))):
            for x in range(min(C, len(prev_grid[0]))):
                if prev_grid[y][x] != grid[y][x]:
                    disappeared_xy.append((x, y))
                    appeared_xy.append((x, y))

        if appeared_xy and disappeared_xy:
            # Displacement = shift direction of ALL changed pixels
            # For a moving character, appeared pixels are biased toward the new position
            # We use a simpler approach: find the bounding box shift
            old_cx = sum(x for x, y in disappeared_xy) / len(disappeared_xy)
            old_cy = sum(y for x, y in disappeared_xy) / len(disappeared_xy)
            new_cx = sum(x for x, y in appeared_xy) / len(appeared_xy)
            new_cy = sum(y for x, y in appeared_xy) / len(appeared_xy)

            # Track player position as center of changed region in new frame
            # Filter to only "new" colored cells (non-background in new frame)
            from collections import Counter
            color_counts = Counter(c for row in grid for c in row)
            bg_colors = {c for c, _ in color_counts.most_common(3)} | {0}
            player_cells = [(x, y) for x, y in appeared_xy if grid[y][x] not in bg_colors]
            if player_cells:
                self._player_pos = (
                    sum(x for x, y in player_cells) // len(player_cells),
                    sum(y for x, y in player_cells) // len(player_cells),
                )

            # Learn action displacement using sign only (more robust)
            # Look at where non-bg pixels moved TO vs FROM
            new_rare = [(x, y) for x, y in appeared_xy if grid[y][x] not in bg_colors]
            old_rare = [(x, y) for x, y in disappeared_xy if prev_grid[y][x] not in bg_colors]
            if new_rare and old_rare:
                dx = sum(x for x, y in new_rare) / len(new_rare) - sum(x for x, y in old_rare) / len(old_rare)
                dy = sum(y for x, y in new_rare) / len(new_rare) - sum(y for x, y in old_rare) / len(old_rare)
                # Normalize to sign: +1, -1, or 0
                sdx = 1 if dx > 0.5 else (-1 if dx < -0.5 else 0)
                sdy = 1 if dy > 0.5 else (-1 if dy < -0.5 else 0)
                if sdx != 0 or sdy != 0:
                    self._action_effects[last_action] = (sdx, sdy)
                    logger.info("Learned: %s -> (%+d, %+d) [raw: dx=%.1f, dy=%.1f]",
                                last_action, sdx, sdy, dx, dy)

            if self._player_pos:
                logger.info("Player at %s", self._player_pos)

        # Find goal on first detection
        if not self._goal_pos and grid:
            player_colors = set()
            if appeared_xy:
                from collections import Counter as Ctr
                player_colors = {grid[y][x] for x, y in appeared_xy if grid[y][x] not in bg_colors}
            self._find_goal(grid, player_colors)

    def _find_goal(self, grid: list[list[int]], player_colors: set[int]) -> None:
        """Find the goal as the rarest non-player, non-background object."""
        unique = find_unique_objects(grid)
        if unique:
            for obj in unique:
                if obj["color_id"] in player_colors:
                    continue
                self._goal_pos = obj["center"]
                logger.info("Goal candidate: %s at %s (count=%d)",
                            obj["color"], self._goal_pos, obj["count"])
                break

    def _action_for_direction(self, want_dx: int, want_dy: int) -> str | None:
        """Find which action produces movement closest to (want_dx, want_dy) using learned effects."""
        if not self._action_effects:
            return None
        best_action = None
        best_dot = -999
        for action_name, (adx, ady) in self._action_effects.items():
            # Normalize: just check sign agreement
            dot = 0
            if want_dx != 0 and adx != 0:
                dot += 1 if (want_dx > 0) == (adx > 0) else -1
            if want_dy != 0 and ady != 0:
                dot += 1 if (want_dy > 0) == (ady > 0) else -1
            if dot > best_dot:
                best_dot = dot
                best_action = action_name
        return best_action if best_dot > 0 else None

    def _navigate(self, grid: list[list[int]]) -> GameAction | None:
        """Programmatic navigation: move player toward goal using learned action mappings."""
        if not self._player_pos or not self._goal_pos:
            return None

        # If stuck in nav mode for too long, reset and try something else
        if self._consecutive_no_change >= 3:
            logger.info("Nav stuck for %d turns — resetting goal", self._consecutive_no_change)
            self._goal_pos = None  # will re-detect next turn
            return None  # fall through to LLM or random

        px, py = self._player_pos
        gx, gy = self._goal_pos
        dx, dy = gx - px, gy - py

        # Use LEARNED action effects (not hardcoded assumptions!)
        if self._action_effects:
            # Primary: move toward goal
            want_dx = 1 if dx > 0 else (-1 if dx < 0 else 0)
            want_dy = 1 if dy > 0 else (-1 if dy < 0 else 0)

            if self._consecutive_no_change == 0:
                # Try primary direction (largest delta)
                if abs(dx) > abs(dy):
                    action_name = self._action_for_direction(want_dx, 0)
                else:
                    action_name = self._action_for_direction(0, want_dy)
            elif self._consecutive_no_change == 1:
                # Try secondary direction
                if abs(dx) > abs(dy):
                    action_name = self._action_for_direction(0, want_dy)
                else:
                    action_name = self._action_for_direction(want_dx, 0)
            else:
                # Try perpendicular to escape walls
                if abs(dx) > abs(dy):
                    action_name = self._action_for_direction(0, -want_dy if want_dy else 1)
                else:
                    action_name = self._action_for_direction(-want_dx if want_dx else 1, 0)

            if action_name:
                action = GameAction.from_name(action_name)
                self._tried_actions.append(action_name)
                logger.info("Nav: player=(%d,%d) goal=(%d,%d) -> %s (learned)", px, py, gx, gy, action_name)
                return action

        # Fallback: haven't learned mappings yet, try all 4 actions sequentially
        fallback_actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
        idx = self.turn_number % len(fallback_actions)
        action_name = fallback_actions[idx]
        action = GameAction.from_name(action_name)
        self._tried_actions.append(action_name)
        logger.info("Nav: player=(%d,%d) goal=(%d,%d) -> %s (fallback)", px, py, gx, gy, action_name)
        return action

    def _filter_action(self, action: GameAction) -> GameAction:
        """Ensure we only send actions the game supports."""
        if hasattr(self, '_available_actions') and action.value not in self._available_actions:
            # Pick a random available action instead
            import random
            avail = [a for a in GameAction if a.value in self._available_actions and a is not GameAction.RESET]
            return random.choice(avail) if avail else GameAction.ACTION1
        return action

    def _systematic_explore(self, grid: list[list[int]], prev_grid: list[list[int]] | None, frame: FrameData) -> GameAction:
        """First N turns: systematically try each action to learn what they do."""
        # Only try available actions during exploration
        avail = sorted(self._available_actions) if hasattr(self, '_available_actions') else [1, 2, 3, 4]
        explore_actions = [GameAction.from_id(a) for a in avail if a != 0]
        if not explore_actions:
            explore_actions = [GameAction.ACTION1]
        idx = (self.turn_number - 1) % len(explore_actions)
        action = explore_actions[idx]

        observation = diff_summary(prev_grid, grid)
        self._record(action.name, {
            "reasoning": f"Systematic exploration turn {self.turn_number}: trying {action.name}",
            "observation": observation,
        }, grid)
        self._tried_actions.append(action.name)
        logger.info("Explore turn %d: %s | changes: %s", self.turn_number, action.name, observation[:80])
        return action

    # ── The RLM REPL loop (core of the paper) ───────────────────────

    def _rlm_loop(
        self, grid: list[list[int]], prev_grid: list[list[int]] | None,
        frame: FrameData, max_iters: int | None = None,
    ) -> tuple[GameAction, dict[str, Any]]:
        """
        Run the iterative LLM ↔ REPL loop.

        1. Initialise a GameREPL with the current grid as a variable.
        2. Send the system prompt + first user message to the LLM.
        3. The LLM replies with Python code (and/or FINAL(...)).
        4. If FINAL(...) → parse and return.
        5. Otherwise exec the code in the REPL, feed stdout back.
        6. Repeat up to MAX_REPL_ITERATIONS.
        """
        repl = GameREPL(grid, prev_grid, self.memory, self.hypothesis, self.turn_number)

        # Determine phase: exploration or exploitation
        is_exploring = self.turn_number <= EXPLORATION_TURNS
        max_iters_default = MAX_REPL_ITERATIONS_EXPLORE if is_exploring else MAX_REPL_ITERATIONS_EXPLOIT
        max_iters = max_iters if max_iters is not None else max_iters_default
        sys_prompt = SYSTEM_PROMPT_EXPLORE if is_exploring else SYSTEM_PROMPT_EXPLOIT

        # Build recent memory summary (last 5 turns)
        mem_summary = ""
        for m in self.memory[-5:]:
            mem_summary += f"  Turn {m.get('turn', '?')}: {m.get('action', '?')} → {m.get('observation', 'no obs')[:60]}\n"

        user_ctx = (
            f"Turn {self.turn_number} | Game: {frame.game_id} | "
            f"State: {frame.state.name} | Levels: {getattr(frame, 'levels_completed', 0)}\n"
            f"Grid summary: {summarize_grid(grid)}\n"
            f"Changes: {diff_summary(prev_grid, grid)}\n"
            f"Hypothesis: {self.hypothesis}\n"
        )
        if mem_summary:
            user_ctx += f"Recent memory:\n{mem_summary}"
        user_ctx += "Analyse quickly, then FINAL({{...}})."

        messages: list[dict[str, str]] = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_ctx},
        ]

        for iteration in range(max_iters):
            # ── LLM call ────────────────────────────────────────────
            self._total_llm_calls += 1
            logger.debug("REPL iter %d / %d", iteration + 1, max_iters)

            response_text = self._chat(messages)

            if self.VERBOSE:
                logger.info("LLM [iter %d]: %s", iteration, response_text[:300])

            # ── Check for FINAL(...) ────────────────────────────────
            final = self._extract_final(response_text)
            if final is not None:
                return self._parse_final(final)

            # ── Extract code and execute ────────────────────────────
            code = self._extract_code(response_text)
            if code:
                repl_output = repl.execute(code)
                if not repl_output.strip():
                    repl_output = "(no output)"
                logger.debug("REPL output: %s", repl_output[:200])

                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": f"REPL output:\n```\n{repl_output}\n```\nContinue analysis or emit FINAL({{...}})."})
            else:
                # No code and no FINAL — nudge the LLM
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": "Please write Python code in a ```python block to analyse the grid, or emit FINAL({...}) with your action."})

        # Exhausted iterations — try to get a final answer
        messages.append({"role": "user", "content": "Max iterations reached. You MUST emit FINAL({...}) NOW with your best action."})
        response_text = self._chat(messages)
        self._total_llm_calls += 1
        final = self._extract_final(response_text)
        if final is not None:
            return self._parse_final(final)

        # Ultimate fallback
        logger.warning("REPL loop exhausted without FINAL — using fallback")
        return self._fallback(), {"reasoning": "REPL loop exhausted"}

    # ── LLM chat helper ─────────────────────────────────────────────

    def _chat(self, messages: list[dict[str, str]]) -> str:
        """Single chat-completion call via OpenAI-compatible API."""
        resp = self._get_client().chat.completions.create(
            model=self.MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
        )
        return resp.choices[0].message.content or ""

    # ── Parsing helpers ─────────────────────────────────────────────

    @staticmethod
    def _extract_final(text: str) -> dict[str, Any] | None:
        """Look for ``FINAL({...})`` in the LLM output."""
        m = re.search(r'FINAL\s*\((\{.*?\})\)', text, re.DOTALL)
        if m:
            try:
                raw = m.group(1).replace("'", '"')
                raw = re.sub(r',\s*}', '}', raw)
                return json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: look for FINAL(ACTION_NAME)
        m2 = re.search(r'FINAL\s*\(\s*["\']?(ACTION\d|RESET)["\']?\s*\)', text, re.IGNORECASE)
        if m2:
            return {"action": m2.group(1).upper()}

        return None

    @staticmethod
    def _extract_code(text: str) -> str | None:
        """Extract the first ```python ... ``` block from the response."""
        m = re.search(r'```(?:python)?\s*\n(.+?)```', text, re.DOTALL)
        if m:
            return m.group(1).strip()
        # Fallback: if the whole response looks like code (has print/for/if)
        if re.search(r'\b(print|for |if |def |import )\b', text) and "FINAL" not in text:
            return text.strip()
        return None

    def _parse_final(self, data: dict[str, Any]) -> tuple[GameAction, dict[str, Any]]:
        """Convert FINAL dict → (GameAction, metadata)."""
        action_name = str(data.get("action", "ACTION5")).upper().strip()
        if action_name not in VALID_ACTIONS:
            logger.warning("Invalid action '%s' in FINAL, falling back", action_name)
            action_name = "ACTION5"

        action = GameAction.from_name(action_name)

        if action_name == "ACTION6" and "action_data" in data:
            try:
                action.set_data(data["action_data"])
            except Exception:
                action = GameAction.ACTION5

        hypothesis = data.get("hypothesis", self.hypothesis)
        self.hypothesis = hypothesis

        meta = {
            "reasoning": data.get("reasoning", ""),
            "hypothesis": hypothesis,
            "observation": data.get("observation", ""),
            "model": self.MODEL,
            "llm_calls": self._total_llm_calls,
            "turn": self.turn_number,
        }
        return action, meta

    # ── State management ────────────────────────────────────────────

    def _handle_reset(self) -> None:
        logger.info("Game reset — clearing memory")
        self.memory.clear()
        self.hypothesis = "Game reset. Starting fresh exploration."
        self.turn_number = 0
        self._consecutive_no_change = 0
        self._last_grid_hash = None

    def _is_stuck(self, frame: FrameData) -> bool:
        h = hash(str(frame.frame))
        if h == self._last_grid_hash:
            self._consecutive_no_change += 1
        else:
            self._consecutive_no_change = 0
            self._last_grid_hash = h
        return self._consecutive_no_change >= 3

    def _exploratory_action(self) -> GameAction:
        self._consecutive_no_change = 0
        return random.choice([
            GameAction.ACTION1, GameAction.ACTION2,
            GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5,
        ])

    def _fallback(self) -> GameAction:
        logger.warning("Using fallback action (ACTION5)")
        return GameAction.ACTION5

    def _record(self, action: str, meta: dict[str, Any],
                grid: list[list[int]] | None = None) -> None:
        self.memory.append({
            "turn": self.turn_number,
            "action": action,
            "observation": meta.get("observation", ""),
            "reasoning": meta.get("reasoning", ""),
            "hypothesis": self.hypothesis,
            "grid": grid,
        })
        if len(self.memory) > MAX_MEMORY_ENTRIES:
            self.memory = self.memory[-MAX_MEMORY_ENTRIES:]
