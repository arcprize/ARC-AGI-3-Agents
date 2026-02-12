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

MAX_REPL_ITERATIONS = 8       # root-LM turns in the REPL loop
MAX_OUTPUT_CHARS    = 4_000   # truncate long REPL stdout
MAX_MEMORY_ENTRIES  = 50

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

SYSTEM_PROMPT = textwrap.dedent("""\
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
  summarize_grid(grid)               → text summary
  diff_summary(prev_grid, grid)      → what changed
  find_objects(grid)                  → list of objects {color, size, center, bbox}
  grid_region(grid, x1, y1, x2, y2)  → sub-grid
  color_name(val)                     → human color name

Available game actions:
  RESET    – restart game/level
  ACTION1  – (often Up)    ACTION2  – (often Down)
  ACTION3  – (often Left)  ACTION4  – (often Right)
  ACTION5  – interact / select / execute
  ACTION6  – click at (x,y)     ACTION7  – undo
  Action meanings may differ per game — discover them!

## How to work
1. Write Python code in a ```python``` block to analyse the grid.
2. I will execute it and show you the printed output.
3. Repeat as many times as needed.
4. When ready, output your decision as:

FINAL({"action": "ACTION1", "reasoning": "...", "hypothesis": "...", "observation": "..."})

Rules:
- You MUST eventually emit a FINAL(...) line with valid JSON inside.
- Explore systematically: try actions, observe changes, form hypotheses.
- Do NOT assume any specific game — be general.
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

    def is_done(self) -> bool:
        return self.action_counter >= self.MAX_ACTIONS

    def choose_action(self, frame: FrameData) -> GameAction:
        self.turn_number += 1

        if frame.state == GameState.NOT_PLAYED:
            self._handle_reset()
            return GameAction.RESET

        if self._is_stuck(frame):
            logger.info("Stuck — exploring randomly")
            return self._exploratory_action()

        grid = frame.frame[0] if frame.frame else []
        prev_grid = self.memory[-1].get("grid") if self.memory else None

        try:
            action, meta = self._rlm_loop(grid, prev_grid, frame)
            logger.info("Action: %s | %s", action.name, meta.get("reasoning", "")[:100])
            self._record(action.name, meta, grid)
            return action
        except Exception as exc:
            logger.error("RLM loop failed: %s", exc)
            return self._fallback()

    # ── The RLM REPL loop (core of the paper) ───────────────────────

    def _rlm_loop(
        self, grid: list[list[int]], prev_grid: list[list[int]] | None,
        frame: FrameData,
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

        # Build initial user context (concise — the LLM can query more via REPL)
        user_ctx = (
            f"Turn {self.turn_number} | Game: {frame.game_id} | "
            f"State: {frame.state.name} | Levels: {getattr(frame, 'levels_completed', 0)}\n"
            f"Grid summary: {summarize_grid(grid)}\n"
            f"Changes: {diff_summary(prev_grid, grid)}\n"
            f"Hypothesis: {self.hypothesis}\n"
            f"Memory entries: {len(self.memory)}\n"
            "Write Python code to analyse the grid, then FINAL({{...}}) when ready."
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_ctx},
        ]

        for iteration in range(MAX_REPL_ITERATIONS):
            # ── LLM call ────────────────────────────────────────────
            self._total_llm_calls += 1
            logger.debug("REPL iter %d / %d", iteration + 1, MAX_REPL_ITERATIONS)

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
            max_tokens=2048,
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
        return self._consecutive_no_change >= 5

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
