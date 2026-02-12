import ast
import contextlib
import io
import json
import logging
import os
from collections import defaultdict
from hashlib import blake2b
from typing import Any, Optional

import openai
from arcengine import FrameData, GameAction, GameState
from openai import OpenAI as OpenAIClient

from .llm_agents import ReasoningLLM

logger = logging.getLogger(__name__)


class RLM(ReasoningLLM):
    """Lean recursive language-model agent.

    - Root level picks a game action.
    - Any level can call `call_subproblem` for symbolic decomposition.
    - Subproblems return compact insights (`return_insight`).
    - Agent keeps compact external memory outside model context.
    - A guarded `python_repl` tool exposes persistent `ctx` for code-space reasoning.
    """

    MAX_ACTIONS: int = 120
    DO_OBSERVATION: bool = True
    MODEL = "gpt-5-mini"
    MODEL_REQUIRES_TOOLS = True
    REASONING_EFFORT: Optional[str] = None

    RLM_MAX_INTERNAL_STEPS = 6
    RLM_MAX_SUB_STEPS = 4
    RLM_MAX_DEPTH = 3
    RLM_MAX_FACTS = 64
    RLM_MAX_TRANSITIONS = 64
    RLM_MAX_SUBPROBLEMS = 32
    RLM_MAX_GRID_SUMMARIES = 2
    RLM_HISTOGRAM_TOP_K = 8
    RLM_FOCUS_WINDOW_DEFAULT = 12

    memory_facts: list[dict[str, Any]]
    transition_log: list[dict[str, Any]]
    subproblem_log: list[dict[str, Any]]
    sent_actions: list[str]
    tested_actions_by_state: dict[str, dict[str, dict[str, float]]]
    state_visits: dict[str, int]
    current_state_key: Optional[str]
    context_store: dict[str, Any]
    client: OpenAIClient

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.memory_facts = []
        self.transition_log = []
        self.subproblem_log = []
        self.sent_actions = []
        self.tested_actions_by_state = {}
        self.state_visits = defaultdict(int)
        self.current_state_key = None
        self.context_store = {
            "globals": {},
            "runs": 0,
            "latest_state_key": "",
        }
        self.client = OpenAIClient(api_key=os.environ.get("OPENAI_API_KEY", ""))

    @property
    def name(self) -> str:
        sanitized_model_name = self.MODEL.replace("/", "-").replace(":", "-")
        return f"{super().name}.{sanitized_model_name}.recursive"

    def build_user_prompt(self, latest_frame: FrameData) -> str:  # unused by RLM
        return ""

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            action = GameAction.RESET
            action.reasoning = {
                "agent": "RLM",
                "mode": "bootstrap_reset",
                "state": latest_frame.state.name,
                "turn": self.action_counter,
            }
            self._record_sent_action(action)
            return action

        self._ingest_transition(frames, latest_frame)
        grid = self._select_planning_grid(latest_frame)
        self.current_state_key = self._state_key_for_grid(grid) if grid else None
        if self.current_state_key:
            self.state_visits[self.current_state_key] += 1
        self.context_store["latest_state_key"] = self.current_state_key or ""
        self.context_store["latest_grid"] = self._select_planning_grid(latest_frame)

        result = self._solve_subproblem(
            latest_frame=latest_frame,
            objective="Choose the next game action.",
            focus="recent_transition",
            depth=0,
            x=None,
            y=None,
            size=None,
            allow_action=True,
        )

        action = result.get("action")
        forced = action is None
        if forced:
            action = self._fallback_action(latest_frame)

        action.reasoning = self._build_replay_reasoning(
            latest_frame=latest_frame,
            selected_action=action,
            turn_trace=result.get("trace", []),
            forced_action_used=forced,
        )
        self._record_sent_action(action)
        return action

    def _solve_subproblem(
        self,
        latest_frame: FrameData,
        objective: str,
        focus: str,
        depth: int,
        x: Any,
        y: Any,
        size: Any,
        allow_action: bool,
    ) -> dict[str, Any]:
        if depth > self.RLM_MAX_DEPTH:
            return {
                "status": "depth_limit",
                "objective": objective,
                "depth": depth,
                "insight": "Maximum recursion depth reached.",
                "confidence": 0.0,
                "trace": [],
            }

        tools = self._build_query_tools(include_return_insight=not allow_action)
        if allow_action:
            tools += self._action_tools(latest_frame)
        messages = [
            {"role": "system", "content": self._build_system_prompt(depth, allow_action)},
            {
                "role": "user",
                "content": self._build_problem_user_prompt(
                    latest_frame=latest_frame,
                    objective=objective,
                    focus=focus,
                    depth=depth,
                    x=x,
                    y=y,
                    size=size,
                    allow_action=allow_action,
                ),
            },
        ]
        trace: list[dict[str, Any]] = []
        step_budget = self.RLM_MAX_INTERNAL_STEPS if allow_action else self._subproblem_step_budget(depth)

        for _ in range(step_budget):
            message = self._call_chat(messages, tools, tool_required=True)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                break

            messages.append({"role": "assistant", "tool_calls": tool_calls})
            tool_call = tool_calls[0]
            fn = tool_call.get("function", {})
            fn_name = str(fn.get("name", ""))
            args = self._json_object(fn.get("arguments", "{}"))
            tool_call_id = str(tool_call.get("id", "call_0"))

            if allow_action and self._is_game_action_name(fn_name):
                action = self._build_action_from_tool(fn_name, args)
                trace.append({"depth": depth, "type": "action", "name": fn_name, "args": args})
                return {
                    "status": "action",
                    "depth": depth,
                    "objective": objective,
                    "action": action,
                    "trace": trace,
                }

            if fn_name == "return_insight":
                insight = str(args.get("insight", "")).strip() or "No insight."
                evidence = str(args.get("evidence", "")).strip()
                confidence = self._safe_float(args.get("confidence"), 0.5)
                self._remember_fact("subproblem_insight", insight, confidence)
                self._remember_subproblem(objective, depth, "insight", confidence)
                trace.append(
                    {
                        "depth": depth,
                        "type": "return_insight",
                        "confidence": confidence,
                    }
                )
                return {
                    "status": "insight",
                    "depth": depth,
                    "objective": objective,
                    "insight": insight,
                    "evidence": evidence,
                    "confidence": confidence,
                    "trace": trace,
                }

            payload: dict[str, Any]
            if fn_name == "peek_window":
                payload = self._query_peek_window(args, latest_frame)
                trace.append({"depth": depth, "type": "peek_window"})
            elif fn_name == "python_repl":
                payload = self._query_python_repl(args, latest_frame)
                trace.append({"depth": depth, "type": "python_repl"})
            elif fn_name == "store_fact":
                payload = self._store_fact_from_args(args)
                trace.append({"depth": depth, "type": "store_fact"})
            elif fn_name == "call_subproblem":
                payload = self._solve_subproblem(
                    latest_frame=latest_frame,
                    objective=str(args.get("objective", objective)),
                    focus=str(args.get("focus", focus)),
                    depth=depth + 1,
                    x=args.get("x"),
                    y=args.get("y"),
                    size=args.get("size"),
                    allow_action=False,
                )
                self._remember_subproblem(
                    objective=str(args.get("objective", objective)),
                    depth=depth + 1,
                    status=str(payload.get("status", "")),
                    confidence=self._safe_float(payload.get("confidence"), 0.2),
                )
                trace.append(
                    {
                        "depth": depth,
                        "type": "subproblem",
                        "payload": self._compact_subproblem_payload(payload),
                    }
                )
            else:
                payload = {"error": f"Unknown tool {fn_name}"}
                trace.append({"depth": depth, "type": "unknown_tool", "name": fn_name})

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(payload),
                }
            )

        if allow_action:
            return {
                "status": "no_action",
                "depth": depth,
                "objective": objective,
                "action": None,
                "trace": trace,
            }

        self._remember_subproblem(objective, depth, "budget_exhausted", 0.2)
        return {
            "status": "insight",
            "depth": depth,
            "objective": objective,
            "insight": "No conclusive subproblem output.",
            "evidence": "Subproblem budget exhausted.",
            "confidence": 0.2,
            "trace": trace,
        }

    def _build_system_prompt(self, depth: int, allow_action: bool) -> str:
        if allow_action:
            return (
                "You are the root controller of a recursive language-model agent. "
                "Use one tool per response. "
                "Inspect state with query tools, delegate with call_subproblem, "
                "use python_repl for code-level context reasoning, "
                "then emit exactly one game action tool."
            )
        remaining_depth = max(0, self.RLM_MAX_DEPTH - depth)
        return (
            "You are solving a bounded recursive subproblem. "
            f"Current recursion depth: {depth}/{self.RLM_MAX_DEPTH}. "
            "Use one tool per response. "
            "You may use call_subproblem to further decompose this problem "
            f"({remaining_depth} recursive level(s) remaining). "
            "Use peek_window or python_repl for inspection and computation. "
            "When you have a conclusion, call return_insight."
        )

    def _build_problem_user_prompt(
        self,
        latest_frame: FrameData,
        objective: str,
        focus: str,
        depth: int,
        x: Any,
        y: Any,
        size: Any,
        allow_action: bool,
    ) -> str:
        payload: dict[str, Any] = {
            "task": {
                "objective": objective,
                "focus": focus,
                "depth": depth,
                "mode": "root" if allow_action else "subproblem",
            },
            "frame": self._frame_summary(latest_frame, include_samples=not allow_action),
            "memory": self._memory_snapshot(),
            "latest_transition": self.transition_log[-1] if self.transition_log else {},
            "budgets": {
                "internal_steps": self.RLM_MAX_INTERNAL_STEPS,
                "subproblem_steps": self._subproblem_step_budget(depth),
                "max_depth": self.RLM_MAX_DEPTH,
            },
        }
        if x is not None and y is not None and size is not None:
            payload["task"]["window"] = {
                "x": self._safe_int(x, default=0, lo=0, hi=63),
                "y": self._safe_int(y, default=0, lo=0, hi=63),
                "size": self._safe_int(size, default=self.RLM_FOCUS_WINDOW_DEFAULT, lo=2, hi=32),
            }
        return json.dumps(payload, indent=2)

    def _build_query_tools(self, include_return_insight: bool) -> list[dict[str, Any]]:
        tools = [
            self._fn_tool(
                name="peek_window",
                description="Inspect a small window from the latest planning grid.",
                properties={
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "size": {"type": "integer"},
                },
            ),
            self._fn_tool(
                name="python_repl",
                description="Execute short Python over persistent ctx, frame, and memory views.",
                properties={"code": {"type": "string"}},
            ),
            self._fn_tool(
                name="call_subproblem",
                description="Recursively solve a focused subproblem.",
                properties={
                    "objective": {"type": "string"},
                    "focus": {
                        "type": "string",
                        "enum": [
                            "full_grid",
                            "window",
                            "recent_transition",
                            "hypothesis_check",
                        ],
                    },
                    "x": {"type": ["integer", "null"]},
                    "y": {"type": ["integer", "null"]},
                    "size": {"type": ["integer", "null"]},
                },
            ),
            self._fn_tool(
                name="store_fact",
                description="Persist a durable observation in external memory.",
                properties={
                    "category": {"type": "string"},
                    "fact": {"type": "string"},
                    "confidence": {"type": "number"},
                },
            ),
        ]

        if include_return_insight:
            tools.append(
                self._fn_tool(
                    name="return_insight",
                    description="Return a concrete conclusion for this subproblem.",
                    properties={
                        "insight": {"type": "string"},
                        "evidence": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                )
            )
        return tools

    def _action_tools(self, latest_frame: FrameData) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for name in self._available_action_names(latest_frame):
            properties: dict[str, Any] = {}
            if name == GameAction.ACTION6.name:
                properties = {
                    "x": {"type": ["integer", "null"]},
                    "y": {"type": ["integer", "null"]},
                }
            tools.append(
                self._fn_tool(
                    name=name,
                    description=f"Emit game action {name}.",
                    properties=properties,
                )
            )
        return tools

    def _fn_tool(
        self,
        name: str,
        description: str,
        properties: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties.keys()),
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

    def _available_action_names(self, latest_frame: FrameData) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        entries = list(getattr(latest_frame, "available_actions", []) or [])
        for row in entries:
            name: Optional[str] = None
            if isinstance(row, int):
                try:
                    name = GameAction.from_id(int(row)).name
                except ValueError:
                    name = None
            elif isinstance(row, str):
                candidate = row.strip().upper()
                if self._is_game_action_name(candidate):
                    name = candidate
            elif hasattr(row, "id"):
                try:
                    raw = getattr(row, "id")
                    rid = int(raw.value) if hasattr(raw, "value") else int(raw)
                    name = GameAction.from_id(rid).name
                except Exception:
                    name = None

            if name and name != GameAction.RESET.name and name not in seen:
                seen.add(name)
                names.append(name)

        if names:
            return names
        return [
            GameAction.ACTION1.name,
            GameAction.ACTION2.name,
            GameAction.ACTION3.name,
            GameAction.ACTION4.name,
        ]

    def _is_game_action_name(self, name: str) -> bool:
        upper = str(name).strip().upper()
        return bool(upper) and any(
            action.name == upper for action in GameAction if action is not GameAction.RESET
        )

    def _build_action_from_tool(self, name: str, args: dict[str, Any]) -> GameAction:
        action = GameAction.from_name(name)
        if action == GameAction.ACTION6:
            action.set_data(
                {
                    "x": self._safe_int(args.get("x"), default=31, lo=0, hi=63),
                    "y": self._safe_int(args.get("y"), default=31, lo=0, hi=63),
                }
            )
            return action
        action.set_data({})
        return action

    def _query_peek_window(
        self, args: dict[str, Any], latest_frame: FrameData
    ) -> dict[str, Any]:
        grid = self._select_planning_grid(latest_frame)
        if not grid:
            return {"error": "empty_grid"}

        x = self._safe_int(args.get("x"), default=0, lo=0, hi=63)
        y = self._safe_int(args.get("y"), default=0, lo=0, hi=63)
        size = self._safe_int(
            args.get("size"), default=self.RLM_FOCUS_WINDOW_DEFAULT, lo=2, hi=32
        )

        h = len(grid)
        w = len(grid[0]) if h else 0
        x1 = max(0, min(w, x + size))
        y1 = max(0, min(h, y + size))

        window: list[list[int]] = []
        for yy in range(y, y1):
            window.append([int(grid[yy][xx]) for xx in range(x, x1)])

        return {
            "x": x,
            "y": y,
            "size": size,
            "shape": [len(window), len(window[0]) if window else 0],
            "window": window,
        }

    def _store_fact_from_args(self, args: dict[str, Any]) -> dict[str, Any]:
        category = str(args.get("category", "observation"))
        fact = str(args.get("fact", "")).strip()
        confidence = self._safe_float(args.get("confidence"), 0.5)
        if not fact:
            return {"stored": False, "error": "empty_fact"}
        self._remember_fact(category, fact, confidence)
        return {
            "stored": True,
            "category": category,
            "confidence": confidence,
            "facts_total": len(self.memory_facts),
        }

    def _query_python_repl(
        self, args: dict[str, Any], latest_frame: FrameData
    ) -> dict[str, Any]:
        code = str(args.get("code", "")).strip()
        if not code:
            return {"ok": False, "error": "empty_code"}
        if not self._is_safe_repl_code(code):
            return {"ok": False, "error": "unsafe_code"}

        ctx = self.context_store.setdefault("globals", {})
        if not isinstance(ctx, dict):
            ctx = {}
            self.context_store["globals"] = ctx

        frame_grid = self._select_planning_grid(latest_frame)
        local_env: dict[str, Any] = {
            "ctx": ctx,
            "frame": frame_grid,
            "facts": self.memory_facts,
            "transitions": self.transition_log,
            "subproblems": self.subproblem_log,
            "result": None,
        }

        stdout_buf = io.StringIO()
        try:
            compiled = compile(code, "<rlm_repl>", "exec")
            with contextlib.redirect_stdout(stdout_buf):
                exec(compiled, self._python_repl_globals(), local_env)
        except Exception as exc:
            logger.debug("python_repl failed", exc_info=True)
            return {
                "ok": False,
                "error": str(exc)[:240],
                "stdout": stdout_buf.getvalue()[:800],
                "ctx_keys": self._context_key_preview(),
            }

        updated_ctx = local_env.get("ctx")
        if isinstance(updated_ctx, dict):
            self.context_store["globals"] = updated_ctx
        self.context_store["runs"] = int(self.context_store.get("runs", 0)) + 1

        return {
            "ok": True,
            "stdout": stdout_buf.getvalue()[:800],
            "result": self._trim_json_value(
                local_env.get("result"),
                max_depth=2,
                max_items=16,
                max_string=400,
            ),
            "ctx_keys": self._context_key_preview(),
            "runs": int(self.context_store.get("runs", 0)),
        }

    def _context_key_preview(self) -> list[str]:
        raw = self.context_store.get("globals", {})
        if not isinstance(raw, dict):
            return []
        return sorted([str(k) for k in raw.keys()])[:24]

    def _python_repl_globals(self) -> dict[str, Any]:
        safe_builtins: dict[str, Any] = {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "print": print,
            "range": range,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
        }
        return {"__builtins__": safe_builtins}

    def _is_safe_repl_code(self, code: str) -> bool:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False

        blocked_names = {
            "__import__",
            "compile",
            "eval",
            "exec",
            "globals",
            "input",
            "locals",
            "open",
            "os",
            "subprocess",
            "sys",
            "vars",
        }
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return False
            if isinstance(node, ast.Attribute) and str(node.attr).startswith("__"):
                return False
            if isinstance(node, ast.Name) and node.id in blocked_names:
                return False
        return True

    def _remember_subproblem(
        self,
        objective: str,
        depth: int,
        status: str,
        confidence: float,
    ) -> None:
        self.subproblem_log.append(
            {
                "turn": self.action_counter,
                "depth": depth,
                "objective": objective[:160],
                "status": status,
                "confidence": round(self._safe_float(confidence, 0.5), 3),
            }
        )
        self.subproblem_log = self.subproblem_log[-self.RLM_MAX_SUBPROBLEMS :]

    def _compact_subproblem_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        keys = ("status", "objective", "depth", "confidence", "insight", "evidence")
        out = {k: payload.get(k) for k in keys}
        if isinstance(payload.get("trace"), list):
            out["trace_len"] = len(payload["trace"])
        return out

    def _ingest_transition(self, frames: list[FrameData], latest_frame: FrameData) -> None:
        if len(frames) < 2:
            return

        previous = frames[-2]
        prev_grid = self._select_planning_grid(previous)
        cur_grid = self._select_planning_grid(latest_frame)
        if not prev_grid or not cur_grid:
            return

        prev_key = self._state_key_for_grid(prev_grid)
        cur_key = self._state_key_for_grid(cur_grid)
        diff = self._grid_diff_summary(prev_grid, cur_grid)
        action_name = self.sent_actions[-1] if self.sent_actions else "UNKNOWN"
        level_delta = int(latest_frame.levels_completed) - int(previous.levels_completed)

        self.transition_log.append(
            {
                "turn": max(0, self.action_counter - 1),
                "action": action_name,
                "prev_state_key": prev_key,
                "state_key": cur_key,
                "level_delta": level_delta,
                "diff": diff,
            }
        )
        self.transition_log = self.transition_log[-self.RLM_MAX_TRANSITIONS :]

        if prev_key and action_name != "UNKNOWN":
            self._remember_action_outcome(
                state_key=prev_key,
                action_name=action_name,
                changed_cells=int(diff.get("changed_cells", 0)),
                level_delta=level_delta,
            )

    def _remember_action_outcome(
        self,
        state_key: str,
        action_name: str,
        changed_cells: int,
        level_delta: int,
    ) -> None:
        state_entry = self.tested_actions_by_state.setdefault(state_key, {})
        entry = state_entry.setdefault(
            action_name,
            {
                "samples": 0.0,
                "sum_changed": 0.0,
                "sum_level_delta": 0.0,
                "max_level_delta": 0.0,
            },
        )
        entry["samples"] += 1.0
        entry["sum_changed"] += float(changed_cells)
        entry["sum_level_delta"] += float(level_delta)
        entry["max_level_delta"] = max(entry["max_level_delta"], float(level_delta))

    def _fallback_action(self, latest_frame: FrameData) -> GameAction:
        candidates = self._available_action_names(latest_frame)
        if not candidates:
            action = GameAction.RESET
            action.set_data({})
            return action

        state_key = self.current_state_key or ""
        tested = self.tested_actions_by_state.get(state_key, {})

        for name in candidates:
            if name not in tested:
                return self._build_action_from_tool(name, {})

        def score(name: str) -> float:
            row = tested.get(name, {})
            samples = max(1.0, float(row.get("samples", 1.0)))
            avg_level = float(row.get("sum_level_delta", 0.0)) / samples
            avg_changed = float(row.get("sum_changed", 0.0)) / samples
            return (avg_level * 1000.0) + avg_changed

        return self._build_action_from_tool(max(candidates, key=score), {})

    def _select_planning_grid(self, frame: FrameData) -> list[list[int]]:
        grids = list(getattr(frame, "frame", []) or [])
        if not grids:
            return []
        first = grids[0]
        return first if isinstance(first, list) else []

    def _state_key_for_grid(self, grid: list[list[int]]) -> str:
        if not grid:
            return ""
        digest = blake2b(digest_size=12)
        for row in grid:
            digest.update(bytes(int(v) & 0xFF for v in row))
        return digest.hexdigest()

    def _grid_diff_summary(
        self, prev_grid: list[list[int]], cur_grid: list[list[int]]
    ) -> dict[str, Any]:
        h = min(len(prev_grid), len(cur_grid))
        w = min(len(prev_grid[0]), len(cur_grid[0])) if h > 0 else 0
        changed = 0
        x0 = y0 = 10**9
        x1 = y1 = -1

        for y in range(h):
            for x in range(w):
                if int(prev_grid[y][x]) == int(cur_grid[y][x]):
                    continue
                changed += 1
                x0 = min(x0, x)
                y0 = min(y0, y)
                x1 = max(x1, x)
                y1 = max(y1, y)

        bbox = None
        if changed > 0:
            bbox = {"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1}

        return {"changed_cells": changed, "bbox": bbox}

    def _memory_snapshot(self) -> dict[str, Any]:
        state_key = self.current_state_key or ""
        return {
            "state_key": state_key,
            "state_visits": int(self.state_visits.get(state_key, 0)),
            "facts": self.memory_facts[-8:],
            "recent_transitions": self.transition_log[-4:],
            "recent_subproblems": self.subproblem_log[-4:],
            "tested_actions_for_state": self.tested_actions_by_state.get(state_key, {}),
            "python_repl_runs": int(self.context_store.get("runs", 0)),
            "context_globals_keys": self._context_key_preview(),
        }

    def _frame_summary(
        self, latest_frame: FrameData, include_samples: bool
    ) -> dict[str, Any]:
        grids = list(getattr(latest_frame, "frame", []) or [])
        return {
            "state": latest_frame.state.name,
            "levels_completed": int(latest_frame.levels_completed),
            "win_levels": int(latest_frame.win_levels),
            "available_actions": self._available_action_names(latest_frame),
            "grid_count": len(grids),
            "state_key": self._state_key_for_grid(self._select_planning_grid(latest_frame)) or None,
            "grids": [
                self._grid_stats(grid, include_samples)
                for grid in grids[: self.RLM_MAX_GRID_SUMMARIES]
                if isinstance(grid, list)
            ],
        }

    def _grid_stats(self, grid: list[list[int]], include_samples: bool) -> dict[str, Any]:
        if not grid:
            return {"shape": [0, 0], "non_zero_cells": 0, "unique_values": 0}

        h = len(grid)
        w = len(grid[0]) if h else 0
        histogram: dict[int, int] = defaultdict(int)
        non_zero = 0

        for row in grid:
            for value in row:
                v = int(value)
                histogram[v] += 1
                if v != 0:
                    non_zero += 1

        top = sorted(histogram.items(), key=lambda item: item[1], reverse=True)[
            : self.RLM_HISTOGRAM_TOP_K
        ]
        out: dict[str, Any] = {
            "shape": [h, w],
            "non_zero_cells": non_zero,
            "unique_values": len(histogram),
            "histogram_top": {str(k): int(v) for k, v in top},
        }
        if include_samples:
            out["sample_rows"] = [
                [int(v) for v in grid[i][: min(16, w)]] for i in range(min(8, h))
            ]
        return out

    def _remember_fact(self, category: str, fact: str, confidence: float) -> None:
        self.memory_facts.append(
            {
                "category": category[:64],
                "fact": fact[:400],
                "confidence": round(self._safe_float(confidence, 0.5), 3),
                "turn": int(self.action_counter),
            }
        )
        self.memory_facts = self.memory_facts[-self.RLM_MAX_FACTS :]

    def _build_replay_reasoning(
        self,
        latest_frame: FrameData,
        selected_action: GameAction,
        turn_trace: list[dict[str, Any]],
        forced_action_used: bool,
    ) -> dict[str, Any]:
        latest = self.transition_log[-1] if self.transition_log else {}
        diff = latest.get("diff") if isinstance(latest.get("diff"), dict) else {}
        return {
            "agent": "RLM",
            "model": self.MODEL,
            "action": selected_action.name,
            "forced": forced_action_used,
            "turn": self.action_counter,
            "state": latest_frame.state.name,
            "levels_completed": int(latest_frame.levels_completed),
            "state_key": self.current_state_key,
            "latest_transition": {
                "level_delta": latest.get("level_delta"),
                "changed_cells": diff.get("changed_cells"),
                "bbox": diff.get("bbox"),
            },
            "facts": len(self.memory_facts),
            "transitions": len(self.transition_log),
            "subproblems": len(self.subproblem_log),
            "trace": turn_trace[-8:],
        }

    def _record_sent_action(self, action: GameAction) -> None:
        self.sent_actions.append(action.name)
        self.sent_actions = self.sent_actions[-128:]

    def _call_chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        tool_required: bool = False,
    ) -> dict[str, Any]:
        create_kwargs: dict[str, Any] = {"model": self.MODEL, "messages": messages}
        if tools:
            create_kwargs["tools"] = tools
            create_kwargs["tool_choice"] = "required" if tool_required else "auto"
        if self.REASONING_EFFORT is not None:
            create_kwargs["reasoning_effort"] = self.REASONING_EFFORT

        try:
            response = self.client.chat.completions.create(**create_kwargs)
        except openai.BadRequestError as exc:
            raise RuntimeError(f"OpenAI request failed in RLM agent: {exc}") from exc

        self.capture_reasoning_from_response(response)
        usage = getattr(response, "usage", None)
        total_tokens = int(getattr(usage, "total_tokens", 0)) if usage else 0
        content = response.choices[0].message.content or ""
        self.track_tokens(total_tokens, content)
        return response.choices[0].message.model_dump(exclude_none=True)

    def _json_object(self, raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str) or not raw.strip():
            return {}
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _safe_int(self, value: Any, default: int, lo: int, hi: int) -> int:
        try:
            num = int(value)
        except (TypeError, ValueError):
            num = default
        return max(lo, min(hi, num))

    def _safe_float(self, value: Any, default: float) -> float:
        try:
            num = float(value)
        except (TypeError, ValueError):
            num = default
        return max(0.0, min(1.0, num))

    def _subproblem_step_budget(self, depth: int) -> int:
        return max(1, self.RLM_MAX_SUB_STEPS - max(0, depth - 1))
