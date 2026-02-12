import json
import os
from typing import Any, Optional

import openai
from arcengine import FrameData, GameAction, GameState
from openai import OpenAI as OpenAIClient

from .llm_agents import ReasoningLLM


class RLM(ReasoningLLM):
    """Recursive Language Model style agent for ARC-AGI-3.

    This agent keeps long-lived state outside the model context and lets the model:
    1) choose a direct game action, or
    2) call recursive subproblems on focused slices of the frame.
    """

    MAX_ACTIONS = 120
    MODEL = "o4-mini"
    MODEL_REQUIRES_TOOLS = True
    REASONING_EFFORT = "medium"

    RLM_MAX_INTERNAL_STEPS = 6
    RLM_MAX_SUB_STEPS = 4
    RLM_MAX_DEPTH = 2
    RLM_MAX_FACTS = 64
    RLM_MAX_TRANSITIONS = 40
    RLM_MAX_SUBPROBLEMS = 40
    RLM_FRAME_SAMPLE_SIZE = 12
    RLM_MAX_GRID_SUMMARIES = 2
    RLM_HISTOGRAM_TOP_K = 8
    RLM_FOCUS_WINDOW_DEFAULT = 12
    RLM_MEMORY_FACTS_VIEW = 8
    RLM_MEMORY_TRANSITIONS_VIEW = 6
    RLM_MEMORY_SUBPROBLEMS_VIEW = 4

    memory_facts: list[dict[str, Any]]
    transition_log: list[dict[str, Any]]
    subproblem_log: list[dict[str, Any]]
    client: OpenAIClient

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.memory_facts = []
        self.transition_log = []
        self.subproblem_log = []
        self._load_runtime_config()
        self.client = OpenAIClient(api_key=os.environ.get("OPENAI_API_KEY", ""))

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        """Used for recorder metadata in parent cleanup."""
        return (
            "RLM agent prompt is generated dynamically per turn from external state, "
            "transition summaries, and compact frame analytics."
        )

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        # Bootstrap and level restarts should remain deterministic.
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            action = GameAction.RESET
            action.reasoning = {
                "agent": "RLM",
                "mode": "bootstrap_reset",
                "state": latest_frame.state.name,
                "action_counter": self.action_counter,
            }
            return action

        self._ingest_transition(frames, latest_frame)

        result = self._run_root_controller(latest_frame)
        action = result["action"]
        action.reasoning = result["reasoning"]
        return action

    def _run_root_controller(self, latest_frame: FrameData) -> dict[str, Any]:
        tools = self._build_root_tools()
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": self._build_root_system_prompt(),
            },
            {
                "role": "user",
                "content": self._build_root_user_prompt(latest_frame),
            },
        ]
        turn_trace: list[dict[str, Any]] = []

        for _ in range(self._internal_step_budget()):
            message = self._call_chat(messages, tools)
            tool_calls = message.get("tool_calls", [])
            if not tool_calls:
                break

            messages.append(message)
            action_from_step: Optional[GameAction] = None

            for tool_call in tool_calls:
                fn_name = tool_call["function"]["name"]
                args = self._parse_tool_arguments(tool_call["function"].get("arguments"))

                if fn_name in self._action_names():
                    action_from_step = self._build_action_from_tool(fn_name, args)
                    turn_trace.append(
                        {
                            "type": "action",
                            "action": action_from_step.name,
                            "args": args,
                        }
                    )
                    break

                if fn_name == "call_subproblem":
                    sub_result = self._solve_subproblem(
                        latest_frame=latest_frame,
                        objective=str(args.get("objective", "")),
                        focus=str(args.get("focus", "full_grid")),
                        depth=1,
                        x=args.get("x"),
                        y=args.get("y"),
                        size=args.get("size"),
                    )
                    self.subproblem_log.append(
                        {
                            "turn": self.action_counter,
                            "objective": str(args.get("objective", "")),
                            "focus": str(args.get("focus", "full_grid")),
                            "result": sub_result,
                        }
                    )
                    self.subproblem_log = self.subproblem_log[
                        -self.RLM_MAX_SUBPROBLEMS :
                    ]
                    turn_trace.append(
                        {
                            "type": "subproblem",
                            "request": args,
                            "result": sub_result,
                        }
                    )
                    messages.append(
                        self._tool_result_message(
                            tool_call_id=tool_call["id"],
                            payload=sub_result,
                        )
                    )
                    continue

                if fn_name == "store_fact":
                    entry = self._store_fact(
                        category=str(args.get("category", "heuristic")),
                        fact=str(args.get("fact", "")),
                        confidence=self._safe_float(args.get("confidence"), default=0.5),
                    )
                    turn_trace.append({"type": "fact", "entry": entry})
                    messages.append(
                        self._tool_result_message(
                            tool_call_id=tool_call["id"],
                            payload={"stored": entry},
                        )
                    )
                    continue

                messages.append(
                    self._tool_result_message(
                        tool_call_id=tool_call["id"],
                        payload={"error": f"Unknown tool: {fn_name}"},
                    )
                )

            if action_from_step is not None:
                reasoning = {
                    "agent": "RLM",
                    "model": self.MODEL,
                    "selected_action": action_from_step.name,
                    "internal_trace": turn_trace[-6:],
                    "facts_total": len(self.memory_facts),
                    "subproblems_total": len(self.subproblem_log),
                    "transitions_total": len(self.transition_log),
                    "reasoning_tokens": self._last_reasoning_tokens,
                    "total_reasoning_tokens": self._total_reasoning_tokens,
                }
                return {"action": action_from_step, "reasoning": reasoning}

        fallback_action = self._fallback_action(latest_frame)
        return {
            "action": fallback_action,
            "reasoning": {
                "agent": "RLM",
                "model": self.MODEL,
                "selected_action": fallback_action.name,
                "mode": "fallback",
                "facts_total": len(self.memory_facts),
                "subproblems_total": len(self.subproblem_log),
                "transitions_total": len(self.transition_log),
            },
        }

    def _solve_subproblem(
        self,
        latest_frame: FrameData,
        objective: str,
        focus: str,
        depth: int,
        x: Any = None,
        y: Any = None,
        size: Any = None,
    ) -> dict[str, Any]:
        if depth > self.RLM_MAX_DEPTH:
            return {
                "objective": objective,
                "status": "depth_limit",
                "depth": depth,
                "insight": "Reached recursion depth limit.",
            }

        tools = self._build_subproblem_tools()
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": self._build_subproblem_system_prompt(depth=depth),
            },
            {
                "role": "user",
                "content": self._build_subproblem_user_prompt(
                    latest_frame=latest_frame,
                    objective=objective,
                    focus=focus,
                    x=x,
                    y=y,
                    size=size,
                    depth=depth,
                ),
            },
        ]

        for _ in range(self._subproblem_step_budget(depth)):
            message = self._call_chat(messages, tools)
            tool_calls = message.get("tool_calls", [])
            if not tool_calls:
                break

            messages.append(message)

            for tool_call in tool_calls:
                fn_name = tool_call["function"]["name"]
                args = self._parse_tool_arguments(tool_call["function"].get("arguments"))

                if fn_name == "return_insight":
                    return {
                        "objective": objective,
                        "status": "solved",
                        "depth": depth,
                        "insight": str(args.get("insight", "")).strip(),
                        "evidence": str(args.get("evidence", "")).strip(),
                        "confidence": self._safe_float(
                            args.get("confidence"), default=0.5
                        ),
                    }

                if fn_name == "call_subproblem":
                    nested = self._solve_subproblem(
                        latest_frame=latest_frame,
                        objective=str(args.get("objective", "")),
                        focus=str(args.get("focus", "full_grid")),
                        depth=depth + 1,
                        x=args.get("x"),
                        y=args.get("y"),
                        size=args.get("size"),
                    )
                    messages.append(
                        self._tool_result_message(
                            tool_call_id=tool_call["id"], payload=nested
                        )
                    )
                    continue

                if fn_name == "store_fact":
                    entry = self._store_fact(
                        category=str(args.get("category", "subproblem_fact")),
                        fact=str(args.get("fact", "")),
                        confidence=self._safe_float(args.get("confidence"), default=0.5),
                    )
                    messages.append(
                        self._tool_result_message(
                            tool_call_id=tool_call["id"],
                            payload={"stored": entry},
                        )
                    )
                    continue

                messages.append(
                    self._tool_result_message(
                        tool_call_id=tool_call["id"],
                        payload={"error": f"Unknown tool: {fn_name}"},
                    )
                )

        return {
            "objective": objective,
            "status": "incomplete",
            "depth": depth,
            "insight": "No conclusive insight returned by subproblem solver.",
        }

    def _call_chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        create_kwargs: dict[str, Any] = {
            "model": self.MODEL,
            "messages": messages,
            "tools": tools,
            "tool_choice": "required",
        }
        if self.REASONING_EFFORT is not None:
            create_kwargs["reasoning_effort"] = self.REASONING_EFFORT

        try:
            response = self.client.chat.completions.create(**create_kwargs)
        except openai.BadRequestError as exc:
            raise RuntimeError(
                f"OpenAI request failed in RLM agent: {exc}"
            ) from exc

        self.capture_reasoning_from_response(response)
        usage = getattr(response, "usage", None)
        total_tokens = getattr(usage, "total_tokens", 0) if usage else 0
        content = response.choices[0].message.content or ""
        self.track_tokens(total_tokens, content)

        return response.choices[0].message.model_dump(exclude_none=True)

    def _build_root_system_prompt(self) -> str:
        return (
            "You are the root controller of a recursive language model agent for ARC-AGI-3. "
            "Pick exactly one tool per response. "
            "Use call_subproblem to reason on smaller objectives when uncertain. "
            "Use store_fact to persist durable hypotheses in external memory. "
            "When ready, emit one game action tool call."
        )

    def _build_subproblem_system_prompt(self, depth: int) -> str:
        return (
            "You are solving a bounded ARC subproblem. "
            f"Current recursion depth is {depth}. "
            "Pick exactly one tool per response. "
            "Use return_insight when you have a concrete conclusion."
        )

    def _build_root_user_prompt(self, latest_frame: FrameData) -> str:
        payload = {
            "objective": "Choose the next game action.",
            "frame": self._frame_summary(latest_frame),
            "memory": self._memory_snapshot(),
            "budgets": {
                "internal_steps": self._internal_step_budget(),
                "subproblem_steps": self.RLM_MAX_SUB_STEPS,
                "max_depth": self.RLM_MAX_DEPTH,
            },
        }
        return json.dumps(payload, indent=2)

    def _build_subproblem_user_prompt(
        self,
        latest_frame: FrameData,
        objective: str,
        focus: str,
        x: Any,
        y: Any,
        size: Any,
        depth: int,
    ) -> str:
        focus_payload: dict[str, Any] = {
            "objective": objective,
            "focus": focus,
            "depth": depth,
        }
        window = self._safe_window_params(x=x, y=y, size=size)
        if window:
            focus_payload["window"] = window

        payload = {
            "task": focus_payload,
            "frame": self._frame_summary(latest_frame, window=window),
            "memory": self._memory_snapshot(),
            "budgets": {
                "subproblem_steps": self._subproblem_step_budget(depth),
                "max_depth": self.RLM_MAX_DEPTH,
            },
        }
        return json.dumps(payload, indent=2)

    def _build_root_tools(self) -> list[dict[str, Any]]:
        return self._action_tools() + [
            {
                "type": "function",
                "function": {
                    "name": "call_subproblem",
                    "description": "Create and solve a focused reasoning subproblem before choosing an action.",
                    "parameters": {
                        "type": "object",
                        "properties": {
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
                            "x": {"type": "integer"},
                            "y": {"type": "integer"},
                            "size": {"type": "integer"},
                        },
                        "required": ["objective", "focus"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "store_fact",
                    "description": "Persist a durable observation to external memory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "category": {"type": "string"},
                            "fact": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["category", "fact", "confidence"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
        ]

    def _build_subproblem_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "call_subproblem",
                    "description": "Recursively split this subproblem into a smaller one.",
                    "parameters": {
                        "type": "object",
                        "properties": {
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
                            "x": {"type": "integer"},
                            "y": {"type": "integer"},
                            "size": {"type": "integer"},
                        },
                        "required": ["objective", "focus"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "store_fact",
                    "description": "Persist a durable subproblem observation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "category": {"type": "string"},
                            "fact": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["category", "fact", "confidence"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "return_insight",
                    "description": "Finish this subproblem with a concrete conclusion.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "insight": {"type": "string"},
                            "evidence": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["insight", "evidence", "confidence"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
        ]

    def _action_tools(self) -> list[dict[str, Any]]:
        return super().build_tools()

    def _action_names(self) -> set[str]:
        return {tool["function"]["name"] for tool in self._action_tools()}

    def _build_action_from_tool(self, name: str, args: dict[str, Any]) -> GameAction:
        action = GameAction.from_name(name)

        if action == GameAction.ACTION6:
            x = self._safe_int(args.get("x"), default=0, lo=0, hi=63)
            y = self._safe_int(args.get("y"), default=0, lo=0, hi=63)
            action.set_data({"x": x, "y": y})
        else:
            action.set_data({})

        return action

    def _fallback_action(self, latest_frame: FrameData) -> GameAction:
        available = self._normalize_available_actions(latest_frame.available_actions)
        for preferred in [
            GameAction.ACTION1.name,
            GameAction.ACTION2.name,
            GameAction.ACTION3.name,
            GameAction.ACTION4.name,
            GameAction.ACTION5.name,
        ]:
            if preferred in available:
                return GameAction.from_name(preferred)

        for name in available:
            if name in {GameAction.RESET.name, GameAction.ACTION6.name}:
                continue
            try:
                return GameAction.from_name(name)
            except ValueError:
                continue

        return GameAction.ACTION1

    def _store_fact(
        self, category: str, fact: str, confidence: float, source: str = "llm"
    ) -> dict[str, Any]:
        clean_category = category.strip()[:64]
        clean_fact = fact.strip()[:600]
        if not clean_fact:
            return {
                "turn": self.action_counter,
                "category": clean_category,
                "fact": "",
                "confidence": 0.0,
                "source": source,
                "ignored": True,
            }

        entry = {
            "turn": self.action_counter,
            "category": clean_category,
            "fact": clean_fact,
            "confidence": max(0.0, min(float(confidence), 1.0)),
            "source": source[:32],
            "observations": 1,
        }
        for existing in reversed(self.memory_facts):
            if (
                existing.get("category") == entry["category"]
                and existing.get("fact") == entry["fact"]
            ):
                existing["turn"] = self.action_counter
                existing["confidence"] = max(
                    float(existing.get("confidence", 0.0)), entry["confidence"]
                )
                existing["observations"] = int(existing.get("observations", 1)) + 1
                return existing

        self.memory_facts.append(entry)
        self.memory_facts = self.memory_facts[-self.RLM_MAX_FACTS :]
        return entry

    def _ingest_transition(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> None:
        if len(frames) < 2:
            return

        previous = frames[-2]
        previous_grid = previous.frame[-1] if previous.frame else []
        latest_grid = latest_frame.frame[-1] if latest_frame.frame else []

        action_name = "UNKNOWN"
        if latest_frame.action_input and latest_frame.action_input.id:
            action_name = latest_frame.action_input.id.name

        summary = {
            "turn": self.action_counter,
            "action": action_name,
            "prev_state": previous.state.name,
            "state": latest_frame.state.name,
            "prev_levels_completed": previous.levels_completed,
            "levels_completed": latest_frame.levels_completed,
            "diff": self._grid_diff_summary(previous_grid, latest_grid),
        }
        self.transition_log.append(summary)
        self.transition_log = self.transition_log[-self.RLM_MAX_TRANSITIONS :]
        self._auto_derive_heuristics(summary)

    def _memory_snapshot(self) -> dict[str, Any]:
        return {
            "facts": self.memory_facts[-self.RLM_MEMORY_FACTS_VIEW :],
            "fact_category_counts": self._fact_category_counts(limit=24),
            "recent_transitions": self.transition_log[
                -self.RLM_MEMORY_TRANSITIONS_VIEW :
            ],
            "recent_subproblems": self.subproblem_log[
                -self.RLM_MEMORY_SUBPROBLEMS_VIEW :
            ],
        }

    def _frame_summary(
        self,
        latest_frame: FrameData,
        window: Optional[dict[str, int]] = None,
    ) -> dict[str, Any]:
        grids = latest_frame.frame or []
        final_grid = grids[-1] if grids else []
        grid_summaries = [
            self._grid_stats(grid) for grid in grids[-self.RLM_MAX_GRID_SUMMARIES :]
        ]

        summary: dict[str, Any] = {
            "state": latest_frame.state.name,
            "levels_completed": latest_frame.levels_completed,
            "win_levels": latest_frame.win_levels,
            "available_actions": self._normalize_available_actions(
                latest_frame.available_actions
            ),
            "grid_count": len(grids),
            "grids": grid_summaries,
        }

        if final_grid:
            sample = self.RLM_FRAME_SAMPLE_SIZE
            summary[f"sample_top_left_{sample}x{sample}"] = [
                row[:sample] for row in final_grid[:sample]
            ]

        if window and final_grid:
            wx = window["x"]
            wy = window["y"]
            size = window["size"]
            h = len(final_grid)
            w = len(final_grid[0]) if h > 0 else 0
            x0 = max(0, min(wx, max(0, w - 1)))
            y0 = max(0, min(wy, max(0, h - 1)))
            x1 = min(w, x0 + size)
            y1 = min(h, y0 + size)
            focus_rows = [row[x0:x1] for row in final_grid[y0:y1]]
            summary["focus_window"] = {
                "x": x0,
                "y": y0,
                "size": size,
                "rows": focus_rows,
            }

        return summary

    def _grid_stats(self, grid: list[list[int]]) -> dict[str, Any]:
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0
        histogram: dict[int, int] = {}
        non_zero = 0
        for row in grid:
            for value in row:
                histogram[value] = histogram.get(value, 0) + 1
                if value != 0:
                    non_zero += 1

        top_hist = sorted(histogram.items(), key=lambda item: item[1], reverse=True)[
            : self.RLM_HISTOGRAM_TOP_K
        ]
        return {
            "shape": [h, w],
            "non_zero_cells": non_zero,
            "unique_values": len(histogram),
            "histogram_top": {str(k): v for k, v in top_hist},
        }

    def _fact_category_counts(self, limit: int) -> dict[str, int]:
        counts: dict[str, int] = {}
        for fact in self.memory_facts[-limit:]:
            category = str(fact.get("category", "uncategorized"))
            counts[category] = counts.get(category, 0) + 1
        return counts

    def _auto_derive_heuristics(self, summary: dict[str, Any]) -> None:
        action = str(summary.get("action", "UNKNOWN"))
        prev_state = str(summary.get("prev_state", "UNKNOWN"))
        state = str(summary.get("state", "UNKNOWN"))
        prev_level = int(summary.get("prev_levels_completed", 0))
        level = int(summary.get("levels_completed", 0))
        diff = summary.get("diff", {})
        changed = int(diff.get("changed_cells", 0))
        ratio = float(diff.get("changed_ratio", 0.0))
        bbox = diff.get("bbox")

        if action in {
            GameAction.ACTION1.name,
            GameAction.ACTION2.name,
            GameAction.ACTION3.name,
            GameAction.ACTION4.name,
            GameAction.ACTION5.name,
        }:
            if changed == 0:
                self._store_fact(
                    "movement_blocked",
                    f"{action} caused no visible cell changes in the latest transition.",
                    0.72,
                    source="auto_heuristic",
                )
            elif changed <= 4:
                self._store_fact(
                    "movement_local",
                    f"{action} usually makes local edits ({changed} changed cells).",
                    0.58,
                    source="auto_heuristic",
                )
            elif ratio >= 0.12:
                self._store_fact(
                    "movement_global",
                    f"{action} can trigger broad board updates ({ratio:.1%} changed).",
                    0.56,
                    source="auto_heuristic",
                )

        if level > prev_level:
            delta = level - prev_level
            self._store_fact(
                "progress_signal",
                f"{action} increased levels_completed by {delta}.",
                0.9,
                source="auto_heuristic",
            )

        if state == GameState.WIN.name:
            self._store_fact(
                "terminal_win",
                f"{action} reached WIN from {prev_state}.",
                0.95,
                source="auto_heuristic",
            )
        elif state == GameState.GAME_OVER.name:
            self._store_fact(
                "terminal_loss",
                f"{action} resulted in GAME_OVER from {prev_state}.",
                0.9,
                source="auto_heuristic",
            )

        if bbox and action == GameAction.ACTION5.name and changed > 0:
            self._store_fact(
                "interaction_area",
                f"{action} changed region bounded by {bbox}.",
                0.62,
                source="auto_heuristic",
            )

    def _grid_diff_summary(
        self, previous_grid: list[list[int]], latest_grid: list[list[int]]
    ) -> dict[str, Any]:
        if not previous_grid or not latest_grid:
            return {"changed_cells": 0, "changed_ratio": 0.0, "bbox": None}

        h = min(len(previous_grid), len(latest_grid))
        w = min(len(previous_grid[0]), len(latest_grid[0]))

        changed = 0
        min_x = w
        min_y = h
        max_x = -1
        max_y = -1

        for y in range(h):
            for x in range(w):
                if previous_grid[y][x] != latest_grid[y][x]:
                    changed += 1
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)

        bbox: Optional[dict[str, int]]
        if changed == 0:
            bbox = None
        else:
            bbox = {"x_min": min_x, "y_min": min_y, "x_max": max_x, "y_max": max_y}

        total = max(h * w, 1)
        return {
            "changed_cells": changed,
            "changed_ratio": round(changed / total, 4),
            "bbox": bbox,
        }

    def _normalize_available_actions(self, actions: Any) -> list[str]:
        names: list[str] = []
        if not actions:
            return [a.name for a in GameAction]

        for item in actions:
            if hasattr(item, "name"):
                names.append(str(item.name))
                continue

            if isinstance(item, str):
                names.append(item.upper())
                continue

            if isinstance(item, int):
                try:
                    names.append(GameAction.from_id(item).name)
                except ValueError:
                    continue

        deduped: list[str] = []
        for name in names:
            if name not in deduped:
                deduped.append(name)
        return deduped

    def _tool_result_message(self, tool_call_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(payload),
        }

    def _parse_tool_arguments(self, raw: Optional[str]) -> dict[str, Any]:
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
            return {}
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

    def _safe_window_params(
        self, x: Any, y: Any, size: Any
    ) -> Optional[dict[str, int]]:
        if x is None or y is None or size is None:
            return None
        safe_size = self._safe_int(size, default=self.RLM_FOCUS_WINDOW_DEFAULT, lo=2, hi=32)
        return {
            "x": self._safe_int(x, default=0, lo=0, hi=63),
            "y": self._safe_int(y, default=0, lo=0, hi=63),
            "size": safe_size,
        }

    def _load_runtime_config(self) -> None:
        self.RLM_MAX_INTERNAL_STEPS = self._env_int(
            "RLM_MAX_INTERNAL_STEPS", self.RLM_MAX_INTERNAL_STEPS, lo=1, hi=12
        )
        self.RLM_MAX_SUB_STEPS = self._env_int(
            "RLM_MAX_SUB_STEPS", self.RLM_MAX_SUB_STEPS, lo=1, hi=8
        )
        self.RLM_MAX_DEPTH = self._env_int(
            "RLM_MAX_DEPTH", self.RLM_MAX_DEPTH, lo=1, hi=4
        )
        self.RLM_MAX_FACTS = self._env_int(
            "RLM_MAX_FACTS", self.RLM_MAX_FACTS, lo=8, hi=256
        )
        self.RLM_MAX_TRANSITIONS = self._env_int(
            "RLM_MAX_TRANSITIONS", self.RLM_MAX_TRANSITIONS, lo=8, hi=200
        )
        self.RLM_MAX_SUBPROBLEMS = self._env_int(
            "RLM_MAX_SUBPROBLEMS", self.RLM_MAX_SUBPROBLEMS, lo=8, hi=200
        )
        self.RLM_FRAME_SAMPLE_SIZE = self._env_int(
            "RLM_FRAME_SAMPLE_SIZE", self.RLM_FRAME_SAMPLE_SIZE, lo=4, hi=20
        )
        self.RLM_MAX_GRID_SUMMARIES = self._env_int(
            "RLM_MAX_GRID_SUMMARIES", self.RLM_MAX_GRID_SUMMARIES, lo=1, hi=4
        )
        self.RLM_HISTOGRAM_TOP_K = self._env_int(
            "RLM_HISTOGRAM_TOP_K", self.RLM_HISTOGRAM_TOP_K, lo=4, hi=16
        )
        self.RLM_FOCUS_WINDOW_DEFAULT = self._env_int(
            "RLM_FOCUS_WINDOW_DEFAULT", self.RLM_FOCUS_WINDOW_DEFAULT, lo=2, hi=32
        )

    def _internal_step_budget(self) -> int:
        budget = self.RLM_MAX_INTERNAL_STEPS
        recent = self.transition_log[-3:]
        if not recent:
            return budget

        unchanged = sum(
            1
            for transition in recent
            if int(transition.get("diff", {}).get("changed_cells", 0)) == 0
        )
        if unchanged >= 2:
            return min(budget + 1, 12)

        progressed = any(
            int(transition.get("levels_completed", 0))
            > int(transition.get("prev_levels_completed", 0))
            for transition in recent
        )
        if progressed:
            return max(2, budget - 1)
        return budget

    def _subproblem_step_budget(self, depth: int) -> int:
        return max(1, self.RLM_MAX_SUB_STEPS - (depth - 1))

    def _env_int(self, name: str, default: int, lo: int, hi: int) -> int:
        raw = os.getenv(name)
        if raw is None:
            return default
        return self._safe_int(raw, default=default, lo=lo, hi=hi)
