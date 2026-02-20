import asyncio
import logging
import os
import textwrap
import traceback
import uuid
from typing import Any, Optional

from arcengine import FrameData, GameAction, GameState
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    SystemMessage,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from ...agent import Agent
from .claude_recorder import ClaudeCodeRecorder
from .claude_tools import create_arc_tools_server

logger = logging.getLogger()


def select_animation_frames(
    frames: list[Any], max_frames: int = 7
) -> list[tuple[int, Any]]:
    """Select evenly-spaced frames from an animation, always including first and last.

    Returns a list of (original_index, frame_data) tuples.
    """
    n = len(frames)
    if n == 0:
        return []
    if n <= max_frames:
        return [(i, frames[i]) for i in range(n)]
    # Pick max_frames indices evenly spaced, always including 0 and n-1
    indices = [round(i * (n - 1) / (max_frames - 1)) for i in range(max_frames)]
    # Deduplicate while preserving order (can happen with very small n relative to max)
    seen = set()
    unique = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique.append(idx)
    return [(idx, frames[idx]) for idx in unique]


class ClaudeCodeAgent(Agent):
    MAX_ACTIONS: int = int(os.getenv("STEP_COUNT", 80))
    MODEL: str = "claude-opus-4-6"
    MAX_CONSECUTIVE_ERRORS: int = 3
    ACTION_TOOL_MAP: dict[str, GameAction] = {
        "mcp__arc-game-tools__reset_game": GameAction.RESET,
        "mcp__arc-game-tools__action1_move_up": GameAction.ACTION1,
        "mcp__arc-game-tools__action2_move_down": GameAction.ACTION2,
        "mcp__arc-game-tools__action3_move_left": GameAction.ACTION3,
        "mcp__arc-game-tools__action4_move_right": GameAction.ACTION4,
        "mcp__arc-game-tools__action5_interact": GameAction.ACTION5,
        "mcp__arc-game-tools__action6_click": GameAction.ACTION6,
        "mcp__arc-game-tools__action7_undo": GameAction.ACTION7,
    }

    token_counter: int
    step_counter: int
    mcp_server: Any
    latest_reasoning: str
    latest_reasoning_dict: dict[str, Any]
    claude_recorder: Optional[ClaudeCodeRecorder]
    captured_messages: list[Any]
    current_prompt: str
    result_message: Optional[Any]
    current_frame: Optional[FrameData]
    session_id: Optional[str]
    consecutive_errors: int
    previous_action_info: Optional[dict[str, str]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.token_counter = 0
        self.step_counter = 0
        self.cumulative_cost_usd = 0.0
        self.latest_reasoning = ""
        self.latest_reasoning_dict = {}
        self.current_frame = None
        self.session_id = None
        self.consecutive_errors = 0
        self.previous_action_info: Optional[dict[str, str]] = None
        self.action_history: list[str] = []
        self.mcp_server = create_arc_tools_server(self)
        self.notes_session_id = str(uuid.uuid4())
        self.notes_dir = os.path.abspath(
            f"./game_notes/{self.game_id}_{self.notes_session_id}"
        )
        os.makedirs(self.notes_dir, exist_ok=True)
        self.notes_path = os.path.join(self.notes_dir, "notes.md")
        with open(self.notes_path, "w") as f:
            f.write(
                f"# Game {self.game_id}\n"
                "\n"
                "## Game Mechanics (carry across levels)\n"
                "(Record confirmed, general mechanics here. These persist when levels change.)\n"
                "\n"
                "## Current Level\n"
                "\n"
                "### Hypothesis\n"
                "(Your current best theory about how THIS level works. Include confidence: LOW/MEDIUM/HIGH. "
                "Replace — don't append — when you form a better one.)\n"
                "\n"
                "### Key Positions\n"
                "(Important objects, buttons, targets with grid coordinates. Update in place when things move.)\n"
                "\n"
                "### Failed Approaches (this level)\n"
                "(What you tried that didn't work. Be specific so you don't retry it.)\n"
                "\n"
                "### Current Plan\n"
                "(Your step-by-step plan with resource budget. Include how many actions/energy it requires.)\n"
                "\n"
                "## Other Notes\n"
                "(Optional space for any other observations, patterns, or ideas worth remembering.)\n"
            )
        logger.info(f"Created notes file: {self.notes_path}")
        self.captured_messages = []
        self.current_prompt = ""
        self.result_message = None

        if kwargs.get("record", False):
            self.claude_recorder = ClaudeCodeRecorder(
                game_id=kwargs.get("game_id", "unknown"),
                agent_name=self.agent_name,
                session_id=self.notes_session_id,
            )
        else:
            self.claude_recorder = None

        logging.getLogger("anthropic").setLevel(logging.CRITICAL)
        logging.getLogger("httpx").setLevel(logging.CRITICAL)

        self._auto_reset_from_game_over = False

        # Persistent async infrastructure — single Claude Code subprocess for all turns
        self._loop = asyncio.new_event_loop()
        self._client: Optional[ClaudeSDKClient] = None
        self._client_connected = False

    @property
    def name(self) -> str:
        sanitized_model_name = self.MODEL.replace("/", "-").replace(":", "-")
        return f"{super().name}.{sanitized_model_name}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return any(
            [
                latest_frame.state is GameState.WIN,
            ]
        )

    def do_action_request(self, action: GameAction) -> FrameData:
        data = action.action_data.model_dump()

        if self.latest_reasoning_dict:
            data["reasoning"] = self.latest_reasoning_dict
            logger.info(
                f"Added reasoning to action request: {len(str(self.latest_reasoning_dict))} chars"
            )

        raw = self.arc_env.step(
            action,
            data=data,
            reasoning=data.get("reasoning", {}),
        )
        if raw is None:
            logger.error(
                "Environment returned no frame for action %s; using last known observation",
                action.name,
            )
            raw = self.arc_env.observation_space if self.arc_env else None
        return self._convert_raw_frame_data(raw)

    def _build_fallback_action(
        self, latest_frame: Optional[FrameData] = None
    ) -> GameAction:
        available = latest_frame.available_actions if latest_frame else []
        action = GameAction.RESET
        if available and action.value not in available:
            for action_id in available:
                try:
                    candidate = GameAction.from_id(action_id)
                except ValueError:
                    logger.warning(
                        "Ignoring unknown action id from available_actions: %s",
                        action_id,
                    )
                    continue
                if candidate != GameAction.ACTION6:
                    action = candidate
                    break
            else:
                try:
                    action = GameAction.from_id(available[0])
                except ValueError:
                    logger.warning(
                        "No known fallback action in available_actions=%s; defaulting to RESET",
                        available,
                    )

        if action == GameAction.ACTION6:
            action.set_data({"game_id": self.game_id, "x": 0, "y": 0})
        else:
            action.set_data({"game_id": self.game_id})
        return action

    def _format_grid(self, frame: FrameData) -> str:
        try:
            if frame.frame and len(frame.frame) > 0:
                # For animated responses, the final layer is the true current state.
                final_layer = frame.frame[-1]
                return "\n".join(
                    [
                        " ".join([str(cell).rjust(2) for cell in row])
                        for row in final_layer
                    ]
                )
            return ""
        except Exception as e:
            logger.error(f"Failed to format grid: {e}")
            return ""

    def _compute_frame_diff(self, frames: list[FrameData]) -> Optional[dict]:
        """Compare previous frame's final state with current frame's final state."""
        if len(frames) < 2:
            return None

        try:
            prev_grid = frames[-2].frame[-1]
            curr_frame = frames[-1]
            curr_grid = curr_frame.frame[-1]
        except (IndexError, TypeError):
            return None

        has_animation = len(curr_frame.frame) > 1

        changes: list[tuple[int, int, int, int]] = []
        for row in range(min(len(prev_grid), len(curr_grid))):
            for col in range(min(len(prev_grid[row]), len(curr_grid[row]))):
                if prev_grid[row][col] != curr_grid[row][col]:
                    changes.append((row, col, prev_grid[row][col], curr_grid[row][col]))

        return {
            "has_diff": len(changes) > 0,
            "has_animation": has_animation,
            "animation_frames": len(curr_frame.frame),
            "changes": changes,
        }

    def _format_sparse_diff(self, changes: list[tuple[int, int, int, int]]) -> str:
        """Format cell changes as a sparse list, grouping consecutive columns with same transition."""
        if not changes:
            return ""

        MAX_DIFF_LINES = 50

        by_row: dict[int, list[tuple[int, int, int]]] = {}
        for row, col, old_val, new_val in changes:
            if row not in by_row:
                by_row[row] = []
            by_row[row].append((col, old_val, new_val))

        lines: list[str] = []
        for row in sorted(by_row.keys()):
            cells = sorted(by_row[row], key=lambda x: x[0])
            # Group consecutive columns with same old->new transition
            groups: list[list[tuple[int, int, int]]] = []
            current_group = [cells[0]]
            for i in range(1, len(cells)):
                col, old_v, new_v = cells[i]
                prev_col, prev_old, prev_new = current_group[-1]
                if col == prev_col + 1 and old_v == prev_old and new_v == prev_new:
                    current_group.append(cells[i])
                else:
                    groups.append(current_group)
                    current_group = [cells[i]]
            groups.append(current_group)

            for group in groups:
                start_col = group[0][0]
                end_col = group[-1][0]
                old_val = group[0][1]
                new_val = group[0][2]
                count = len(group)
                if start_col == end_col:
                    lines.append(f"  row {row}, col {start_col}: {old_val}->{new_val}")
                else:
                    lines.append(
                        f"  row {row}, cols {start_col}-{end_col}: {old_val}->{new_val} ({count} cells)"
                    )

        if len(lines) > MAX_DIFF_LINES:
            truncated = lines[:MAX_DIFF_LINES]
            truncated.append(
                f"  ... and {len(lines) - MAX_DIFF_LINES} more change groups"
            )
            return f"Changed cells ({len(changes)} total):\n" + "\n".join(truncated)

        return f"Changed cells ({len(changes)} total):\n" + "\n".join(lines)

    def _build_previous_action_section(self, frames: list[FrameData]) -> str:
        """Build the section describing the previous action and its effect on the grid."""
        if self._auto_reset_from_game_over:
            self._auto_reset_from_game_over = False
            return (
                "\nThe previous level ended in GAME_OVER. "
                "The game has been automatically reset. Study the new grid carefully."
            )

        if not self.previous_action_info:
            return ""

        action_name = self.previous_action_info.get("name", "unknown")
        action_details = self.previous_action_info.get("details", "")

        diff = self._compute_frame_diff(frames)
        header = f"Your last action: {action_name}{action_details}"

        if diff is None:
            return f"\n{header}\nResult: (first turn — no prior grid state to compare against)"

        if diff["has_diff"]:
            diff_text = self._format_sparse_diff(diff["changes"])
            anim_note = (
                f" (with {diff['animation_frames']}-frame animation)"
                if diff["has_animation"]
                else ""
            )
            return f"\n{header}\nResult{anim_note}: State changed.\n{diff_text}"
        elif diff["has_animation"]:
            return (
                f"\n{header}\n"
                f"Result: Produced {diff['animation_frames']}-frame animation but "
                f"NO change in final grid state. This action did not modify the puzzle."
            )
        else:
            return (
                f"\n{header}\n"
                f"Result: No change. The grid is identical to before this action."
            )

    def _build_action_info(self, action: GameAction) -> dict[str, str]:
        """Build a dict describing an action for the next turn's prompt."""
        name_map = {
            GameAction.RESET: "reset_game",
            GameAction.ACTION1: "action1_move_up",
            GameAction.ACTION2: "action2_move_down",
            GameAction.ACTION3: "action3_move_left",
            GameAction.ACTION4: "action4_move_right",
            GameAction.ACTION5: "action5_interact",
            GameAction.ACTION6: "action6_click",
            GameAction.ACTION7: "action7_undo",
        }
        name = name_map.get(action, f"action_{action.value}")
        details = ""
        if action == GameAction.ACTION6:
            try:
                data = action.action_data
                details = f" (x={data.x}, y={data.y})"
            except Exception:
                pass
        return {"name": name, "details": details}

    def _detect_action_loop(self, window: int = 15) -> Optional[str]:
        """Check if recent actions form a repeating loop.

        Looks at the last `window` actions and checks if they consist entirely
        of a short repeating pattern (1-3 unique actions cycling). Returns a
        warning string if a loop is detected, None otherwise.
        """
        history = self.action_history
        if len(history) < window:
            return None
        recent = history[-window:]
        # Check repeating patterns of length 1, 2, and 3
        for pattern_len in range(1, 4):
            pattern = recent[:pattern_len]
            is_loop = all(
                recent[i] == pattern[i % pattern_len] for i in range(window)
            )
            if is_loop:
                cycle_str = " -> ".join(pattern)
                return (
                    f"WARNING: You are stuck in a loop — you have repeated "
                    f"[{cycle_str}] for the last {window} actions. This is not "
                    f"making progress. Try a completely different action or strategy."
                )
        return None

    def build_game_prompt(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> str:
        grid_str = self._format_grid(latest_frame) or "No grid data available"

        # Build the available tools list dynamically from available_actions
        tool_descriptions = {
            0: "- reset_game: Reset the game to start over",
            1: "- action1_move_up: Execute ACTION1",
            2: "- action2_move_down: Execute ACTION2",
            3: "- action3_move_left: Execute ACTION3",
            4: "- action4_move_right: Execute ACTION4",
            5: "- action5_interact: Execute ACTION5",
            6: "- action6_click: Execute ACTION6 with coordinates (x, y). x: horizontal coordinate (0 = left, 63 = right, range 0-63). y: vertical coordinate (0 = top, 63 = bottom, range 0-63)",
            7: "- action7_undo: Execute ACTION7 (undo)",
        }
        try:
            available_tools_lines = [
                tool_descriptions[a]
                for a in latest_frame.available_actions
                if a in tool_descriptions
            ]
            available_tools_str = (
                "\n".join(available_tools_lines)
                if available_tools_lines
                else "No actions available"
            )
        except Exception as e:
            available_tools_str = "ERROR determining available actions"
            logger.error(f"Failed to format available actions: {e}")

        # Show animation frames from the latest API response only (not cumulative history).
        # Each step() returns a single FrameData whose .frame may contain multiple layers
        # (e.g. 3 frames for an animation). Show evenly-spaced layers, excluding the last
        # layer since it is already displayed as the Current Grid below.
        MAX_ANIMATION_FRAMES = 7
        animation_section = ""
        if latest_frame.frame and len(latest_frame.frame) > 1:
            total_layers = len(latest_frame.frame)
            animation_layers = latest_frame.frame[
                :-1
            ]  # exclude last; it's the Current Grid
            selected = select_animation_frames(animation_layers, MAX_ANIMATION_FRAMES)
            parts = []
            for orig_idx, layer in selected:
                frame_num = orig_idx + 1
                grid = "\n".join(
                    [" ".join([str(cell).rjust(2) for cell in row]) for row in layer]
                )
                if grid:
                    parts.append(
                        f"--- Animation Frame {frame_num} of {total_layers} ---\n{grid}"
                    )
            if parts:
                truncation_note = (
                    f"\n(Showing {len(selected)} evenly-spaced frames of {total_layers} total; final frame is the Current Grid below)\n"
                    if len(selected) < total_layers - 1
                    else ""
                )
                animation_section = (
                    f"\n\nAnimation Frames (from latest action response):{truncation_note}\n"
                    + "\n\n".join(parts)
                    + "\n\n--- End of Animation Frames ---"
                )

        previous_action_section = self._build_previous_action_section(frames)
        loop_warning = self._detect_action_loop() or ""

        prompt = textwrap.dedent(f"""
            You are playing an ARC-AGI-3 game. Your goal is to solve the puzzle.

            Game: {self.game_id}
            Current State: {latest_frame.state.value}
            Levels Completed: {latest_frame.levels_completed}
            {previous_action_section}
            {loop_warning}
            {animation_section}

            Current Grid (64x64, values 0-15):
            {grid_str}

            Note: Some actions trigger animations that return multiple frames. When this
            happens, the animation frames are shown above (excluding the final frame, which
            is the Current Grid). If there are more frames than can be shown, up to
            {MAX_ANIMATION_FRAMES} evenly-spaced frames are displayed. The Current Grid
            always reflects the final state.

            Note: Games may contain progress bars or energy bars visible in the grid. The
            goal of the game is never to simply waste actions to fill or drain these bars.

            Available game action tools (only these are valid this turn):
            {available_tools_str}

            STRUCTURED NOTES: You have a notes file at: {self.notes_path}
            Use the built-in Read, Edit, and Write tools to manage it. The file has pre-built sections.

            Each turn:
            1. Read your notes file to recall your strategy and what you know.
            2. Edit the notes to update with new observations. Use targeted edits to update
               specific sections IN PLACE — don't append to the bottom, don't rewrite the whole file.
            3. Call exactly ONE game action tool to make your move.

            Notes structure rules:
            - "Game Mechanics": General knowledge that applies across ALL levels. When you confirm
              a mechanic through 2+ consistent observations, record it here. Keep it abstract
              (e.g., "9-blocks are buttons that shift content") not level-specific.
            - "Hypothesis": Your single best theory about how the CURRENT level works. REPLACE it
              when you have a better one — never stack multiple contradictory hypotheses. Include
              confidence (LOW/MEDIUM/HIGH) and brief evidence for/against.
            - "Key Positions": Coordinates of important objects. Update in place when things move.
            - "Failed Approaches": What didn't work this level. Check this BEFORE trying an action
              to avoid repeating failed strategies.
            - "Current Plan": Your immediate plan with estimated cost in actions/energy.
              If cost exceeds remaining budget, revise the plan before executing.
            - Keep total notes under 80 lines. Prune ruthlessly — remove disproven hypotheses
              and consolidate verbose entries.

            LEVEL TRANSITIONS: When levels_completed increases (you solved a level!):
            1. Promote any newly confirmed mechanics to "Game Mechanics".
            2. Clear the level-specific sections (Hypothesis, Key Positions, Failed Approaches, Plan).
            3. Spend your first 2-3 actions on the new level OBSERVING — study the grid layout
               and identify key objects before taking actions. Apply your Game Mechanics knowledge
               but don't assume the level works identically to the previous one.

            CRITICAL RESET RULES (violation will force quit the game):
            - Do NOT restart the game.
            - Do NOT restart the level at the beginning of a level.
            - Do NOT restart the level twice in a row.

            Before calling a game action tool, explain your reasoning.
        """).strip()

        strategy_prompt = os.getenv("STRATEGY_PROMPT", "").strip()
        if strategy_prompt:
            prompt += f"\n\n## Strategy Prompt\n{strategy_prompt}"

        return prompt

    def _ensure_client_connected(self) -> None:
        """Connect the ClaudeSDKClient if not already connected.

        Uses a single persistent subprocess across all game turns.
        """
        if self._client_connected:
            return

        async def _connect():
            # Clean up stale client from a previous failed connection
            if self._client:
                try:
                    await self._client.disconnect()
                except Exception:
                    pass

            options = ClaudeAgentOptions(
                model=self.MODEL,
                mcp_servers={"arc-game-tools": self.mcp_server},
                permission_mode="bypassPermissions",
                cwd=self.notes_dir,
                tools=["Read", "Edit", "Write"],
                system_prompt={
                    "type": "preset",
                    "preset": "claude_code",
                    "append": textwrap.dedent("""
                        IMPORTANT - INTERRUPT BEHAVIOR: After you call a game action tool,
                        the system will interrupt you to process the action and advance the game.
                        This is expected and normal — do not try to prevent or work around interrupts.
                        Each turn, you should:
                        1. Optionally read/update your notes file for strategy tracking
                        2. Analyze the current game state
                        3. Call exactly ONE game action tool
                        You will then be interrupted, and the next game state will be provided.
                    """).strip(),
                },
            )
            self._client = ClaudeSDKClient(options=options)
            await self._client.connect()
            self._client_connected = True
            logger.info("ClaudeSDKClient connected (persistent subprocess)")

        self._loop.run_until_complete(_connect())

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        self.step_counter += 1
        logger.info(f"Step {self.step_counter}: Choosing action...")

        if self.consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
            logger.error(
                f"FATAL: {self.consecutive_errors} consecutive errors, stopping agent"
            )
            raise RuntimeError(
                f"Too many consecutive errors ({self.consecutive_errors}), cannot continue"
            )

        # Auto-reset on GAME_OVER without querying Claude (mirrors random agent behavior)
        if latest_frame.state is GameState.GAME_OVER:
            logger.info(
                f"GAME_OVER detected — auto-resetting (step {self.step_counter})"
            )
            action = GameAction.RESET
            action.reasoning = "Auto-reset: game over state detected"
            self.previous_action_info = self._build_action_info(action)
            self.action_history.append(
                self.previous_action_info["name"]
                + self.previous_action_info.get("details", "")
            )
            self._auto_reset_from_game_over = True
            return action

        self.current_frame = latest_frame
        self.latest_reasoning = ""
        self.latest_reasoning_dict = {}
        action_taken: Optional[GameAction] = None
        self.captured_messages = []
        self.current_prompt = self.build_game_prompt(frames, latest_frame)
        self.result_message = None

        # Ensure the persistent client is connected (single subprocess for all turns)
        self._ensure_client_connected()

        async def _run_turn():
            nonlocal action_taken
            reasoning_parts: list[str] = []
            pending_action_tool_use_id: Optional[str] = None
            pending_action_tool_name: Optional[str] = None
            pending_action_tool_input: dict[str, Any] = {}

            try:
                # Send prompt to the existing persistent session
                await self._client.query(self.current_prompt)

                # receive_response() yields messages and auto-terminates after ResultMessage
                async for message in self._client.receive_response():
                    self.captured_messages.append(message)

                    if isinstance(message, SystemMessage) and message.subtype == "init":
                        if not self.session_id:
                            self.session_id = message.data.get("session_id")
                            logger.info(f"Session started: {self.session_id}")
                        else:
                            resumed_session = message.data.get("session_id")
                            if resumed_session != self.session_id:
                                logger.warning(
                                    f"Session ID mismatch: expected {self.session_id}, got {resumed_session}"
                                )

                    if isinstance(message, ResultMessage):
                        self.result_message = message
                        if message.is_error:
                            logger.error(
                                "ResultMessage indicates error occurred during query: %s",
                                getattr(message, "result", "no result payload"),
                            )

                    if isinstance(message, AssistantMessage) and not action_taken:
                        for block in message.content:
                            if hasattr(block, "text") and block.text:
                                reasoning_parts.append(block.text)
                                logger.info(f"Claude reasoning: {block.text[:100]}...")

                                if "credit balance is too low" in block.text.lower():
                                    logger.error(
                                        "FATAL: Credit balance too low - stopping immediately"
                                    )
                                    print("\n" + "=" * 80)
                                    print(
                                        "\033[91m"
                                        + "ERROR: Insufficient Anthropic API Credits"
                                        + "\033[0m"
                                    )
                                    print(
                                        "Please add credits to your Anthropic account to continue."
                                    )
                                    print("=" * 80 + "\n")
                                    os._exit(1)

                            if isinstance(block, ToolUseBlock):
                                tool_name = block.name
                                logger.info(f"Claude calling tool: {tool_name}")

                                if reasoning_parts:
                                    self.latest_reasoning = " ".join(reasoning_parts)

                                if tool_name in self.ACTION_TOOL_MAP:
                                    if (
                                        pending_action_tool_use_id
                                        and pending_action_tool_use_id != block.id
                                    ):
                                        logger.warning(
                                            "Received additional game action tool call while another is pending; "
                                            "tracking latest one only (old=%s, new=%s)",
                                            pending_action_tool_use_id,
                                            block.id,
                                        )
                                    pending_action_tool_use_id = block.id
                                    pending_action_tool_name = tool_name
                                    pending_action_tool_input = (
                                        block.input
                                        if isinstance(block.input, dict)
                                        else {}
                                    )
                                    logger.info(
                                        "Queued action tool call %s (tool_use_id=%s); waiting for tool_result",
                                        tool_name,
                                        block.id,
                                    )
                                else:
                                    logger.debug(f"Non-action tool called: {tool_name}")

                    if (
                        isinstance(message, UserMessage)
                        and pending_action_tool_use_id
                        and not action_taken
                    ):
                        for block in message.content:
                            if not isinstance(block, ToolResultBlock):
                                continue
                            if block.tool_use_id != pending_action_tool_use_id:
                                continue

                            if bool(block.is_error):
                                logger.warning(
                                    "Ignoring action tool %s (tool_use_id=%s): tool_result returned is_error=True",
                                    pending_action_tool_name,
                                    pending_action_tool_use_id,
                                )
                                pending_action_tool_use_id = None
                                pending_action_tool_name = None
                                pending_action_tool_input = {}
                                break

                            action_taken = self.parse_action_from_tool(
                                pending_action_tool_name or "",
                                pending_action_tool_input,
                            )
                            pending_action_tool_use_id = None
                            pending_action_tool_name = None
                            pending_action_tool_input = {}

                            if action_taken:
                                if (
                                    latest_frame.available_actions
                                    and action_taken.value
                                    not in latest_frame.available_actions
                                ):
                                    logger.warning(
                                        f"Action {action_taken.name} (value={action_taken.value}) "
                                        f"not in available_actions: {latest_frame.available_actions}"
                                    )
                                logger.info(
                                    "Validated action from successful tool_result: %s; sending interrupt",
                                    action_taken.name,
                                )
                                try:
                                    await self._client.interrupt()
                                    logger.debug("Interrupt sent successfully")
                                except Exception as e:
                                    logger.debug(
                                        f"Interrupt error (may be expected): {e}"
                                    )
                            else:
                                logger.warning(
                                    "Tool result succeeded but no valid action could be parsed"
                                )
                            break
            except Exception as e:
                if "credit balance" in str(e).lower():
                    raise
                logger.error(f"Error during query execution: {e}")
                logger.debug(traceback.format_exc())
                # Mark disconnected so next turn reconnects
                self._client_connected = False

        try:
            self._loop.run_until_complete(_run_turn())
        except RuntimeError as e:
            if "credit balance" in str(e).lower():
                print("\n" + "=" * 80)
                print(
                    "\033[91m" + "ERROR: Insufficient Anthropic API Credits" + "\033[0m"
                )
                print("Please add credits to your Anthropic account to continue.")
                print("=" * 80 + "\n")
                os._exit(1)
            raise
        except Exception as e:
            if "credit balance" in str(e).lower():
                print("\n" + "=" * 80)
                print(
                    "\033[91m" + "ERROR: Insufficient Anthropic API Credits" + "\033[0m"
                )
                print("Please add credits to your Anthropic account to continue.")
                print("=" * 80 + "\n")
                os._exit(1)
            logger.error(f"Error running event loop: {e}")
            logger.debug(traceback.format_exc())

        if action_taken:
            self.consecutive_errors = 0
            if not self.latest_reasoning:
                logger.warning("Action taken but no reasoning captured")

            if self.claude_recorder and not self.is_playback:
                parsed_action = {
                    "action": action_taken.value,
                    "reasoning": self.latest_reasoning,
                }

                cost_usd = 0.0
                if self.result_message and hasattr(
                    self.result_message, "total_cost_usd"
                ):
                    cost_usd = self.result_message.total_cost_usd or 0.0
                    logger.debug(f"Cost from API: ${cost_usd:.6f}")
                else:
                    logger.warning("No total_cost_usd in ResultMessage")

                self.cumulative_cost_usd += cost_usd

                if self.result_message:
                    try:
                        self.track_tokens_from_result(self.result_message)
                    except Exception as e:
                        logger.error(f"Failed to track tokens: {e}")

                try:
                    self.claude_recorder.save_step(
                        step=self.step_counter,
                        prompt=self.current_prompt,
                        messages=self.captured_messages,
                        parsed_action=parsed_action,
                        total_cost_usd=cost_usd,
                    )
                except Exception as e:
                    logger.error(f"Failed to save step recording: {e}")
                    import traceback

                    logger.debug(traceback.format_exc())

            self.previous_action_info = self._build_action_info(action_taken)
            self.action_history.append(
                self.previous_action_info["name"]
                + self.previous_action_info.get("details", "")
            )
            return action_taken

        self.consecutive_errors += 1
        if self.result_message and getattr(self.result_message, "is_error", False):
            logger.warning(
                "Last ResultMessage had is_error=True; forcing Claude client reconnect on next turn"
            )
            self._client_connected = False
        logger.warning(
            f"No action was taken by Claude (consecutive errors: {self.consecutive_errors}/{self.MAX_CONSECUTIVE_ERRORS}), defaulting to RESET"
        )
        if not self.captured_messages:
            logger.error(
                "No messages captured at all - query may have failed completely"
            )
            if self.session_id:
                logger.error(f"Session may be corrupted: {self.session_id}")
        else:
            logger.warning(
                f"Captured {len(self.captured_messages)} messages but no valid action found"
            )
        fallback = self._build_fallback_action(latest_frame)
        self.previous_action_info = self._build_action_info(fallback)
        self.action_history.append(
            self.previous_action_info["name"]
            + self.previous_action_info.get("details", "")
        )
        return fallback

    def cleanup(self, scorecard=None):
        """Clean up the persistent client and event loop."""
        if self._client_connected and self._client:
            try:
                self._loop.run_until_complete(self._client.disconnect())
                logger.info("ClaudeSDKClient disconnected")
            except Exception as e:
                logger.warning(f"Error disconnecting client: {e}")
            self._client = None
            self._client_connected = False

        try:
            pending = asyncio.all_tasks(self._loop)
            if pending:
                for task in pending:
                    task.cancel()
                self._loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            self._loop.close()
        except Exception as e:
            logger.warning(f"Error closing event loop: {e}")

        super().cleanup(scorecard)

    def parse_action_from_tool(
        self, tool_name: str, tool_input: dict[str, Any]
    ) -> Optional[GameAction]:
        if tool_name in self.ACTION_TOOL_MAP:
            action = self.ACTION_TOOL_MAP[tool_name]

            if self.current_frame and self.current_frame.available_actions:
                if action.value not in self.current_frame.available_actions:
                    logger.warning(
                        f"Rejecting {action.name} (value={action.value}): not in available_actions {self.current_frame.available_actions}"
                    )
                    return None

            if action == GameAction.ACTION6:
                x = tool_input.get("x", 0)
                y = tool_input.get("y", 0)

                if not (0 <= x <= 63 and 0 <= y <= 63):
                    logger.warning(f"ACTION6 coordinates out of range: x={x}, y={y}")
                    return None

                action.set_data({"game_id": self.game_id, "x": x, "y": y})
            else:
                action.set_data({"game_id": self.game_id})

            if self.latest_reasoning:
                action_label = tool_name.replace("mcp__arc-game-tools__", "")
                if action == GameAction.ACTION6:
                    action_label = f"{action_label} (x={tool_input.get('x', 0)}, y={tool_input.get('y', 0)})"
                thought_text = f"{action_label}\n\n{self.latest_reasoning}"
                self.latest_reasoning_dict = {"thought": thought_text[:16000]}
                logger.info(
                    f"Prepared reasoning for action ({len(thought_text)} chars)"
                )
            else:
                self.latest_reasoning_dict = {}
                logger.warning(
                    "No reasoning captured for action - reasoning logs will not appear in replay"
                )

            return action

        logger.debug(f"Non-action tool called: {tool_name}")
        return None

    def track_tokens_from_result(self, result_message: Any) -> None:
        if not hasattr(result_message, "usage") or not result_message.usage:
            logger.debug("No usage data in ResultMessage")
            return

        try:
            usage = result_message.usage
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            cached_tokens = usage.get("cache_read_input_tokens", 0)
            total = input_tokens + output_tokens
            self.token_counter += total

            logger.info(
                f"Token usage: +{total} (in: {input_tokens}, out: {output_tokens}, cached: {cached_tokens}), total: {self.token_counter}"
            )
        except Exception as e:
            logger.error(f"Error tracking tokens: {e}")

        if hasattr(self, "recorder") and not self.is_playback:
            self.recorder.record(
                {
                    "tokens": total,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": self.token_counter,
                }
            )
