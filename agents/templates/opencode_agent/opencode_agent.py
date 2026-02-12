import logging
import os
import subprocess
import textwrap
import time
import uuid
from typing import Any, Optional

from arcengine import FrameData, GameAction, GameState

from ...agent import Agent
from .opencode_client import OpenCodeClient, MessageParser
from .opencode_recorder import OpenCodeRecorder

logger = logging.getLogger()


class OpenCodeAgent(Agent):
    MAX_ACTIONS: int = int(os.getenv("STEP_COUNT", 5000))
    MODEL: str = "openai/gpt-5.2"
    MAX_CONSECUTIVE_ERRORS: int = 3
    MAX_SERVER_RESTARTS: int = 2
    ACTION_TOOL_MAP: dict[str, GameAction] = {
        "reset_game": GameAction.RESET,
        "action1_move_up": GameAction.ACTION1,
        "action2_move_down": GameAction.ACTION2,
        "action3_move_left": GameAction.ACTION3,
        "action4_move_right": GameAction.ACTION4,
        "action5_interact": GameAction.ACTION5,
        "action6_click": GameAction.ACTION6,
        "action7_undo": GameAction.ACTION7,
    }
    
    token_counter: int
    step_counter: int
    latest_reasoning: str
    latest_reasoning_dict: dict[str, Any]
    opencode_recorder: Optional[OpenCodeRecorder]
    opencode_client: OpenCodeClient
    session_id: Optional[str]
    consecutive_errors: int
    previous_action_info: Optional[dict[str, str]]
    current_frame: Optional[FrameData]
    cumulative_cost_usd: float
    notes_session_id: str
    notes_dir: str
    notes_path: str
    opencode_server_process: Optional[subprocess.Popen]
    opencode_server_port: int
    server_restart_count: int
    
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
        self.previous_action_info = None
        self.server_restart_count = 0
        
        self.notes_session_id = str(uuid.uuid4())
        self.notes_dir = os.path.abspath(f"./game_notes/{self.game_id}_{self.notes_session_id}")
        try:
            os.makedirs(self.notes_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create notes directory {self.notes_dir}: {e}")
            raise
        self.notes_path = os.path.join(self.notes_dir, "notes.md")
        with open(self.notes_path, 'w') as f:
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
        
        if kwargs.get("record", False):
            self.opencode_recorder = OpenCodeRecorder(
                game_id=kwargs.get("game_id", "unknown"),
                agent_name=self.agent_name,
                session_id=self.notes_session_id
            )
        else:
            self.opencode_recorder = None
        
        self.opencode_server_port = int(os.getenv("OPENCODE_SERVER_PORT", 4096))
        self.opencode_server_process = None
        
        self._start_opencode_server()
        
        base_url = f"http://localhost:{self.opencode_server_port}"
        self.opencode_client = OpenCodeClient(base_url=base_url)
        
        try:
            mcp_status = self.opencode_client.get_mcp_status()
            logger.info(f"MCP servers status: {mcp_status}")
            
            arc_tools_status = mcp_status.get("arc-game-tools", {}).get("status")
            if arc_tools_status != "connected":
                logger.error(f"ARC game tools MCP server not connected: {arc_tools_status}")
                logger.error("Agent will not be able to make game moves")
            
        except Exception as e:
            logger.warning(f"Could not get MCP status: {e}")
            logger.warning("Unable to verify MCP tools are loaded - agent may fail to take actions")
        
        self._auto_reset_from_game_over = False
        
        logger.info(f"OpenCodeAgent initialized for game {self.game_id}")
    
    def _start_opencode_server(self) -> None:
        logger.info(f"Starting OpenCode server on port {self.opencode_server_port}...")
        
        opencode_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(opencode_dir, "opencode.json")
        
        if not os.path.exists(config_file):
            logger.error(f"OpenCode config file not found: {config_file}")
            raise FileNotFoundError(f"Missing opencode.json")
        
        env = os.environ.copy()
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        if not openrouter_key:
            logger.error("OPENROUTER_API_KEY not set in environment - LLM calls will fail")
        env["OPENROUTER_API_KEY"] = openrouter_key
        
        try:
            self.opencode_server_process = subprocess.Popen(
                ["opencode", "serve", "--port", str(self.opencode_server_port), "--hostname", "127.0.0.1"],
                cwd=opencode_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            logger.info(f"OpenCode server started (PID: {self.opencode_server_process.pid})")
            
            max_wait = 15
            for i in range(max_wait):
                poll_result = self.opencode_server_process.poll()
                if poll_result is not None:
                    stdout, stderr = self.opencode_server_process.communicate(timeout=1)
                    logger.error(f"OpenCode server crashed with exit code {poll_result}")
                    logger.error(f"STDOUT: {stdout}")
                    logger.error(f"STDERR: {stderr}")
                    raise RuntimeError(f"OpenCode server failed to start (exit code {poll_result})")
                
                try:
                    import httpx
                    response = httpx.get(f"http://localhost:{self.opencode_server_port}/global/health", timeout=1.0)
                    if response.status_code == 200:
                        health = response.json()
                        logger.info(f"OpenCode server ready (version: {health.get('version')})")
                        self._configure_openrouter_auth()
                        return
                except Exception as e:
                    if i == 0 or i == max_wait - 1:
                        logger.debug(f"Health check attempt {i+1}/{max_wait}: {e}")
                
                if i < max_wait - 1:
                    time.sleep(1)
            
            stdout, stderr = self.opencode_server_process.communicate(timeout=1)
            logger.error(f"OpenCode server timeout. STDOUT: {stdout[:500]}")
            logger.error(f"OpenCode server timeout. STDERR: {stderr[:500]}")
            raise RuntimeError("OpenCode server failed to become ready")
            
        except FileNotFoundError:
            logger.error("OpenCode CLI not found. Please install: npm install -g @opencode-ai/cli")
            raise RuntimeError("OpenCode CLI not installed")
        except Exception as e:
            logger.error(f"Failed to start OpenCode server: {e}")
            raise
    
    def _configure_openrouter_auth(self) -> None:
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        if not openrouter_key:
            logger.warning("OPENROUTER_API_KEY not set - model calls may fail")
            return
        
        try:
            import httpx
            response = httpx.put(
                f"http://localhost:{self.opencode_server_port}/auth/openrouter",
                json={
                    "type": "api",
                    "key": openrouter_key
                },
                timeout=5.0
            )
            if response.status_code == 200:
                logger.info("OpenRouter authentication configured")
                result = response.json()
                logger.debug(f"Auth response: {result}")
            else:
                logger.warning(f"OpenRouter auth returned status {response.status_code}: {response.text}")
        except Exception as e:
            logger.warning(f"Failed to configure OpenRouter auth: {e}")
    
    def _get_server_logs(self, max_lines: int = 50) -> str:
        if not self.opencode_server_process:
            return "No server process"
        
        try:
            if self.opencode_server_process.poll() is not None:
                try:
                    stdout, stderr = self.opencode_server_process.communicate(timeout=0.5)
                    if stderr:
                        lines = stderr.strip().split('\n')
                        return '\n'.join(lines[-max_lines:]) if lines else "No stderr output"
                except Exception:
                    pass
            
            if not self.opencode_server_process.stderr:
                return "No stderr available"
            
            logs = []
            while True:
                line = self.opencode_server_process.stderr.readline()
                if not line:
                    break
                logs.append(line.strip())
                if len(logs) >= max_lines:
                    break
            return "\n".join(logs) if logs else "No new logs"
        except Exception as e:
            logger.debug(f"Error reading server logs: {e}")
            return f"Error reading logs: {e}"
    
    def _stop_opencode_server(self) -> None:
        if self.opencode_server_process:
            pid = self.opencode_server_process.pid
            logger.info(f"Stopping OpenCode server (PID: {pid})...")
            try:
                self.opencode_server_process.terminate()
                self.opencode_server_process.wait(timeout=5)
                logger.info("OpenCode server stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("OpenCode server didn't stop gracefully, killing...")
                self.opencode_server_process.kill()
                try:
                    self.opencode_server_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    logger.error(f"OpenCode server (PID {pid}) won't die, force killing process tree...")
                    import signal
                    try:
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                    except Exception as kill_err:
                        logger.warning(f"Could not kill process group: {kill_err}")
            except Exception as e:
                logger.warning(f"Error stopping OpenCode server: {e}")
            finally:
                self.opencode_server_process = None
                
                time.sleep(1)
                
                try:
                    result = subprocess.run(
                        ["lsof", "-ti", f":{self.opencode_server_port}"],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    if result.stdout.strip():
                        pids = result.stdout.strip().split('\n')
                        for pid_str in pids:
                            try:
                                zombie_pid = int(pid_str)
                                logger.warning(f"Killing zombie process {zombie_pid} on port {self.opencode_server_port}")
                                os.kill(zombie_pid, 9)
                            except Exception as e:
                                logger.debug(f"Could not kill PID {pid_str}: {e}")
                        time.sleep(0.5)
                except Exception as e:
                    logger.debug(f"Port cleanup check failed: {e}")
    
    @property
    def name(self) -> str:
        sanitized_model_name = self.MODEL.replace("/", "-").replace(":", "-")
        return f"{super().name}.{sanitized_model_name}"
    
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return any([
            latest_frame.state is GameState.WIN,
        ])
    
    def do_action_request(self, action: GameAction) -> FrameData:
        data = action.action_data.model_dump()
        
        if self.latest_reasoning_dict:
            data["reasoning"] = self.latest_reasoning_dict
            logger.info(f"Added reasoning to action request: {len(str(self.latest_reasoning_dict))} chars")
        
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
    
    def _build_fallback_action(self, latest_frame: Optional[FrameData] = None) -> GameAction:
        available = latest_frame.available_actions if latest_frame else []
        action = GameAction.RESET
        if available and action.value not in available:
            for action_id in available:
                try:
                    candidate = GameAction.from_id(action_id)
                except ValueError:
                    logger.warning("Ignoring unknown action id from available_actions: %s", action_id)
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
                final_layer = frame.frame[-1]
                return "\n".join(
                    [" ".join([str(cell).rjust(2) for cell in row]) for row in final_layer]
                )
            return ""
        except Exception as e:
            logger.error(f"Failed to format grid: {e}")
            return ""
    
    def _compute_frame_diff(self, frames: list[FrameData]) -> Optional[dict]:
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
            'has_diff': len(changes) > 0,
            'has_animation': has_animation,
            'animation_frames': len(curr_frame.frame),
            'changes': changes,
        }
    
    def _format_sparse_diff(self, changes: list[tuple[int, int, int, int]]) -> str:
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
                    lines.append(f"  row {row}, cols {start_col}-{end_col}: {old_val}->{new_val} ({count} cells)")
        
        if len(lines) > MAX_DIFF_LINES:
            truncated = lines[:MAX_DIFF_LINES]
            truncated.append(f"  ... and {len(lines) - MAX_DIFF_LINES} more change groups")
            return f"Changed cells ({len(changes)} total):\n" + "\n".join(truncated)
        
        return f"Changed cells ({len(changes)} total):\n" + "\n".join(lines)
    
    def _build_previous_action_section(self, frames: list[FrameData]) -> str:
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
        
        if diff['has_diff']:
            diff_text = self._format_sparse_diff(diff['changes'])
            anim_note = f" (with {diff['animation_frames']}-frame animation)" if diff['has_animation'] else ""
            return f"\n{header}\nResult{anim_note}: State changed.\n{diff_text}"
        elif diff['has_animation']:
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
    
    def build_game_prompt(self, frames: list[FrameData], latest_frame: FrameData) -> str:
        grid_str = self._format_grid(latest_frame) or "No grid data available"
        
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
            available_tools_str = "\n".join(available_tools_lines) if available_tools_lines else "No actions available"
        except Exception as e:
            available_tools_str = "ERROR determining available actions"
            logger.error(f"Failed to format available actions: {e}")
        
        MAX_ANIMATION_FRAMES = 7
        animation_section = ""
        if latest_frame.frame and len(latest_frame.frame) > 1:
            total_layers = len(latest_frame.frame)
            layers_to_show = latest_frame.frame[-MAX_ANIMATION_FRAMES:]
            skipped = total_layers - len(layers_to_show)
            parts = []
            for idx, layer in enumerate(layers_to_show):
                frame_num = skipped + idx + 1
                grid = "\n".join(
                    [" ".join([str(cell).rjust(2) for cell in row]) for row in layer]
                )
                if grid:
                    parts.append(f"--- Animation Frame {frame_num} of {total_layers} ---\n{grid}")
            if parts:
                truncation_note = f"\n(Showing last {len(layers_to_show)} of {total_layers} frames)\n" if skipped > 0 else ""
                animation_section = (
                    f"\n\nAnimation Frames (from latest action response):{truncation_note}\n"
                    + "\n\n".join(parts)
                    + "\n\n--- End of Animation Frames ---"
                )
        
        previous_action_section = self._build_previous_action_section(frames)
        
        prompt = textwrap.dedent(f"""
            You are playing an ARC-AGI-3 game. Your goal is to solve the puzzle.
            
            Game: {self.game_id}
            Current State: {latest_frame.state.value}
            Levels Completed: {latest_frame.levels_completed}
            {previous_action_section}
            {animation_section}
            
            Current Grid (64x64, values 0-15):
            {grid_str}
            
            Note: Some actions trigger animations that return multiple frames. When this
            happens, the animation frames are shown above. If an animation has more than
            {MAX_ANIMATION_FRAMES} frames, only the last {MAX_ANIMATION_FRAMES} are shown.
            The Current Grid always reflects the final state.
            
            **GAME ACTION TOOLS**: Use the arc-game-tools MCP server to take game actions.
            Available tools this turn: {available_tools_str}
            
            CRITICAL: You can only call ONE game action per turn. Calling multiple actions will corrupt the game state.
            Choose the single best action, then STOP. Do not call multiple action tools in sequence.
            
            To use a game action tool, explicitly reference the MCP server:
            - use the arc-game-tools reset_game tool (when needed)
            - use the arc-game-tools action1_move tool (move character)
            - use the arc-game-tools action2_interact tool (interact with objects)
            - use the arc-game-tools action3_... (and so on for other actions)
            
            STRUCTURED NOTES: You have a notes file at: {self.notes_path}
            Use the built-in Read, Edit, and Write tools to manage it. The file has pre-built sections.
            {'NOTE: This is step 1 - the notes file is fresh/empty. Skip reading it and start analyzing the grid.' if self.step_counter == 1 else ''}
            
            Each turn:
            {'1. Analyze the grid and update your notes with initial observations (skip reading notes on step 1).' if self.step_counter == 1 else '1. Read your notes file to recall your strategy and what you know.'}
            2. Edit the notes to update with new observations. Use targeted edits to update
               specific sections IN PLACE — don't append to the bottom, don't rewrite the whole file.
            3. Think carefully and choose the SINGLE BEST game action for this turn.
            4. Call EXACTLY ONE game action tool from arc-game-tools, then STOP immediately.
               Example: "use the arc-game-tools reset_game tool" or "use the arc-game-tools action2_interact tool"
            
            STOP AFTER CALLING ONE ACTION TOOL. Do not call a second action tool. One action per turn only.
            
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
    
    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        self.step_counter += 1
        logger.info(f"Step {self.step_counter}: Choosing action...")
        
        if self.consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
            logger.error(f"FATAL: {self.consecutive_errors} consecutive errors, stopping agent")
            logger.error(f"Last session ID: {self.session_id}")
            logger.error(f"Game state: {self.current_frame.state if self.current_frame else 'NO FRAME'}")
            raise RuntimeError(f"Too many consecutive errors ({self.consecutive_errors}), cannot continue")
        
        if latest_frame.state is GameState.GAME_OVER:
            logger.info(f"GAME_OVER detected — auto-resetting (step {self.step_counter})")
            action = GameAction.RESET
            action.reasoning = "Auto-reset: game over state detected"
            self.previous_action_info = self._build_action_info(action)
            self._auto_reset_from_game_over = True
            return action
        
        if not latest_frame:
            logger.warning("choose_action called with no latest_frame - game state may be inconsistent")
        elif not latest_frame.frame or len(latest_frame.frame) == 0:
            logger.warning(f"latest_frame has no frame data - game state: {latest_frame.state}")
        
        self.current_frame = latest_frame
        self.latest_reasoning = ""
        self.latest_reasoning_dict = {}
        
        prompt = self.build_game_prompt(frames, latest_frame)
        
        if self.opencode_server_process and self.opencode_server_process.poll() is not None:
            exit_code = self.opencode_server_process.poll()
            logger.error(f"OpenCode server has died (exit code: {exit_code})")
            
            server_logs = self._get_server_logs(max_lines=50)
            if server_logs and server_logs != "No new logs":
                logger.error(f"OpenCode server logs before crash:\n{server_logs}")
            
            if self.server_restart_count >= self.MAX_SERVER_RESTARTS:
                logger.error(f"OpenCode server has crashed {self.server_restart_count} times, giving up")
                raise RuntimeError(f"OpenCode server crashed too many times ({self.server_restart_count})")
            
            self.server_restart_count += 1
            logger.warning(f"Attempting to restart OpenCode server (attempt {self.server_restart_count}/{self.MAX_SERVER_RESTARTS})...")
            try:
                self._stop_opencode_server()
                
                logger.info("Waiting 2 seconds for port to be fully released...")
                time.sleep(2)
                
                self._start_opencode_server()
                
                if self.session_id:
                    logger.warning(f"Session {self.session_id} was lost, creating new session")
                    self.session_id = None
                
                logger.info("OpenCode server restarted successfully")
            except Exception as e:
                logger.error(f"Failed to restart OpenCode server: {e}")
                raise RuntimeError("OpenCode server process died and could not be restarted")
        
        try:
            if not self.session_id:
                logger.info(f"Creating new OpenCode session (step {self.step_counter})...")
                if self.step_counter > 1:
                    logger.warning(f"Creating session at step {self.step_counter} - may indicate session was lost or reset")
                try:
                    session = self.opencode_client.create_session(
                        title=f"ARC Game: {self.game_id}"
                    )
                    if not session or not session.get("id"):
                        logger.error(f"Session creation returned invalid response: {session}")
                        raise ValueError("Invalid session response")
                    self.session_id = session.get("id")
                    logger.info(f"Session created: {self.session_id}")
                except Exception as e:
                    logger.error(f"Failed to create OpenCode session: {e}")
                    self.consecutive_errors += 1
                    raise
            else:
                logger.debug(f"Reusing existing session: {self.session_id}")
            
            logger.info(f"Sending async prompt to OpenCode (length: {len(prompt)})")
            
            self.opencode_client.send_message_async(
                session_id=self.session_id,
                prompt=prompt,
                tools=tools_config,
                agent="build",
                model={
                    "providerID": "openrouter",
                    "modelID": self.MODEL
                }
            )
            
            game_action_detected = False
            first_tool_call_time = None
            poll_interval = 0.3
            max_polls = 120
            poll_count = 0
            has_any_assistant_messages = False
            has_step_finish = False
            
            logger.info("Polling for tool calls (will abort after first game action)...")
            
            while poll_count < max_polls:
                time.sleep(poll_interval)
                poll_count += 1
                
                try:
                    status_response = self.opencode_client.get_session_status()
                    session_status_obj = status_response.get(self.session_id, {})
                    
                    if isinstance(session_status_obj, dict):
                        session_type = session_status_obj.get("type", "unknown")
                    else:
                        session_type = str(session_status_obj)
                    
                    messages = self.opencode_client.get_messages(self.session_id, limit=10)
                    
                    if not messages:
                        logger.debug(f"Poll {poll_count}: No messages yet, waiting...")
                        continue
                    
                    for msg in messages:
                        if msg.get("info", {}).get("role") != "assistant":
                            continue
                        
                        has_any_assistant_messages = True
                        parts = msg.get("parts", [])
                        
                        for part in parts:
                            if part.get("type") == "step-finish":
                                has_step_finish = True
                            
                            if not self.is_complete_tool_call(part):
                                continue
                            
                            tool_info = part.get("tool")
                            tool_name = ""
                            if isinstance(tool_info, dict):
                                tool_name = tool_info.get("name", "")
                            elif isinstance(tool_info, str):
                                tool_name = tool_info
                            
                            if not tool_name:
                                continue
                            
                            clean_name = tool_name.replace("arc-game-tools_", "")
                            if clean_name in self.ACTION_TOOL_MAP:
                                if not game_action_detected:
                                    first_tool_call_time = time.time()
                                    game_action_detected = True
                                    logger.info(f"🎯 First game action detected at poll {poll_count}: {tool_name}")
                                    logger.info(f"Session status: {session_type}, has_step_finish: {has_step_finish}")
                                    
                                    if not has_step_finish and session_type in ["busy", "running"]:
                                        try:
                                            logger.info("⚡ Aborting session to prevent additional tool calls...")
                                            abort_success = self.opencode_client.abort_session(self.session_id)
                                            if abort_success:
                                                logger.info("✅ Successfully aborted session after first tool call")
                                            else:
                                                logger.warning("⚠️  Abort returned false")
                                            time.sleep(0.3)
                                        except Exception as abort_err:
                                            logger.warning(f"Failed to abort: {abort_err}")
                                    else:
                                        logger.info(f"Session already finished or has step-finish, skipping abort")
                                    break
                        
                        if game_action_detected:
                            break
                    
                    if game_action_detected:
                        break
                    
                    if has_any_assistant_messages and has_step_finish:
                        logger.info(f"Session finished naturally at poll {poll_count} (step-finish detected)")
                        break
                
                except Exception as e:
                    logger.warning(f"Error during polling at iteration {poll_count}: {e}")
                    break
            
            if not game_action_detected and poll_count >= max_polls:
                logger.warning(f"Polling timeout after {max_polls} iterations (~{max_polls * poll_interval}s)")
            
            messages = self.opencode_client.get_messages(self.session_id, limit=10)
            
            if messages:
                logger.info(f"Retrieved {len(messages)} messages after polling")
                for i, msg in enumerate(messages):
                    role = msg.get("info", {}).get("role", "unknown")
                    parts = msg.get("parts", [])
                    part_types = [p.get("type") for p in parts]
                    logger.info(f"  Message {i}: role={role}, parts={part_types}")
                    
                    for part in parts:
                        if part.get("type") == "text":
                            text = part.get("text", "")
                            if text:
                                logger.info(f"    LLM text: {text[:200]}{'...' if len(text) > 200 else ''}")
                        elif part.get("type") == "tool":
                            tool_info = part.get("tool")
                            if isinstance(tool_info, dict):
                                tool_name = tool_info.get("name", "unknown")
                            elif isinstance(tool_info, str):
                                tool_name = tool_info
                            else:
                                tool_name = str(tool_info)
                            logger.info(f"    Tool call: {tool_name}")
            
            if first_tool_call_time:
                elapsed = time.time() - first_tool_call_time
                logger.info(f"⏱️  Time from first tool call detection to message retrieval: {elapsed:.2f}s")
            
            if not messages:
                logger.warning("Received empty message list from OpenCode - possible session issue")
            
            parsed = MessageParser.parse_messages(messages)
            
            if parsed["reasoning"]:
                self.latest_reasoning = " ".join(parsed["reasoning"])
                logger.info(f"Agent reasoning: {self.latest_reasoning[:200]}{'...' if len(self.latest_reasoning) > 200 else ''}")
            
            step_cost = parsed.get("total_cost", 0.0)
            if step_cost > 0:
                self.cumulative_cost_usd += step_cost
                logger.info(f"Step cost: ${step_cost:.6f}, cumulative: ${self.cumulative_cost_usd:.6f}")
            
            if parsed.get("usage"):
                usage = parsed["usage"]
                logger.info(f"Token usage: prompt={usage['prompt_tokens']}, completion={usage['completion_tokens']}, cached={usage['cached_tokens']}")
            
            action_taken: Optional[GameAction] = None
            game_actions_found = []
            
            if parsed["tool_calls"]:
                for tool_call in parsed["tool_calls"]:
                    tool_name = tool_call.get("name", "")
                    tool_input = tool_call.get("input", {})
                    
                    if not tool_name:
                        logger.warning(f"Tool call with empty name: {tool_call}")
                        continue
                    
                    potential_action = self.parse_action_from_tool(tool_name, tool_input)
                    if potential_action:
                        game_actions_found.append(tool_name)
                        if not action_taken:
                            action_taken = potential_action
                            logger.info(f"Action chosen: {action_taken.name}")
                
                if len(game_actions_found) > 1:
                    logger.error(f"LLM called {len(game_actions_found)} game actions in one turn: {game_actions_found}")
                    logger.error("This violates the one-action-per-turn rule and may cause game state corruption")
                    logger.warning(f"Using first action ({action_taken.name}) and ignoring others")
            else:
                logger.warning("LLM response contained no tool calls - may indicate prompting issue")
            
            if action_taken:
                self.consecutive_errors = 0
                self.previous_action_info = self._build_action_info(action_taken)
                
                if self.opencode_recorder and not self.is_playback:
                    try:
                        self.opencode_recorder.save_step(
                            step=self.step_counter,
                            prompt=prompt,
                            messages=[msg for msg in messages],
                            parsed_action={
                                "action": action_taken.value,
                            "reasoning": self.latest_reasoning
                        },
                        total_cost_usd=step_cost
                    )
                    except Exception as e:
                        logger.error(f"Failed to save recording for step {self.step_counter}: {e}")
                        logger.warning("Continuing gameplay despite recording failure")
                
                return action_taken
            
            logger.warning("No valid action found in response")
            self.consecutive_errors += 1
            
        except Exception as e:
            logger.error(f"Error during choose_action: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            
            if self.opencode_server_process and self.opencode_server_process.poll() is not None:
                logger.error(f"OpenCode server has crashed (exit code: {self.opencode_server_process.poll()})")
            
            server_logs = self._get_server_logs(max_lines=20)
            if server_logs and server_logs != "No new logs":
                logger.error(f"OpenCode server logs:\n{server_logs}")
            
            self.consecutive_errors += 1
        
        if self.consecutive_errors > 0:
            logger.warning(f"Falling back to default action (consecutive errors: {self.consecutive_errors}/{self.MAX_CONSECUTIVE_ERRORS})")
        fallback = self._build_fallback_action(latest_frame)
        self.previous_action_info = self._build_action_info(fallback)
        return fallback
    
    def is_complete_tool_call(self, part: dict[str, Any]) -> bool:
        if part.get("type") != "tool":
            return False
        
        tool_info = part.get("tool")
        if isinstance(tool_info, dict):
            return "name" in tool_info
        elif isinstance(tool_info, str):
            return bool(tool_info)
        
        return False
    
    def parse_action_from_tool(self, tool_name: str, tool_input: dict[str, Any]) -> Optional[GameAction]:
        clean_tool_name = tool_name.replace("arc-game-tools_", "")
        
        if clean_tool_name not in self.ACTION_TOOL_MAP:
            logger.debug(f"Non-action tool called: {tool_name}")
            return None
        
        action = self.ACTION_TOOL_MAP[clean_tool_name]
        
        if self.current_frame and self.current_frame.available_actions:
            if action.value not in self.current_frame.available_actions:
                logger.warning(f"LLM chose unavailable action: {action.name} (value={action.value}), available: {self.current_frame.available_actions}")
                return None
        elif not self.current_frame:
            logger.warning("No current_frame available for action validation - may indicate state sync issue")
        
        if action == GameAction.ACTION6:
            x = tool_input.get("x")
            y = tool_input.get("y")
            
            if x is None or y is None:
                logger.error(f"ACTION6 missing required coordinates: x={x}, y={y}")
                return None
            
            try:
                x = int(x)
                y = int(y)
            except (ValueError, TypeError) as e:
                logger.error(f"ACTION6 coordinates not integers: x={x}, y={y}, error={e}")
                return None
            
            if not (0 <= x <= 63 and 0 <= y <= 63):
                logger.warning(f"ACTION6 coordinates out of bounds: x={x}, y={y} (valid range: 0-63)")
                return None
            
            action.set_data({"game_id": self.game_id, "x": x, "y": y})
        else:
            action.set_data({"game_id": self.game_id})
        
        if self.latest_reasoning:
            action_label = tool_name
            if action == GameAction.ACTION6:
                action_label = f"{action_label} (x={tool_input.get('x', 0)}, y={tool_input.get('y', 0)})"
            thought_text = f"{action_label}\n\n{self.latest_reasoning}"
            self.latest_reasoning_dict = {
                "thought": thought_text[:16000]
            }
            logger.info(f"Prepared reasoning for action ({len(thought_text)} chars)")
        else:
            self.latest_reasoning_dict = {}
            logger.warning("No reasoning captured for action")
        
        return action
    
    def cleanup(self, scorecard=None):
        logger.info(f"OpenCodeAgent cleanup starting (played {self.step_counter} steps)...")
        
        if self.session_id:
            try:
                self.opencode_client.delete_session(self.session_id)
                logger.info(f"Deleted OpenCode session: {self.session_id}")
            except Exception as e:
                logger.warning(f"Error deleting session: {e}")
        else:
            logger.debug("No session to delete (session_id was None)")
        
        if self.opencode_client:
            try:
                self.opencode_client.close()
            except Exception as e:
                logger.warning(f"Error closing OpenCode client: {e}")
        
        self._stop_opencode_server()
        logger.info("OpenCodeAgent cleanup completed")
        
        super().cleanup(scorecard)
