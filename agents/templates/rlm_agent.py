"""
RLM Agent — Recursive Language Model agent for ARC-AGI-3.

Uses the `rlms` library to recursively decompose game frames via a Python
REPL environment, with OpenRouter as the default LLM backend.

The agent seeds the REPL with grid-analysis utilities and
game history, then lets the RLM recursively examine patterns, form hypotheses,
and choose actions.
"""

from __future__ import annotations

import json
import logging
import os
import random
import textwrap
from typing import Any

from arcengine import FrameData, GameAction, GameState

from ..agent import Agent

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

VALID_ACTIONS = ["RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"]

ACTION_DESCRIPTIONS = {
    "RESET": "Start/restart the game",
    "ACTION1": "Move Up (W)",
    "ACTION2": "Move Down (S)",
    "ACTION3": "Move Left (A)",
    "ACTION4": "Move Right (D)",
    "ACTION5": "Interact (Enter/Space/Delete)",
    "ACTION6": "Click/Point at (x,y)",
}

# ──────────────────────────────────────────────────────────────────────
# Grid Analysis Utilities (embedded to avoid dependency issues)
# ──────────────────────────────────────────────────────────────────────

COLOR_MAP = {
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "gray",
    6: "magenta",
    7: "orange",
    8: "light blue",
    9: "purple",
    10: "brown",
}

def color_name(val: int) -> str:
    """Convert grid color value to human-readable name."""
    return COLOR_MAP.get(val, f"unknown({val})")

def summarize_grid(grid: list[list[int]], size: int = 64) -> str:
    """Generate a comprehensive text summary of the grid state."""
    if not grid or not grid[0]:
        return "Empty grid"
    
    summary = []
    summary.append(f"Grid size: {len(grid)}x{len(grid[0])}")
    
    # Count colors
    color_counts = {}
    for row in grid:
        for val in row:
            color_counts[val] = color_counts.get(val, 0) + 1
    
    summary.append("Colors present:")
    for val, count in sorted(color_counts.items()):
        if count > 0:
            summary.append(f"  {color_name(val)}: {count} cells")
    
    # Find distinct objects
    objects = []
    visited = set()
    
    for y in range(len(grid)):
        for x in range(len(grid[0])):
            if grid[y][x] != 0 and (x, y) not in visited:
                # BFS to find connected component
                color = grid[y][x]
                queue = [(x, y)]
                visited.add((x, y))
                cells = []
                
                while queue:
                    cx, cy = queue.pop()
                    cells.append((cx, cy))
                    
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nx, ny = cx + dx, cy + dy
                        if (0 <= nx < len(grid[0]) and 0 <= ny < len(grid) and
                            grid[ny][nx] == color and (nx, ny) not in visited):
                            visited.add((nx, ny))
                            queue.append((nx, ny))
                
                if len(cells) > 1:
                    min_x = min(x for x, y in cells)
                    max_x = max(x for x, y in cells)
                    min_y = min(y for x, y in cells)
                    max_y = max(y for x, y in cells)
                    
                    objects.append({
                        'color': color_name(color),
                        'size': len(cells),
                        'bbox': (min_x, min_y, max_x, max_y),
                        'cells': cells
                    })
    
    if objects:
        summary.append(f"Found {len(objects)} distinct objects:")
        for i, obj in enumerate(objects[:5]):  # Limit to first 5 objects
            summary.append(f"  Object {i+1}: {obj['color']} ({obj['size']} cells) at {obj['bbox']}")
    
    return "\n".join(summary)

def diff_summary(prev_grid: list[list[int]], curr_grid: list[list[int]]) -> str:
    """Generate a compact summary of differences between two grids."""
    if not prev_grid or not curr_grid:
        return "No previous grid available"
    
    changes = []
    moved_objects = []
    
    # Track object movements
    prev_objects = _find_objects(prev_grid)
    curr_objects = _find_objects(curr_grid)
    
    for prev_obj in prev_objects:
        # Find closest current object
        best_match = None
        best_dist = float('inf')
        
        for curr_obj in curr_objects:
            if prev_obj['color'] == curr_obj['color'] and abs(prev_obj['size'] - curr_obj['size']) <= 1:
                dist = abs(prev_obj['center'][0] - curr_obj['center'][0]) + abs(prev_obj['center'][1] - curr_obj['center'][1])
                if dist < best_dist and dist < 10:  # Reasonable movement threshold
                    best_dist = dist
                    best_match = curr_obj
        
        if best_match:
            if best_dist > 0:
                moved_objects.append(f"{prev_obj['color']} object moved from {prev_obj['center']} to {best_match['center']}")
    
    # Count pixel-level changes
    pixel_changes = 0
    for y in range(min(len(prev_grid), len(curr_grid))):
        for x in range(min(len(prev_grid[0]), len(curr_grid[0]))):
            if prev_grid[y][x] != curr_grid[y][x]:
                pixel_changes += 1
    
    summary = []
    if pixel_changes > 0:
        summary.append(f"{pixel_changes} pixels changed")
    if moved_objects:
        summary.extend(moved_objects)
    if not summary:
        summary.append("No significant changes detected")
    
    return "; ".join(summary)

def _find_objects(grid: list[list[int]]) -> list[dict]:
    """Helper to find objects in grid."""
    objects = []
    visited = set()
    
    for y in range(len(grid)):
        for x in range(len(grid[0])):
            if grid[y][x] != 0 and (x, y) not in visited:
                color = grid[y][x]
                queue = [(x, y)]
                visited.add((x, y))
                cells = []
                
                while queue:
                    cx, cy = queue.pop()
                    cells.append((cx, cy))
                    
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nx, ny = cx + dx, cy + dy
                        if (0 <= nx < len(grid[0]) and 0 <= ny < len(grid) and
                            grid[ny][nx] == color and (nx, ny) not in visited):
                            visited.add((nx, ny))
                            queue.append((nx, ny))
                
                if len(cells) > 1:
                    center_x = sum(x for x, y in cells) // len(cells)
                    center_y = sum(y for x, y in cells) // len(cells)
                    objects.append({
                        'color': color_name(color),
                        'size': len(cells),
                        'center': (center_x, center_y),
                        'cells': cells
                    })
    
    return objects

def find_player(grid: list[list[int]]) -> dict[str, Any] | None:
    """Try to find the player character in the grid."""
    # Look for distinctive player patterns
    player_patterns = [
        [(2, 0), (2, 1), (2, 2)],  # Vertical red line
        [(1, 0), (2, 0), (3, 0)],  # Horizontal red line
        [(2, 0), (1, 1), (2, 2)],  # T-shape
    ]
    
    for y in range(len(grid) - 2):
        for x in range(len(grid[0]) - 2):
            for pattern in player_patterns:
                matches = True
                for px, py in pattern:
                    if grid[y + py][x + px] != 2:  # Red (player color)
                        matches = False
                        break
                if matches:
                    return {
                        'x': x + 1,
                        'y': y + 1,
                        'bbox': (x, y, x + 2, y + 2),
                        'pattern': pattern
                    }
    
    return None

def find_door(grid: list[list[int]]) -> dict[str, Any] | None:
    """Try to find a door in the grid."""
    # Look for door patterns (usually yellow/green rectangles)
    for y in range(len(grid) - 1):
        for x in range(len(grid[0]) - 2):
            # Yellow rectangle pattern
            if (grid[y][x] == 4 and grid[y][x + 1] == 4 and grid[y][x + 2] == 4 and
                grid[y + 1][x] == 4 and grid[y + 1][x + 2] == 4):
                return {
                    'x': x + 1,
                    'y': y,
                    'inner_pattern': [[grid[y][x + 1]]],
                }
    
    return None

def find_key(grid: list[list[int]], size: int = 64) -> list[list[int]] | None:
    """Try to find a key pattern in the grid."""
    # Simple key pattern detection
    key_patterns = [
        [[3, 0], [3, 3]],  # Green L-shape
        [[3, 3, 0], [0, 3, 3]],  # Green zigzag
    ]
    
    for y in range(len(grid) - 1):
        for x in range(len(grid[0]) - 2):
            for pattern in key_patterns:
                matches = True
                for py, row in enumerate(pattern):
                    for px, val in enumerate(row):
                        if grid[y + py][x + px] != val:
                            matches = False
                            break
                    if not matches:
                        break
                if matches:
                    return pattern
    
    return None

# ──────────────────────────────────────────────────────────────────────
# System prompt for the RLM
# ──────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert AI game agent playing an ARC-AGI-3 interactive reasoning game.

## Your Environment
You have a persistent Python REPL with these tools pre-loaded:
- `grid`: the current 64x64 game grid (list[list[int]])
- `prev_grid`: the previous grid (or None on first turn)
- `memory`: list of past observations (list[dict])
- `hypothesis`: your current theory about the game rules (str)
- `turn_number`: current turn count (int)

## Pre-loaded helper functions:
- `diff_summary(prev, curr)` → compact text diff between two grids
- `find_player(grid)` → {"x": int, "y": int, "bbox": {...}} or None
- `find_door(grid)` → {"x": int, "y": int, "inner_pattern": [[int]]} or None
- `find_key(grid, 64)` → 6x6 key pattern or None
- `summarize_grid(grid, 64)` → full text summary of grid state
- `color_name(val)` → human-readable name for a grid int value
- `color_map` → dict mapping ints to color names

## Your Task
Write Python code to analyze the game state and decide your next action.
You MUST end by setting a variable called `result` to a dict with these keys:

```python
result = {
    "action": "ACTION1",  # one of: RESET, ACTION1, ACTION2, ACTION3, ACTION4, ACTION5, ACTION6
    "reasoning": "I moved up because...",  # explain your reasoning
    "hypothesis": "The game seems to be about...",  # current theory about game rules
    "observation": "I noticed that...",  # what you observed this turn
}
```

IMPORTANT: The `result` variable MUST be valid JSON. No trailing commas. Use double quotes for strings.

## Action Meanings
- RESET: start or restart the game
- ACTION1: Move Up
- ACTION2: Move Down
- ACTION3: Move Left
- ACTION4: Move Right
- ACTION5: Interact (Enter/Space)
- ACTION6: Click at coordinates (requires x, y in result["action_data"])

## Strategy Guidelines
1. First call summarize_grid(grid, 64) to understand the current state
2. If prev_grid exists, call diff_summary(prev_grid, grid) to see what changed
3. Look for patterns, objects, and relationships
4. Form a hypothesis about the game rules
5. Choose an action that tests your hypothesis
6. Update your hypothesis based on observations

## Example
```python
# Analyze the grid
summary = summarize_grid(grid, 64)
print(f"Grid summary: {summary}")

# Check for changes
if prev_grid:
    changes = diff_summary(prev_grid, grid)
    print(f"Changes: {changes}")

# Look for key objects
player = find_player(grid)
door = find_door(grid)
key = find_key(grid, 64)

print(f"Player: {player}")
print(f"Door: {door}")
print(f"Key: {key}")

# Make decision
result = {
    "action": "ACTION3",
    "reasoning": "I see the player at position and need to move left to reach the door",
    "hypothesis": "This is a maze game where I need to collect keys and reach doors",
    "observation": f"Found player at {player}, door at {door}, key at {key}"
}
```
""")

# ──────────────────────────────────────────────────────────────────────
# RLM Agent Implementation
# ──────────────────────────────────────────────────────────────────────

class RLMAgent(Agent):
    """
    Recursive Language Model agent for ARC-AGI-3.
    
    Uses the `rlms` library to create a recursive reasoning loop with a Python REPL.
    The agent can analyze grid patterns, form hypotheses, and make informed decisions.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        # RLM configuration
        self.BACKEND = os.getenv("RLM_BACKEND", "openrouter")
        self.MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
        self.ENVIRONMENT = os.getenv("RLM_ENVIRONMENT", "local")
        self.VERBOSE = os.getenv("RLM_VERBOSE", "false").lower() == "true"
        
        # Agent state
        self.hypothesis = "Unknown game. Need to explore and discover the rules."
        self.memory: list[dict[str, Any]] = []
        self.turn_number = 0
        self._total_rlm_calls = 0
        self._consecutive_no_change = 0
        self._last_grid_hash = None
        
        # Initialize RLM client
        self._rlm_client = self._create_rlm_client()
        
        logger.info(
            f"RLMAgent initialized: backend={self.BACKEND}, model={self.MODEL}, env={self.ENVIRONMENT}"
        )

    def _create_rlm_client(self) -> Any:
        """Create and configure the RLM client."""
        try:
            import rlms
            backend_kwargs = self._build_backend_kwargs()
            client = rlms.RLM(**backend_kwargs)
            return client
        except ImportError as e:
            logger.error(f"Failed to import rlms: {e}")
            raise ImportError("RLM agent requires 'rlms' package. Install with: pip install rlms")
        except Exception as e:
            logger.error(f"Failed to create RLM client: {e}")
            raise

    def _build_backend_kwargs(self) -> dict[str, Any]:
        """Build backend-specific configuration for RLM."""
        if self.BACKEND == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable required for OpenRouter backend")
            
            return {
                "backend": "openrouter",
                "model": self.MODEL,
                "api_key": api_key,
                "environment": self.ENVIRONMENT,
                "verbose": self.VERBOSE,
            }
        
        elif self.BACKEND == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable required for OpenAI backend")
            
            return {
                "backend": "openai",
                "model": os.getenv("OPENAI_MODEL", "gpt-4"),
                "api_key": api_key,
                "environment": self.ENVIRONMENT,
                "verbose": self.VERBOSE,
            }
        
        elif self.BACKEND == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable required for Anthropic backend")
            
            return {
                "backend": "anthropic",
                "model": os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
                "api_key": api_key,
                "environment": self.ENVIRONMENT,
                "verbose": self.VERBOSE,
            }
        
        else:
            raise ValueError(f"Unsupported backend: {self.BACKEND}")

    @property
    def MAX_ACTIONS(self) -> int:
        return 80

    def choose_action(self, frame: FrameData) -> GameAction:
        """Choose the next action using recursive reasoning."""
        self.turn_number += 1
        
        # Handle game reset
        if frame.state == GameState.RESET:
            self._handle_game_reset()
            return GameAction.RESET
        
        # Check for stuck state
        if self._is_stuck(frame):
            logger.info("Agent appears stuck, taking exploratory action")
            return self._exploratory_action()
        
        # Build RLM prompt with current context
        prompt = self._build_prompt(frame)
        
        try:
            # Call RLM
            self._total_rlm_calls += 1
            logger.info(f"RLM call #{self._total_rlm_calls} | turn {self.turn_number} | model={self.MODEL}")
            
            rlm_result = self._rlm_client.completion(prompt)
            
            # Parse result
            action, reasoning_meta = self._parse_rlm_result(rlm_result, frame)
            
            # Log reasoning
            logger.info(f"Chosen action: {action.name} | {reasoning_meta.get('reasoning', 'N/A')[:100]}")
            
            return action
            
        except Exception as e:
            logger.error(f"RLM call failed: {e}")
            return self._fallback_action()

    def _build_prompt(self, frame: FrameData) -> str:
        """Build the complete prompt for the RLM."""
        # Get previous grid for comparison
        prev_grid = self.memory[-1].get("grid") if self.memory else None
        
        # Build REPL namespace
        namespace = {
            "grid": frame.frame,
            "prev_grid": prev_grid,
            "memory": self.memory,
            "hypothesis": self.hypothesis,
            "turn_number": self.turn_number,
            # Helper functions
            "diff_summary": diff_summary,
            "find_player": find_player,
            "find_door": find_door,
            "find_key": find_key,
            "summarize_grid": summarize_grid,
            "color_name": color_name,
            "color_map": COLOR_MAP,
        }
        
        # Build prompt
        prompt_parts = [
            SYSTEM_PROMPT,
            "\n" + "="*80 + "\n",
            "## Current Game State\n",
            f"Turn: {self.turn_number}\n",
            f"Current hypothesis: {self.hypothesis}\n",
            f"Memory entries: {len(self.memory)}\n",
            "\n" + "="*80 + "\n",
            "## Your Analysis\n",
            "Write Python code to analyze the current grid and decide your action.\n",
            "Remember to set the `result` variable with your decision.\n",
        ]
        
        return "\n".join(prompt_parts)

    def _parse_rlm_result(
        self, rlm_result: Any, latest_frame: FrameData
    ) -> tuple[GameAction, dict[str, Any]]:
        """
        Parse the RLM response and extract the chosen action.
        """
        response_text = str(rlm_result.response) if rlm_result else ""
        
        # Debug logging to see what we got
        logger.debug(f"RLM Response (first 500 chars): {response_text[:500]}")

        # Try to parse as JSON from the response
        parsed = self._extract_result_dict(response_text)

        if parsed and "action" in parsed:
            action_name = parsed["action"].upper().strip()
            logger.debug(f"Successfully parsed action: {action_name}")

            # Validate action name
            if action_name not in VALID_ACTIONS:
                logger.warning(
                    f"Invalid action '{action_name}' from RLM, falling back"
                )
                action = self._fallback_action()
                action_name = action.name
            else:
                action = GameAction.from_name(action_name)

            # Handle ACTION6 coordinates
            if action_name == "ACTION6" and "action_data" in parsed:
                try:
                    action.set_data(parsed["action_data"])
                except Exception as e:
                    logger.warning(f"Invalid ACTION6 data: {e}, falling back")
                    action = self._fallback_action()

            # Extract reasoning metadata
            reasoning = parsed.get("reasoning", "No reasoning provided")
            hypothesis = parsed.get("hypothesis", self.hypothesis)
            observation = parsed.get("observation", "")

            # Update agent state
            self.hypothesis = hypothesis
            self._record_observation(action.name, observation, reasoning)

            reasoning_meta = {
                "model": self.MODEL,
                "backend": self.BACKEND,
                "agent_type": "rlm_agent",
                "action_chosen": action.name,
                "reasoning": reasoning,
                "hypothesis": hypothesis,
                "observation": observation,
                "turn": self.turn_number,
                "rlm_calls_total": self._total_rlm_calls,
                "consecutive_no_change": self._consecutive_no_change,
                "game_context": {
                    "score": latest_frame.levels_completed,
                    "state": latest_frame.state.name,
                    "action_counter": self.action_counter,
                    "frame_count": len([latest_frame]),
                },
                "response_preview": reasoning[:300],
            }

            return action, reasoning_meta

        # If parsing failed, try extracting action from text
        logger.warning("Could not parse structured result from RLM, attempting text extraction")
        logger.debug(f"Failed to parse response: {response_text[:200]}")
        action, meta = self._extract_action_from_text(response_text, latest_frame)
        return action, meta

    def _extract_result_dict(self, text: str) -> dict[str, Any] | None:
        """
        Try to extract a `result = {...}` dict from the RLM response text.
        Uses multiple strategies: JSON parsing, regex extraction.
        """
        import re

        # Strategy 1: Look for a JSON block in the response
        json_patterns = [
            r'result\s*=\s*(\{[^}]+\})',
            r'```(?:json)?\s*(\{[^}]+\})\s*```',
            r'(\{"action":\s*"[^"]+?"[^}]*\})',
            r'(\{[^"\'\s]*action[^}]*\})',  # More flexible action matching
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                try:
                    # Clean up common issues
                    cleaned = match.replace("'", '"')
                    # Handle trailing commas
                    cleaned = re.sub(r',\s*}', '}', cleaned)
                    cleaned = re.sub(r',\s*]', ']', cleaned)
                    # Fix common JSON issues
                    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                    
                    parsed = json.loads(cleaned)
                    if isinstance(parsed, dict) and "action" in parsed:
                        return parsed
                except (json.JSONDecodeError, ValueError) as e:
                    continue

        # Strategy 2: Look for action keyword with more flexible patterns
        action_patterns = [
            r'["\']?action["\']?\s*[:=]\s*["\']?(ACTION\d|RESET)["\']?',
            r'action\s*=\s*["\']?(ACTION\d|RESET)["\']?',
            r'I\s+(?:choose|select|will|should)\s+(?:action\s+)?["\']?(ACTION\d|RESET)["\']?',
        ]

        for pattern in action_patterns:
            action_match = re.search(pattern, text, re.IGNORECASE)
            if action_match:
                action_name = action_match.group(1).upper()
                
                # Try to extract reasoning and observation
                reasoning_match = re.search(
                    r'["\']?reasoning["\']?\s*[:=]\s*["\']([^"\']+)["\']',
                    text,
                    re.IGNORECASE,
                )
                
                obs_match = re.search(
                    r'["\']?observation["\']?\s*[:=]\s*["\']([^"\']+)["\']',
                    text,
                    re.IGNORECASE,
                )
                
                hypothesis_match = re.search(
                    r'["\']?hypothesis["\']?\s*[:=]\s*["\']([^"\']+)["\']',
                    text,
                    re.IGNORECASE,
                )
                
                return {
                    "action": action_name,
                    "reasoning": reasoning_match.group(1) if reasoning_match else "Extracted from text",
                    "hypothesis": hypothesis_match.group(1) if hypothesis_match else self.hypothesis,
                    "observation": obs_match.group(1) if obs_match else "Parsed from unstructured response",
                }

        # Strategy 3: Try parsing the entire response as JSON
        try:
            cleaned_text = text.strip()
            if cleaned_text.startswith('{') and cleaned_text.endswith('}'):
                parsed = json.loads(cleaned_text)
                if isinstance(parsed, dict) and "action" in parsed:
                    return parsed
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 4: Look for any ACTION mention as last resort
        action_mention = re.search(r'\b(ACTION\d|RESET)\b', text, re.IGNORECASE)
        if action_mention:
            action_name = action_mention.group(1).upper()
            return {
                "action": action_name,
                "reasoning": "Found action mention in text",
                "hypothesis": self.hypothesis,
                "observation": "Minimal extraction from text",
            }

        return None

    def _extract_action_from_text(
        self, text: str, latest_frame: FrameData
    ) -> tuple[GameAction, dict[str, Any]]:
        """Last-resort extraction: scan for action keywords in text."""
        import re

        # Look for any action mention
        action_match = re.search(r'\b(ACTION\d|RESET)\b', text, re.IGNORECASE)
        if action_match:
            action_name = action_match.group(1).upper()
            if action_name in VALID_ACTIONS:
                action = GameAction.from_name(action_name)
                
                meta = {
                    "model": self.MODEL,
                    "backend": self.BACKEND,
                    "agent_type": "rlm_agent",
                    "action_chosen": action.name,
                    "reasoning": "Extracted from unstructured text",
                    "hypothesis": self.hypothesis,
                    "observation": "Fallback parsing from text",
                    "turn": self.turn_number,
                    "rlm_calls_total": self._total_rlm_calls,
                    "consecutive_no_change": self._consecutive_no_change,
                    "game_context": {
                        "score": latest_frame.levels_completed,
                        "state": latest_frame.state.name,
                        "action_counter": self.action_counter,
                        "frame_count": len([latest_frame]),
                    },
                    "response_preview": text[:300],
                    "fallback": True,
                }
                
                return action, meta

        # Ultimate fallback
        action = self._fallback_action()
        meta = {
            "model": self.MODEL,
            "backend": self.BACKEND,
            "agent_type": "rlm_agent",
            "action_chosen": action.name,
            "reasoning": "Ultimate fallback - no action found",
            "hypothesis": self.hypothesis,
            "observation": "Complete parsing failure",
            "turn": self.turn_number,
            "rlm_calls_total": self._total_rlm_calls,
            "consecutive_no_change": self._consecutive_no_change,
            "game_context": {
                "score": latest_frame.levels_completed,
                "state": latest_frame.state.name,
                "action_counter": self.action_counter,
                "frame_count": len([latest_frame]),
            },
            "response_preview": text[:300],
            "fallback": True,
        }
        
        return action, meta

    def _handle_game_reset(self) -> None:
        """Handle game reset by clearing memory and resetting state."""
        logger.info("Game reset detected, clearing memory")
        self.memory.clear()
        self.hypothesis = "Game reset. Starting fresh exploration."
        self.turn_number = 0
        self._consecutive_no_change = 0
        self._last_grid_hash = None

    def _is_stuck(self, frame: FrameData) -> bool:
        """Detect if the agent is stuck in a loop."""
        # Check if grid hasn't changed
        current_hash = hash(str(frame.frame))
        if current_hash == self._last_grid_hash:
            self._consecutive_no_change += 1
        else:
            self._consecutive_no_change = 0
            self._last_grid_hash = current_hash
        
        # Stuck if no change for multiple turns
        return self._consecutive_no_change >= 5

    def _exploratory_action(self) -> GameAction:
        """Choose an exploratory action to get unstuck."""
        # Random exploration
        return random.choice([
            GameAction.ACTION1,  # Up
            GameAction.ACTION2,  # Down
            GameAction.ACTION3,  # Left
            GameAction.ACTION4,  # Right
            GameAction.ACTION5,  # Interact
        ])

    def _fallback_action(self) -> GameAction:
        """Fallback action when all else fails."""
        logger.warning("Using fallback action")
        return GameAction.ACTION5  # Interact is usually safe

    def _record_observation(self, action: str, observation: str, reasoning: str) -> None:
        """Record observation in memory."""
        entry = {
            "turn": self.turn_number,
            "action": action,
            "observation": observation,
            "reasoning": reasoning,
            "hypothesis": self.hypothesis,
            "timestamp": self.turn_number,
        }
        
        self.memory.append(entry)
        
        # Keep memory size manageable
        if len(self.memory) > 50:
            self.memory = self.memory[-50:]

    def is_done(self) -> bool:
        """Check if the agent should stop."""
        return self.action_counter >= self.MAX_ACTIONS
