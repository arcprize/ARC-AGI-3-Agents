"""Claude Extended Thinking Agent for ARC-AGI-3 challenges.

This agent leverages Claude Sonnet 4.5's extended thinking capabilities to solve
abstract reasoning puzzles through hypothesis-driven exploration and pattern recognition.

Architecture:
1. Observe: Convert grid frames to semantic descriptions
2. Reason: Use extended thinking to analyze patterns
3. Hypothesize: Build mental model of puzzle rules
4. Act: Select action based on hypothesis
5. Learn: Update hypothesis based on results
"""

import json
import logging
import os
from typing import Any, List

from anthropic import Anthropic
from arcengine import FrameData, GameAction, GameState

from ..agent import Agent

logger = logging.getLogger(__name__)


class HypothesisTracker:
    """Tracks and manages hypotheses about game mechanics."""

    def __init__(self):
        self.hypotheses: List[dict[str, Any]] = []
        self.confirmed_rules: List[str] = []
        self.discarded_hypotheses: List[str] = []

    def add_hypothesis(self, hypothesis: str, reasoning: str):
        """Add a new hypothesis to track."""
        self.hypotheses.append(
            {
                "hypothesis": hypothesis,
                "reasoning": reasoning,
                "evidence_count": 0,
                "counter_evidence_count": 0,
            }
        )

    def update(self, findings: str):
        """Update hypotheses based on new findings."""
        # Simple implementation: track findings
        for h in self.hypotheses:
            h["evidence_count"] += 1

    def get_summary(self) -> str:
        """Get a summary of current hypotheses and confirmed rules."""
        summary = "Confirmed rules:\n"
        for rule in self.confirmed_rules[-5:]:  # Last 5 confirmed rules
            summary += f"- {rule}\n"

        summary += "\nActive hypotheses:\n"
        for h in self.hypotheses[-3:]:  # Last 3 hypotheses
            summary += f"- {h['hypothesis']}\n"

        return (
            summary if self.confirmed_rules or self.hypotheses else "No hypotheses yet."
        )


class ClaudeThinkingAgent(Agent):
    """Extended thinking agent using Claude Sonnet 4.5 for abstract reasoning."""

    MAX_ACTIONS = 200  # Allow more exploration than default
    MODEL = "claude-sonnet-4-5-20250929"
    MAX_HYPOTHESIS_HISTORY = 10

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Initialize Anthropic client
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not found in environment")
            api_key = ""
        self.client = Anthropic(api_key=api_key)

        # Reasoning and memory components
        self.hypothesis_tracker = HypothesisTracker()
        self.action_history: List[dict[str, Any]] = []
        self.conversation_history: List[dict[str, Any]] = []
        self.stuck_counter = 0
        self.last_levels_completed = 0

        # Token tracking
        self.total_tokens = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Determine if the agent should stop playing."""
        # Stop if we won
        if latest_frame.state == GameState.WIN:
            logger.info("Agent completed the game - WIN state reached")
            return True

        # Stop if stuck (no progress for many actions)
        if self.stuck_counter > 50:
            logger.warning(
                f"Agent appears stuck (no progress for {self.stuck_counter} actions)"
            )
            return True

        return False

    def clear_history(self):
        """Clear history when transitioning between levels."""
        logger.info("Clearing history for new level")
        self.action_history = []
        self.conversation_history = []
        self.stuck_counter = 0
        # Don't clear hypothesis tracker - knowledge should carry over

    def describe_grid(self, grid: List[List[int]]) -> str:
        """Convert grid to semantic description."""
        if not grid or not grid[0]:
            return "Empty grid"

        height = len(grid)
        width = len(grid[0])

        # Color mapping
        color_names = {
            0: "white",
            1: "light gray",
            2: "gray",
            3: "dark gray",
            4: "darker gray",
            5: "black",
            6: "pink",
            7: "light pink",
            8: "red",
            9: "blue",
            10: "light blue",
            11: "yellow",
            12: "orange",
            13: "dark red",
            14: "green",
            15: "purple",
        }

        # Build description
        desc = f"Grid size: {width}x{height} (width x height)\n\n"

        # Count colors
        color_counts = {}
        for row in grid:
            for cell in row:
                color = color_names.get(cell, f"color-{cell}")
                color_counts[color] = color_counts.get(color, 0) + 1

        desc += "Color distribution:\n"
        for color, count in sorted(
            color_counts.items(), key=lambda x: x[1], reverse=True
        ):
            if count > 0:
                desc += f"  - {color}: {count} cells\n"

        # Add grid representation
        desc += "\nGrid (top to bottom, left to right):\n"
        for y, row in enumerate(grid):
            row_desc = f"Row {y}: "
            row_desc += ", ".join(
                [color_names.get(cell, f"color-{cell}") for cell in row]
            )
            desc += row_desc + "\n"

        return desc

    def analyze_frame_changes(self, frames: List[FrameData]) -> str:
        """Analyze what changed between the last two frames."""
        if len(frames) < 2:
            return "First frame - no changes to analyze"

        prev_frame = frames[-2]
        curr_frame = frames[-1]

        changes = []

        # Check levels completed
        if curr_frame.levels_completed != prev_frame.levels_completed:
            changes.append(
                f"Levels completed changed: {prev_frame.levels_completed} → {curr_frame.levels_completed}"
            )

        # Check state changes
        if curr_frame.state != prev_frame.state:
            changes.append(
                f"State changed: {prev_frame.state.name} → {curr_frame.state.name}"
            )

        # Check if grid changed
        if len(curr_frame.frame) > 0 and len(prev_frame.frame) > 0:
            curr_grid = curr_frame.frame[-1]
            prev_grid = prev_frame.frame[-1]

            if curr_grid != prev_grid:
                changes.append("Grid contents changed")

        if not changes:
            changes.append("No observable changes")

        return "\n".join(changes)

    def build_thinking_prompt(
        self, frames: List[FrameData], latest_frame: FrameData
    ) -> str:
        """Create prompt that encourages step-by-step reasoning."""
        # Describe current state
        current_grid_desc = "No grid available"
        if latest_frame.frame and len(latest_frame.frame) > 0:
            current_grid_desc = self.describe_grid(latest_frame.frame[-1])

        # Analyze recent changes
        changes = self.analyze_frame_changes(frames)

        # Get available actions (convert to strings if they're integers)
        available_actions_raw = latest_frame.available_actions or [1, 2, 3, 4, 5]
        if available_actions_raw and isinstance(available_actions_raw[0], int):
            # Convert action IDs to action names
            action_map = {
                1: "ACTION1",
                2: "ACTION2",
                3: "ACTION3",
                4: "ACTION4",
                5: "RESET",
                6: "ACTION5",
                7: "ACTION6",
                8: "ACTION7",
            }
            available_actions = [
                action_map.get(action_id, f"ACTION{action_id}")
                for action_id in available_actions_raw
            ]
        else:
            available_actions = available_actions_raw

        # Build comprehensive prompt
        prompt = f"""You are solving an abstract reasoning puzzle game. This is a 2D grid-based game where you need to discover the rules through experimentation.

CURRENT STATE:
{current_grid_desc}

GAME STATE: {latest_frame.state.name}
LEVELS COMPLETED: {latest_frame.levels_completed}
ACTIONS TAKEN SO FAR: {self.action_counter}

AVAILABLE ACTIONS:
{", ".join(available_actions)}
- ACTION1: MOVE_UP
- ACTION2: MOVE_DOWN
- ACTION3: MOVE_LEFT
- ACTION4: MOVE_RIGHT
- RESET: Start new level or reset current level

RECENT CHANGES:
{changes}

PREVIOUS KNOWLEDGE:
{self.hypothesis_tracker.get_summary()}

YOUR TASK:
Think step-by-step to determine the best action. Consider:

1. PATTERN OBSERVATION
   - What patterns do you see in the grid?
   - How did the grid change after the last action?
   - Are there any symmetries or repeating elements?

2. HYPOTHESIS FORMATION
   - What might the game rules be?
   - How do actions affect the grid?
   - What is the goal/objective?

3. ACTION SELECTION
   - Which action would best test your hypothesis?
   - Which action moves you closer to the goal?
   - Should you explore or exploit current knowledge?

Please think through this carefully and provide:
1. Your detailed reasoning
2. Your current hypothesis about game mechanics
3. The action you want to take (one of: {", ".join(available_actions)})

Format your response as JSON:
{{
    "reasoning": "your detailed step-by-step thinking",
    "hypothesis": "your current understanding of game rules",
    "action": "ACTION_NAME",
    "confidence": "high/medium/low"
}}
"""
        return prompt

    def call_claude_with_thinking(self, prompt: str) -> dict[str, Any]:
        """Call Claude API with extended thinking enabled."""
        try:
            # Build messages
            messages = [{"role": "user", "content": prompt}]

            # Call Claude with extended thinking
            response = self.client.messages.create(
                model=self.MODEL,
                max_tokens=4000,
                thinking={"type": "enabled", "budget_tokens": 3000},
                messages=messages,
            )

            # Track tokens
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens
            self.total_tokens = self.total_input_tokens + self.total_output_tokens

            logger.info(
                f"Claude API call - Input tokens: {response.usage.input_tokens}, Output tokens: {response.usage.output_tokens}"
            )

            # Extract thinking and response
            thinking_content = None
            response_content = None

            for block in response.content:
                if block.type == "thinking":
                    thinking_content = block.thinking
                elif block.type == "text":
                    response_content = block.text

            if thinking_content:
                logger.info(f"Extended thinking: {thinking_content[:200]}...")

            if not response_content:
                logger.warning("No text response from Claude")
                return {
                    "reasoning": "No response from Claude",
                    "hypothesis": "Unknown",
                    "action": "RESET",
                    "confidence": "low",
                }

            # Parse JSON response
            try:
                # Try to extract JSON from response
                json_start = response_content.find("{")
                json_end = response_content.rfind("}") + 1

                if json_start >= 0 and json_end > json_start:
                    json_str = response_content[json_start:json_end]
                    parsed = json.loads(json_str)
                    return parsed
                else:
                    logger.warning("No JSON found in response")
                    return {
                        "reasoning": response_content,
                        "hypothesis": "Unable to parse hypothesis",
                        "action": "ACTION1",
                        "confidence": "low",
                    }

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {e}")
                logger.debug(f"Response content: {response_content}")

                # Fallback: try to extract action from text
                response_upper = response_content.upper()
                for action in ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "RESET"]:
                    if action in response_upper:
                        return {
                            "reasoning": response_content,
                            "hypothesis": "Extracted from text response",
                            "action": action,
                            "confidence": "low",
                        }

                # Last resort
                return {
                    "reasoning": response_content,
                    "hypothesis": "Failed to parse",
                    "action": "ACTION1",
                    "confidence": "low",
                }

        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            # Return safe default
            return {
                "reasoning": f"API call failed: {str(e)}",
                "hypothesis": "Error state",
                "action": "RESET",
                "confidence": "low",
            }

    def choose_action(
        self, frames: List[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Choose action using Claude's extended thinking."""

        # Handle level transitions
        if latest_frame.full_reset:
            logger.info("Full reset detected - starting new level")
            self.clear_history()
            return GameAction.RESET

        # First action must be RESET
        if len(self.action_history) == 0:
            logger.info("First action - sending RESET")
            self.action_history.append(
                {"action": "RESET", "reasoning": "Initial reset to start game"}
            )
            return GameAction.RESET

        # Build thinking prompt
        prompt = self.build_thinking_prompt(frames, latest_frame)

        # Get Claude's response with extended thinking
        response = self.call_claude_with_thinking(prompt)

        # Extract action
        action_name = response.get("action", "ACTION1").upper()
        reasoning = response.get("reasoning", "No reasoning provided")
        hypothesis = response.get("hypothesis", "No hypothesis")
        confidence = response.get("confidence", "unknown")

        logger.info(f"Claude chose: {action_name} (confidence: {confidence})")
        logger.debug(f"Reasoning: {reasoning[:200]}...")

        # Update hypothesis tracker
        self.hypothesis_tracker.add_hypothesis(hypothesis, reasoning)

        # Track stuck state
        if latest_frame.levels_completed == self.last_levels_completed:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            self.last_levels_completed = latest_frame.levels_completed

        # Record this action
        self.action_history.append(
            {
                "action": action_name,
                "reasoning": reasoning,
                "hypothesis": hypothesis,
                "confidence": confidence,
                "levels_completed": latest_frame.levels_completed,
            }
        )

        # Map to GameAction
        try:
            action = GameAction.from_name(action_name)
        except (ValueError, AttributeError):
            logger.warning(f"Unknown action {action_name}, defaulting to ACTION1")
            action = GameAction.ACTION1

        # Attach reasoning metadata
        action.reasoning = {
            "model": self.MODEL,
            "agent_type": "claude_thinking",
            "hypothesis": hypothesis,
            "confidence": confidence,
            "reasoning_preview": reasoning[:300] + "..."
            if len(reasoning) > 300
            else reasoning,
            "total_tokens": self.total_tokens,
            "action_counter": self.action_counter,
            "stuck_counter": self.stuck_counter,
        }

        return action
