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
        """Convert grid to semantic description (optimized for token efficiency)."""
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

        # Build description - SIMPLIFIED to reduce tokens
        desc = f"Grid: {width}x{height}. "

        # Count only significant colors (>1% of grid)
        color_counts = {}
        total_cells = width * height
        for row in grid:
            for cell in row:
                color = color_names.get(cell, f"color-{cell}")
                color_counts[color] = color_counts.get(color, 0) + 1

        # Only include colors that are >1% of grid
        significant_colors = [
            (color, count)
            for color, count in color_counts.items()
            if count > total_cells * 0.01
        ]
        if significant_colors:
            desc += "Colors: "
            desc += ", ".join(
                [
                    f"{color}({count})"
                    for color, count in sorted(
                        significant_colors, key=lambda x: x[1], reverse=True
                    )[:5]
                ]
            )

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

        # Build OPTIMIZED prompt (shorter to save tokens)
        recent_actions = [a["action"] for a in self.action_history[-3:]]
        prompt = f"""Abstract reasoning puzzle. Discover rules through experimentation.

{current_grid_desc}
State: {latest_frame.state.name} | Levels: {latest_frame.levels_completed}/{latest_frame.win_levels} | Action: {self.action_counter}

Changes: {changes}
Recent actions: {recent_actions}

Actions: ACTION1(up), ACTION2(down), ACTION3(left), ACTION4(right), RESET

Analyze patterns, form hypothesis, choose action. Respond in JSON:
{{"reasoning": "brief analysis", "hypothesis": "game rules", "action": "ACTION_NAME", "confidence": "high/medium/low"}}
"""
        return prompt

    def call_claude_with_thinking(self, prompt: str) -> dict[str, Any]:
        """Call Claude API with extended thinking enabled."""
        try:
            # Build messages
            messages = [{"role": "user", "content": prompt}]

            # Call Claude with extended thinking (reduced budget to save costs)
            response = self.client.messages.create(
                model=self.MODEL,
                max_tokens=2000,
                thinking={"type": "enabled", "budget_tokens": 1500},
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

        # ANTI-STUCK MECHANISM: If same action repeated 3+ times with no progress, force different action
        if len(self.action_history) >= 3:
            last_3_actions = [a["action"] for a in self.action_history[-3:]]
            if (
                len(set(last_3_actions)) == 1
                and last_3_actions[0] == action_name
                and self.stuck_counter > 2
            ):
                # Force a different action
                all_actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
                available = [a for a in all_actions if a != action_name]
                action_name = available[self.stuck_counter % len(available)]
                reasoning += f" [Auto-switched from repeated {last_3_actions[0]}]"
                logger.info(f"Forced action change to break repetition: {action_name}")

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
