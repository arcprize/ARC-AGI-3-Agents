"""Unit tests for ClaudeThinkingAgent."""

import os
from unittest.mock import Mock, patch

import pytest
from arcengine import FrameData, GameAction, GameState

from agents.templates.claude_thinking_agent import (
    ClaudeThinkingAgent,
    HypothesisTracker,
)


class TestHypothesisTracker:
    """Test the HypothesisTracker component."""

    def test_initialization(self):
        """Test hypothesis tracker initializes correctly."""
        tracker = HypothesisTracker()
        assert tracker.hypotheses == []
        assert tracker.confirmed_rules == []
        assert tracker.discarded_hypotheses == []

    def test_add_hypothesis(self):
        """Test adding a hypothesis."""
        tracker = HypothesisTracker()
        tracker.add_hypothesis(
            "Player moves with arrow keys", "Based on action observations"
        )

        assert len(tracker.hypotheses) == 1
        assert tracker.hypotheses[0]["hypothesis"] == "Player moves with arrow keys"
        assert tracker.hypotheses[0]["reasoning"] == "Based on action observations"
        assert tracker.hypotheses[0]["evidence_count"] == 0

    def test_update_hypotheses(self):
        """Test updating hypotheses with new findings."""
        tracker = HypothesisTracker()
        tracker.add_hypothesis("Test hypothesis", "Test reasoning")
        tracker.update("New finding")

        assert tracker.hypotheses[0]["evidence_count"] == 1

    def test_get_summary_empty(self):
        """Test getting summary when no hypotheses exist."""
        tracker = HypothesisTracker()
        summary = tracker.get_summary()
        assert "No hypotheses yet" in summary

    def test_get_summary_with_data(self):
        """Test getting summary with hypotheses and rules."""
        tracker = HypothesisTracker()
        tracker.confirmed_rules.append("Rule 1: Player can move")
        tracker.add_hypothesis("Hypothesis 1", "Reasoning 1")

        summary = tracker.get_summary()
        assert "Rule 1: Player can move" in summary
        assert "Hypothesis 1" in summary


class TestClaudeThinkingAgent:
    """Test the ClaudeThinkingAgent."""

    @pytest.fixture
    def mock_env(self):
        """Create a mock environment wrapper."""
        env = Mock()
        env.observation_space = Mock()
        return env

    @pytest.fixture
    def agent(self, mock_env):
        """Create a ClaudeThinkingAgent instance for testing."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            agent = ClaudeThinkingAgent(
                card_id="test_card",
                game_id="test_game",
                agent_name="claude_test",
                ROOT_URL="https://test.url",
                record=False,
                arc_env=mock_env,
            )
        return agent

    def test_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.MAX_ACTIONS == 200
        assert agent.MODEL == "claude-sonnet-4-5-20250929"
        assert isinstance(agent.hypothesis_tracker, HypothesisTracker)
        assert agent.action_history == []
        assert agent.stuck_counter == 0
        assert agent.total_tokens == 0

    def test_is_done_on_win(self, agent):
        """Test agent stops when game is won."""
        frame = FrameData(levels_completed=1, state=GameState.WIN)
        assert agent.is_done([], frame) is True

    def test_is_done_when_stuck(self, agent):
        """Test agent stops when stuck."""
        agent.stuck_counter = 51
        frame = FrameData(levels_completed=0, state=GameState.NOT_FINISHED)
        assert agent.is_done([], frame) is True

    def test_is_done_continues_playing(self, agent):
        """Test agent continues when not done."""
        frame = FrameData(levels_completed=0, state=GameState.NOT_FINISHED)
        assert agent.is_done([], frame) is False

    def test_clear_history(self, agent):
        """Test clearing history resets appropriate state."""
        agent.action_history = [{"action": "TEST"}]
        agent.stuck_counter = 10

        agent.clear_history()

        assert agent.action_history == []
        assert agent.stuck_counter == 0
        # Hypothesis tracker should persist
        assert agent.hypothesis_tracker is not None

    def test_describe_grid_empty(self, agent):
        """Test describing an empty grid."""
        result = agent.describe_grid([])
        assert result == "Empty grid"

    def test_describe_grid_simple(self, agent):
        """Test describing a simple grid."""
        grid = [[0, 1], [2, 3]]
        result = agent.describe_grid(grid)

        assert "Grid size: 2x2" in result
        assert "white" in result
        assert "light gray" in result

    def test_analyze_frame_changes_first_frame(self, agent):
        """Test analyzing changes on first frame."""
        result = agent.analyze_frame_changes([])
        assert "First frame" in result

    def test_analyze_frame_changes_no_changes(self, agent):
        """Test analyzing when nothing changed."""
        frame1 = FrameData(
            levels_completed=0, state=GameState.NOT_FINISHED, frame=[[[0]]]
        )
        frame2 = FrameData(
            levels_completed=0, state=GameState.NOT_FINISHED, frame=[[[0]]]
        )

        result = agent.analyze_frame_changes([frame1, frame2])
        assert "No observable changes" in result

    def test_analyze_frame_changes_level_complete(self, agent):
        """Test analyzing when level is completed."""
        frame1 = FrameData(
            levels_completed=0, state=GameState.NOT_FINISHED, frame=[[[0]]]
        )
        frame2 = FrameData(
            levels_completed=1, state=GameState.NOT_FINISHED, frame=[[[0]]]
        )

        result = agent.analyze_frame_changes([frame1, frame2])
        assert "0 → 1" in result

    def test_build_thinking_prompt(self, agent):
        """Test building the thinking prompt."""
        frame = FrameData(
            levels_completed=0,
            state=GameState.NOT_FINISHED,
            frame=[[[0, 1], [2, 3]]],
            available_actions=[1, 2, 5],  # Action IDs as integers
        )

        prompt = agent.build_thinking_prompt([frame], frame)

        assert "abstract reasoning puzzle" in prompt.lower()
        assert "ACTION1" in prompt
        assert "ACTION2" in prompt
        assert "RESET" in prompt
        assert "PATTERN OBSERVATION" in prompt
        assert "HYPOTHESIS FORMATION" in prompt

    @patch("agents.templates.claude_thinking_agent.Anthropic")
    def test_call_claude_with_thinking_success(self, mock_anthropic_class, agent):
        """Test successful Claude API call."""
        # Mock response
        mock_response = Mock()
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_response.content = [
            Mock(type="thinking", thinking="Let me think..."),
            Mock(
                type="text",
                text='{"reasoning": "test", "hypothesis": "test", "action": "ACTION1", "confidence": "high"}',
            ),
        ]

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client
        agent.client = mock_client

        result = agent.call_claude_with_thinking("test prompt")

        assert result["action"] == "ACTION1"
        assert result["reasoning"] == "test"
        assert result["confidence"] == "high"
        assert agent.total_input_tokens == 100
        assert agent.total_output_tokens == 50

    @patch("agents.templates.claude_thinking_agent.Anthropic")
    def test_call_claude_with_thinking_json_parse_error(
        self, mock_anthropic_class, agent
    ):
        """Test Claude API call with malformed JSON."""
        mock_response = Mock()
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_response.content = [
            Mock(type="text", text="I think we should do ACTION2 because...")
        ]

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        agent.client = mock_client

        result = agent.call_claude_with_thinking("test prompt")

        # Should extract ACTION2 from text or fallback to ACTION1
        assert result["action"] in ["ACTION1", "ACTION2"]

    @patch("agents.templates.claude_thinking_agent.Anthropic")
    def test_call_claude_with_thinking_api_error(self, mock_anthropic_class, agent):
        """Test Claude API call failure."""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")
        agent.client = mock_client

        result = agent.call_claude_with_thinking("test prompt")

        assert result["action"] == "RESET"
        assert "API call failed" in result["reasoning"]

    @patch.object(ClaudeThinkingAgent, "call_claude_with_thinking")
    def test_choose_action_first_action(self, mock_call, agent):
        """Test choosing first action (should be RESET)."""
        frame = FrameData(levels_completed=0, state=GameState.NOT_PLAYED, frame=[[[0]]])

        action = agent.choose_action([frame], frame)

        assert action == GameAction.RESET
        assert len(agent.action_history) == 1

    @patch.object(ClaudeThinkingAgent, "call_claude_with_thinking")
    def test_choose_action_normal_flow(self, mock_call, agent):
        """Test choosing action in normal flow."""
        # Set up initial action
        agent.action_history = [{"action": "RESET", "reasoning": "Initial"}]

        mock_call.return_value = {
            "reasoning": "Moving up seems logical",
            "hypothesis": "UP moves the player upward",
            "action": "ACTION1",
            "confidence": "high",
        }

        frame = FrameData(
            levels_completed=0, state=GameState.NOT_FINISHED, frame=[[[0]]]
        )

        action = agent.choose_action([frame], frame)

        assert action == GameAction.ACTION1
        assert len(agent.action_history) == 2
        assert agent.action_history[-1]["action"] == "ACTION1"

    @patch.object(ClaudeThinkingAgent, "call_claude_with_thinking")
    def test_choose_action_full_reset(self, mock_call, agent):
        """Test choosing action on full reset."""
        frame = FrameData(
            levels_completed=0,
            state=GameState.NOT_FINISHED,
            full_reset=True,
            frame=[[[0]]],
        )

        action = agent.choose_action([frame], frame)

        assert action == GameAction.RESET
        # History should be cleared
        assert agent.action_history == []

    @patch.object(ClaudeThinkingAgent, "call_claude_with_thinking")
    def test_choose_action_tracks_stuck(self, mock_call, agent):
        """Test that stuck counter increments when no progress."""
        agent.action_history = [{"action": "RESET"}]
        agent.last_levels_completed = 0

        mock_call.return_value = {
            "reasoning": "test",
            "hypothesis": "test",
            "action": "ACTION1",
            "confidence": "high",
        }

        frame = FrameData(
            levels_completed=0, state=GameState.NOT_FINISHED, frame=[[[0]]]
        )

        agent.choose_action([frame], frame)
        assert agent.stuck_counter == 1

        agent.choose_action([frame], frame)
        assert agent.stuck_counter == 2

    @patch.object(ClaudeThinkingAgent, "call_claude_with_thinking")
    def test_choose_action_resets_stuck_on_progress(self, mock_call, agent):
        """Test that stuck counter resets on progress."""
        agent.action_history = [{"action": "RESET"}]
        agent.last_levels_completed = 0
        agent.stuck_counter = 5

        mock_call.return_value = {
            "reasoning": "test",
            "hypothesis": "test",
            "action": "ACTION1",
            "confidence": "high",
        }

        frame = FrameData(
            levels_completed=1, state=GameState.NOT_FINISHED, frame=[[[0]]]
        )

        agent.choose_action([frame], frame)
        assert agent.stuck_counter == 0
        assert agent.last_levels_completed == 1
