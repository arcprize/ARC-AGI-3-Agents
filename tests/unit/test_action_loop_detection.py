"""Tests for action loop detection logic."""

from unittest.mock import MagicMock

from agents.templates.claude_agent.claude_agents import ClaudeCodeAgent


def _make_agent() -> ClaudeCodeAgent:
    """Create a minimal ClaudeCodeAgent with just action_history initialized."""
    agent = object.__new__(ClaudeCodeAgent)
    agent.action_history = []
    return agent


class TestDetectActionLoop:
    """Test the _detect_action_loop method."""

    def test_no_loop_with_few_actions(self):
        """No loop detected when history is shorter than the window."""
        agent = _make_agent()
        agent.action_history = ["action1_move_up", "action2_move_down"] * 5  # 10 actions
        assert agent._detect_action_loop(window=15) is None

    def test_no_loop_with_varied_actions(self):
        """No loop detected when actions are varied."""
        agent = _make_agent()
        agent.action_history = [
            "action1_move_up",
            "action2_move_down",
            "action3_move_left",
            "action4_move_right",
            "action5_interact",
            "action1_move_up",
            "action3_move_left",
            "action2_move_down",
            "action5_interact",
            "action4_move_right",
            "action1_move_up",
            "action2_move_down",
            "action3_move_left",
            "action5_interact",
            "action4_move_right",
        ]
        assert agent._detect_action_loop(window=15) is None

    def test_detects_single_action_loop(self):
        """Detects when the same action is repeated N times."""
        agent = _make_agent()
        agent.action_history = ["action1_move_up"] * 15
        result = agent._detect_action_loop(window=15)
        assert result is not None
        assert "stuck in a loop" in result
        assert "action1_move_up" in result

    def test_detects_two_action_loop(self):
        """Detects alternating up/down pattern."""
        agent = _make_agent()
        agent.action_history = ["action1_move_up", "action2_move_down"] * 10  # 20 actions
        result = agent._detect_action_loop(window=15)
        assert result is not None
        assert "stuck in a loop" in result
        assert "action1_move_up" in result
        assert "action2_move_down" in result

    def test_detects_three_action_loop(self):
        """Detects a repeating 3-action cycle."""
        agent = _make_agent()
        agent.action_history = [
            "action1_move_up",
            "action2_move_down",
            "action3_move_left",
        ] * 5  # 15 actions
        result = agent._detect_action_loop(window=15)
        assert result is not None
        assert "stuck in a loop" in result

    def test_no_false_positive_for_four_action_cycle(self):
        """A 4-action cycle should NOT be detected (only checks patterns up to length 3)."""
        agent = _make_agent()
        agent.action_history = [
            "action1_move_up",
            "action2_move_down",
            "action3_move_left",
            "action4_move_right",
        ] * 4  # 16 actions
        result = agent._detect_action_loop(window=15)
        # 15 actions from a 4-cycle won't match patterns of length 1, 2, or 3
        assert result is None

    def test_loop_only_checks_recent_window(self):
        """Loop detection only considers the last `window` actions."""
        agent = _make_agent()
        # Varied history followed by a loop
        agent.action_history = (
            ["action3_move_left", "action4_move_right", "action5_interact"]
            + ["action1_move_up", "action2_move_down"] * 10
        )
        result = agent._detect_action_loop(window=15)
        assert result is not None

    def test_loop_not_triggered_if_broken_recently(self):
        """If the loop is broken in the last window, no detection."""
        agent = _make_agent()
        agent.action_history = (
            ["action1_move_up", "action2_move_down"] * 7
            + ["action5_interact"]  # breaks the pattern
        )
        result = agent._detect_action_loop(window=15)
        assert result is None

    def test_default_window_is_15(self):
        """Default window parameter is 15."""
        agent = _make_agent()
        agent.action_history = ["action1_move_up"] * 14
        assert agent._detect_action_loop() is None
        agent.action_history.append("action1_move_up")
        assert agent._detect_action_loop() is not None

    def test_empty_history(self):
        """Empty history returns None."""
        agent = _make_agent()
        assert agent._detect_action_loop() is None

    def test_warning_mentions_action_count(self):
        """Warning message includes the window size."""
        agent = _make_agent()
        agent.action_history = ["action1_move_up"] * 20
        result = agent._detect_action_loop(window=20)
        assert "last 20 actions" in result

    def test_click_different_coordinates_no_loop(self):
        """Clicking different x/y coordinates should NOT trigger a loop."""
        agent = _make_agent()
        # Simulate action6_click with different coordinates appended
        agent.action_history = [
            f"action6_click (x={i}, y={i})" for i in range(15)
        ]
        assert agent._detect_action_loop(window=15) is None

    def test_click_same_coordinates_triggers_loop(self):
        """Clicking the same spot repeatedly SHOULD trigger a loop."""
        agent = _make_agent()
        agent.action_history = ["action6_click (x=5, y=10)"] * 15
        result = agent._detect_action_loop(window=15)
        assert result is not None
        assert "stuck in a loop" in result
