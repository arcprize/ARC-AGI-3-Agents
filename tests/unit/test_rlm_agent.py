"""
Unit tests for the general-purpose RLM Agent.

Tests grid utilities directly (no mocking needed) and agent logic
with a minimal mock for the rlms library.
"""

import json
import sys
import pytest
from unittest.mock import Mock, patch

from arcengine import FrameData, GameAction, GameState

# ── General grid utility tests (no mocking, real code) ──────────────

from agents.templates.rlm_agent import (
    color_name,
    summarize_grid,
    diff_summary,
    find_objects,
    grid_region,
    COLOR_MAP,
    VALID_ACTIONS,
)


@pytest.mark.unit
class TestGridUtilities:
    """Test general-purpose grid analysis utilities."""

    def test_color_name_known(self):
        assert color_name(0) == "black"
        assert color_name(1) == "blue"
        assert color_name(2) == "red"
        assert color_name(4) == "yellow"

    def test_color_name_unknown(self):
        assert color_name(99) == "color_99"

    def test_summarize_empty_grid(self):
        assert "Empty" in summarize_grid([])
        assert "Empty" in summarize_grid([[]])

    def test_summarize_small_grid(self):
        grid = [[0, 1], [2, 0]]
        s = summarize_grid(grid)
        assert "2x2" in s
        assert "Non-zero cells: 2" in s

    def test_summarize_large_grid(self):
        grid = [[0] * 64 for _ in range(64)]
        grid[10][10] = 3
        grid[20][20] = 5
        s = summarize_grid(grid)
        assert "64x64" in s
        assert "Non-zero cells: 2" in s

    def test_diff_summary_no_prev(self):
        assert "First frame" in diff_summary(None, [[1]])

    def test_diff_summary_no_changes(self):
        grid = [[0, 1], [2, 0]]
        assert "No changes" in diff_summary(grid, grid)

    def test_diff_summary_with_changes(self):
        prev = [[0, 0], [0, 0]]
        curr = [[1, 0], [0, 2]]
        s = diff_summary(prev, curr)
        assert "2 pixels changed" in s

    def test_find_objects_empty(self):
        assert find_objects([[0, 0], [0, 0]]) == []

    def test_find_objects_single(self):
        grid = [[0, 0, 0], [0, 1, 1], [0, 1, 0]]
        objs = find_objects(grid)
        assert len(objs) == 1
        assert objs[0]["color"] == "blue"
        assert objs[0]["size"] == 3

    def test_find_objects_multiple_colors(self):
        grid = [[1, 0, 2], [1, 0, 2], [0, 0, 0]]
        objs = find_objects(grid)
        assert len(objs) == 2
        colors = {o["color"] for o in objs}
        assert colors == {"blue", "red"}

    def test_grid_region(self):
        grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        region = grid_region(grid, 1, 1, 2, 2)
        assert region == [[5, 6], [8, 9]]

    def test_valid_actions_includes_all(self):
        assert "RESET" in VALID_ACTIONS
        assert "ACTION1" in VALID_ACTIONS
        assert "ACTION7" in VALID_ACTIONS


# ── Agent logic tests (mock rlms only) ──────────────────────────────

# Minimal mock so the agent can be instantiated without the real rlms
class _MockRLM:
    def __init__(self, **kwargs):
        self._response = json.dumps({
            "action": "ACTION1",
            "reasoning": "Exploring the grid",
            "hypothesis": "This game requires movement",
            "observation": "I see colored objects on the grid",
        })

    def completion(self, prompt):
        r = Mock()
        r.response = self._response
        return r


# Patch rlms at module level so imports succeed
sys.modules["rlms"] = Mock()
sys.modules["rlms"].RLM = _MockRLM

from agents.templates.rlm_agent import RLMAgent  # noqa: E402


def _make_agent() -> RLMAgent:
    """Helper to create an RLMAgent with patched rlms."""
    with patch("rlms.RLM", _MockRLM):
        return RLMAgent(
            card_id="test", game_id="test", agent_name="test",
            ROOT_URL="http://test", record=False, arc_env=Mock(),
        )


def _frame(state=GameState.NOT_FINISHED, levels=0, game_id="test") -> FrameData:
    return FrameData(
        game_id=game_id,
        frame=[[
            [0, 0, 1, 0],
            [0, 2, 2, 0],
            [0, 0, 0, 3],
            [0, 0, 0, 0],
        ]],
        state=state,
        levels_completed=levels,
    )


@pytest.mark.unit
class TestRLMAgentInit:

    def test_defaults(self):
        agent = _make_agent()
        assert agent.BACKEND == "openrouter"
        assert agent.MODEL == "google/gemini-2.5-flash"
        assert agent.ENVIRONMENT == "local"
        assert agent.turn_number == 0
        assert agent.memory == []
        assert "explore" in agent.hypothesis.lower() or "unknown" in agent.hypothesis.lower()

    def test_max_actions(self):
        assert _make_agent().MAX_ACTIONS == 80


@pytest.mark.unit
class TestRLMAgentActions:

    def test_choose_action_returns_valid_action(self):
        agent = _make_agent()
        action = agent.choose_action(_frame())
        assert isinstance(action, GameAction)
        assert action.name in VALID_ACTIONS

    def test_turn_increments(self):
        agent = _make_agent()
        agent.choose_action(_frame())
        assert agent.turn_number == 1
        agent.choose_action(_frame())
        assert agent.turn_number == 2

    def test_memory_recorded(self):
        agent = _make_agent()
        agent.choose_action(_frame())
        assert len(agent.memory) == 1

    def test_reset_handling(self):
        agent = _make_agent()
        agent.memory.append({"turn": 1})
        agent.turn_number = 5
        action = agent.choose_action(_frame(state=GameState.NOT_PLAYED))
        assert action == GameAction.RESET
        assert agent.memory == []
        assert agent.turn_number == 0

    def test_exploratory_action(self):
        agent = _make_agent()
        action = agent._exploratory_action()
        assert action in [
            GameAction.ACTION1, GameAction.ACTION2,
            GameAction.ACTION3, GameAction.ACTION4,
            GameAction.ACTION5,
        ]

    def test_fallback_action(self):
        assert _make_agent()._fallback_action() == GameAction.ACTION5


@pytest.mark.unit
class TestRLMAgentParsing:

    def test_parse_json(self):
        agent = _make_agent()
        result = Mock()
        result.response = '{"action": "ACTION2", "reasoning": "test"}'
        action, meta = agent._parse_rlm_result(result, _frame())
        assert action == GameAction.ACTION2

    def test_parse_result_assignment(self):
        agent = _make_agent()
        result = Mock()
        result.response = 'result = {"action": "ACTION3", "reasoning": "left"}'
        action, _ = agent._parse_rlm_result(result, _frame())
        assert action == GameAction.ACTION3

    def test_parse_bare_action(self):
        agent = _make_agent()
        result = Mock()
        result.response = "I think ACTION4 is the best move"
        action, _ = agent._parse_rlm_result(result, _frame())
        assert action == GameAction.ACTION4

    def test_parse_code_block(self):
        agent = _make_agent()
        result = Mock()
        result.response = '```json\n{"action": "ACTION5", "reasoning": "interact"}\n```'
        action, _ = agent._parse_rlm_result(result, _frame())
        assert action == GameAction.ACTION5

    def test_parse_garbage_falls_back(self):
        agent = _make_agent()
        result = Mock()
        result.response = "I have no idea what to do lol"
        action, _ = agent._parse_rlm_result(result, _frame())
        assert action == GameAction.ACTION5  # fallback


@pytest.mark.unit
class TestRLMAgentMemory:

    def test_memory_cap(self):
        agent = _make_agent()
        for i in range(60):
            agent._record_observation(f"ACTION{i % 5 + 1}", f"obs {i}", f"reason {i}")
        assert len(agent.memory) == 50
        assert agent.memory[-1]["observation"] == "obs 59"

    def test_stuck_detection(self):
        agent = _make_agent()
        f = _frame()
        for i in range(6):
            stuck = agent._is_stuck(f)
            if i >= 5:
                assert stuck
            else:
                assert not stuck

    def test_stuck_resets_on_new_grid(self):
        agent = _make_agent()
        f1 = _frame()
        for _ in range(4):
            agent._is_stuck(f1)
        f2 = FrameData(
            game_id="test",
            frame=[[[9, 9], [9, 9]]],
            state=GameState.NOT_FINISHED,
            levels_completed=0,
        )
        assert not agent._is_stuck(f2)


@pytest.mark.unit
class TestRLMAgentGenerality:
    """Verify the agent contains NO game-specific hardcoded logic."""

    def test_no_find_player(self):
        import agents.templates.rlm_agent as mod
        assert not hasattr(mod, "find_player")

    def test_no_find_door(self):
        import agents.templates.rlm_agent as mod
        assert not hasattr(mod, "find_door")

    def test_no_find_key(self):
        import agents.templates.rlm_agent as mod
        assert not hasattr(mod, "find_key")

    def test_system_prompt_is_general(self):
        from agents.templates.rlm_agent import SYSTEM_PROMPT
        prompt_lower = SYSTEM_PROMPT.lower()
        # Should NOT contain game-specific terms
        assert "player" not in prompt_lower or "game" in prompt_lower
        assert "door" not in prompt_lower
        assert "key pattern" not in prompt_lower
        # Should contain general terms
        assert "explore" in prompt_lower or "discover" in prompt_lower
        assert "hypothesis" in prompt_lower

    def test_works_with_different_game_ids(self):
        agent = _make_agent()
        for gid in ["ls20", "ft09", "vc33"]:
            f = _frame(game_id=gid)
            action = agent.choose_action(f)
            assert action.name in VALID_ACTIONS
