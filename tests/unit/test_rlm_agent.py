"""
Unit tests for the RLM Agent (real REPL-loop architecture).

Grid utilities are tested directly (no mocking).
Agent logic mocks only the OpenAI chat API to avoid network calls.
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock

from arcengine import FrameData, GameAction, GameState

from agents.templates.rlm_agent import (
    color_name,
    summarize_grid,
    diff_summary,
    find_objects,
    grid_region,
    GameREPL,
    COLOR_MAP,
    VALID_ACTIONS,
    RLMAgent,
)


# ── Grid utility tests (real code, no mocking) ─────────────────────


@pytest.mark.unit
class TestGridUtilities:

    def test_color_name_known(self):
        assert color_name(0) == "black"
        assert color_name(1) == "blue"
        assert color_name(2) == "red"

    def test_color_name_unknown(self):
        assert color_name(99) == "color_99"

    def test_summarize_empty(self):
        assert "Empty" in summarize_grid([])
        assert "Empty" in summarize_grid([[]])

    def test_summarize_small(self):
        s = summarize_grid([[0, 1], [2, 0]])
        assert "2x2" in s
        assert "non-zero=2" in s

    def test_diff_no_prev(self):
        assert "First frame" in diff_summary(None, [[1]])

    def test_diff_no_changes(self):
        g = [[0, 1], [2, 0]]
        assert "No changes" in diff_summary(g, g)

    def test_diff_with_changes(self):
        s = diff_summary([[0, 0], [0, 0]], [[1, 0], [0, 2]])
        assert "2 pixels changed" in s

    def test_find_objects_empty(self):
        assert find_objects([[0, 0], [0, 0]]) == []

    def test_find_objects_single(self):
        objs = find_objects([[0, 0, 0], [0, 1, 1], [0, 1, 0]])
        assert len(objs) == 1
        assert objs[0]["color"] == "blue"
        assert objs[0]["size"] == 3

    def test_find_objects_multi(self):
        objs = find_objects([[1, 0, 2], [1, 0, 2], [0, 0, 0]])
        assert len(objs) == 2

    def test_grid_region(self):
        g = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assert grid_region(g, 1, 1, 2, 2) == [[5, 6], [8, 9]]

    def test_valid_actions(self):
        assert "RESET" in VALID_ACTIONS
        assert "ACTION7" in VALID_ACTIONS


# ── GameREPL tests (real exec sandbox) ──────────────────────────────


@pytest.mark.unit
class TestGameREPL:

    def _repl(self) -> GameREPL:
        grid = [[0, 1], [2, 0]]
        return GameREPL(grid, None, [], "test hypothesis", 1)

    def test_exec_print(self):
        r = self._repl()
        out = r.execute("print('hello')")
        assert "hello" in out

    def test_grid_accessible(self):
        r = self._repl()
        out = r.execute("print(grid)")
        assert "[[0, 1], [2, 0]]" in out

    def test_helpers_accessible(self):
        r = self._repl()
        out = r.execute("print(summarize_grid(grid))")
        assert "2x2" in out

    def test_find_objects_in_repl(self):
        r = self._repl()
        out = r.execute("print(find_objects(grid))")
        assert "blue" in out or "red" in out

    def test_error_handling(self):
        r = self._repl()
        out = r.execute("1/0")
        assert "ZeroDivisionError" in out

    def test_state_persists(self):
        r = self._repl()
        r.execute("x = 42")
        out = r.execute("print(x)")
        assert "42" in out

    def test_get_var(self):
        r = self._repl()
        r.execute("result = {'action': 'ACTION1'}")
        assert r.get_var("result") == {"action": "ACTION1"}


# ── Agent tests (mock only the OpenAI API) ──────────────────────────


def _frame(state=GameState.NOT_FINISHED, levels=0, gid="test") -> FrameData:
    return FrameData(
        game_id=gid,
        frame=[[[0, 0, 1, 0], [0, 2, 2, 0], [0, 0, 0, 3], [0, 0, 0, 0]]],
        state=state,
        levels_completed=levels,
    )


def _mock_chat_response(content: str) -> Mock:
    """Build a mock that looks like openai ChatCompletion response."""
    msg = Mock()
    msg.content = content
    choice = Mock()
    choice.message = msg
    resp = Mock()
    resp.choices = [choice]
    return resp


def _make_agent(chat_responses: list[str] | None = None) -> RLMAgent:
    """Create an RLMAgent with a mocked OpenAI client."""
    if chat_responses is None:
        chat_responses = [
            'FINAL({"action": "ACTION1", "reasoning": "exploring", '
            '"hypothesis": "testing movement", "observation": "grid has objects"})'
        ]

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        _mock_chat_response(c) for c in chat_responses
    ]

    agent = RLMAgent(
        card_id="t", game_id="t", agent_name="t",
        ROOT_URL="http://t", record=False, arc_env=Mock(),
    )
    agent._client = mock_client
    return agent


@pytest.mark.unit
class TestRLMAgentInit:

    def test_defaults(self):
        a = _make_agent()
        assert a.MODEL == "google/gemini-2.5-flash"
        assert a.turn_number == 0
        assert a.memory == []

    def test_max_actions(self):
        assert _make_agent().MAX_ACTIONS == 80


@pytest.mark.unit
class TestRLMAgentActions:

    def test_returns_valid_action(self):
        a = _make_agent()
        action = a.choose_action(_frame())
        assert action.name in VALID_ACTIONS

    def test_turn_increments(self):
        # Need two FINAL responses for two calls
        a = _make_agent([
            'FINAL({"action": "ACTION1", "reasoning": "r"})',
            'FINAL({"action": "ACTION2", "reasoning": "r"})',
        ])
        a.choose_action(_frame())
        assert a.turn_number == 1
        a.choose_action(_frame())
        assert a.turn_number == 2

    def test_memory_recorded(self):
        a = _make_agent()
        a.choose_action(_frame())
        assert len(a.memory) == 1

    def test_reset_handling(self):
        a = _make_agent()
        a.memory.append({"turn": 1})
        a.turn_number = 5
        action = a.choose_action(_frame(state=GameState.NOT_PLAYED))
        assert action == GameAction.RESET
        assert a.memory == []
        assert a.turn_number == 0

    def test_fallback_action(self):
        assert _make_agent()._fallback() == GameAction.ACTION5

    def test_exploratory_action(self):
        a = _make_agent()
        action = a._exploratory_action()
        assert action in [
            GameAction.ACTION1, GameAction.ACTION2,
            GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5,
        ]


@pytest.mark.unit
class TestRLMAgentREPLLoop:
    """Test the iterative REPL loop — the core RLM mechanism."""

    def test_immediate_final(self):
        """LLM emits FINAL on first turn → one LLM call."""
        a = _make_agent(['FINAL({"action": "ACTION3", "reasoning": "go left"})'])
        action = a.choose_action(_frame())
        assert action == GameAction.ACTION3

    def test_code_then_final(self):
        """LLM writes code first, then FINAL on second turn."""
        a = _make_agent([
            '```python\nprint(summarize_grid(grid))\n```',
            'FINAL({"action": "ACTION2", "reasoning": "grid analysis done"})',
        ])
        action = a.choose_action(_frame())
        assert action == GameAction.ACTION2
        # Should have made 2 LLM calls
        assert a._total_llm_calls == 2

    def test_multiple_repl_rounds(self):
        """LLM iterates through multiple code cells."""
        a = _make_agent([
            '```python\nprint(summarize_grid(grid))\n```',
            '```python\nobjs = find_objects(grid)\nprint(len(objs))\n```',
            '```python\nprint(diff_summary(prev_grid, grid))\n```',
            'FINAL({"action": "ACTION4", "reasoning": "found pattern"})',
        ])
        action = a.choose_action(_frame())
        assert action == GameAction.ACTION4
        assert a._total_llm_calls == 4

    def test_bare_action_in_final(self):
        """FINAL(ACTION1) without JSON."""
        a = _make_agent(['FINAL(ACTION1)'])
        action = a.choose_action(_frame())
        assert action == GameAction.ACTION1


@pytest.mark.unit
class TestRLMAgentParsing:

    def test_extract_final_json(self):
        d = RLMAgent._extract_final('FINAL({"action": "ACTION2", "reasoning": "test"})')
        assert d is not None
        assert d["action"] == "ACTION2"

    def test_extract_final_bare(self):
        d = RLMAgent._extract_final("FINAL(ACTION3)")
        assert d is not None
        assert d["action"] == "ACTION3"

    def test_extract_final_none(self):
        assert RLMAgent._extract_final("no final here") is None

    def test_extract_code(self):
        code = RLMAgent._extract_code("```python\nprint(1)\n```")
        assert code == "print(1)"

    def test_extract_code_no_block(self):
        assert RLMAgent._extract_code("just text") is None


@pytest.mark.unit
class TestRLMAgentMemory:

    def test_memory_cap(self):
        a = _make_agent()
        for i in range(60):
            a._record(f"ACTION{i % 5 + 1}", {"observation": f"obs {i}"})
        assert len(a.memory) == 50
        assert a.memory[-1]["observation"] == "obs 59"

    def test_stuck_detection(self):
        a = _make_agent()
        f = _frame()
        for i in range(6):
            stuck = a._is_stuck(f)
            if i >= 5:
                assert stuck
            else:
                assert not stuck

    def test_stuck_resets(self):
        a = _make_agent()
        for _ in range(4):
            a._is_stuck(_frame())
        f2 = FrameData(game_id="t", frame=[[[9, 9], [9, 9]]],
                       state=GameState.NOT_FINISHED, levels_completed=0)
        assert not a._is_stuck(f2)


@pytest.mark.unit
class TestRLMAgentGenerality:
    """Verify NO game-specific hardcoded logic."""

    def test_no_find_player(self):
        import agents.templates.rlm_agent as mod
        assert not hasattr(mod, "find_player")

    def test_no_find_door(self):
        import agents.templates.rlm_agent as mod
        assert not hasattr(mod, "find_door")

    def test_no_find_key(self):
        import agents.templates.rlm_agent as mod
        assert not hasattr(mod, "find_key")

    def test_system_prompt_general(self):
        from agents.templates.rlm_agent import SYSTEM_PROMPT
        low = SYSTEM_PROMPT.lower()
        assert "discover" in low
        assert "hypothesis" in low
        assert "door" not in low
        assert "key pattern" not in low

    def test_all_game_ids(self):
        """Agent works across ls20, ft09, vc33."""
        for gid in ("ls20", "ft09", "vc33"):
            a = _make_agent(['FINAL({"action": "ACTION1", "reasoning": "r"})'])
            action = a.choose_action(_frame(gid=gid))
            assert action.name in VALID_ACTIONS


@pytest.mark.unit
class TestNoRlmsDependency:
    """The agent must NOT depend on the external ``rlms`` package."""

    def test_no_rlms_import(self):
        import inspect
        import agents.templates.rlm_agent as mod
        src = inspect.getsource(mod)
        assert "import rlms" not in src
        assert "from rlms" not in src
