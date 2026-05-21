"""Unit tests for the OPENCLAW_MODEL env var override.

The override is intended to let callers compare provider models without
editing ~/.openclaw/openclaw.json. It must be forwarded as the
`x-openclaw-model` header on every chat completion request, and must
collapse to "no header" when the env var is unset or empty (an empty
header value would be invalid).
"""

from unittest.mock import MagicMock

import pytest
from arcengine import FrameData, GameState

from agents.templates.openclaw_agent import openclaw_agent as oc_module
from agents.templates.openclaw_agent.openclaw_agent import OpenClaw


def _make_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock(total_tokens=0)
    return resp


def _make_agent(monkeypatch: pytest.MonkeyPatch) -> tuple[OpenClaw, MagicMock]:
    """Instantiate OpenClaw with the OpenAI client patched out."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_response(
        '{"action":"ACTION1"}'
    )
    monkeypatch.setattr(oc_module, "OpenAIClient", lambda **kw: mock_client)

    agent = OpenClaw(
        card_id="card-abc",
        game_id="ls20-test",
        agent_name="openclaw-test",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=MagicMock(),
    )
    return agent, mock_client


def _frame() -> FrameData:
    return FrameData(
        game_id="ls20-test",
        state=GameState.NOT_FINISHED,
        available_actions=[1, 2, 3, 4, 5],
        frame=[[[0] * 4 for _ in range(4)]],
    )


@pytest.mark.unit
class TestModelOverride:
    def test_header_persists_across_turns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # __init__ reads the env var once and caches it on self; if anyone
        # moves the read into choose_action this still passes only because
        # the env is stable, so we also assert call_count to keep this honest.
        monkeypatch.setenv("OPENCLAW_MODEL", "openai/gpt-5")
        agent, client = _make_agent(monkeypatch)

        for _ in range(3):
            agent.choose_action([], _frame())

        assert client.chat.completions.create.call_count == 3
        for call in client.chat.completions.create.call_args_list:
            assert call.kwargs["extra_headers"] == {"x-openclaw-model": "openai/gpt-5"}

    def test_no_header_when_env_var_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENCLAW_MODEL", raising=False)
        agent, client = _make_agent(monkeypatch)

        agent.choose_action([], _frame())

        # `None` (not an empty dict) so the openai SDK omits the header
        # entirely and the gateway falls back to its configured primary.
        assert client.chat.completions.create.call_args.kwargs["extra_headers"] is None

    def test_empty_env_var_treated_as_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # OPENCLAW_MODEL="" must NOT send an empty x-openclaw-model header
        # (which would be invalid). The `or None` in __init__ collapses
        # empty strings to None; dropping that one character breaks this.
        monkeypatch.setenv("OPENCLAW_MODEL", "")
        agent, client = _make_agent(monkeypatch)

        agent.choose_action([], _frame())

        assert agent._model_override is None
        assert client.chat.completions.create.call_args.kwargs["extra_headers"] is None
