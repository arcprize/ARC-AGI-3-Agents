"""Unit tests for the OpenClaw agent's JSON-in-text action parser.

The OpenClaw OpenAI-compat endpoint silently drops the OpenAI `tools` field
for some providers, so this agent has the model reply with one JSON object
on each turn. Parsing that JSON is the most failure-prone bit of the agent
and is exercised here without spinning up a real gateway.
"""

import pytest
from arcengine import GameAction

from agents.templates.openclaw_agent.openclaw_agent import OpenClaw


class _Msg:
    def __init__(self, content: str) -> None:
        self.content = content


def _bare_agent() -> OpenClaw:
    """Construct an OpenClaw instance without running __init__.

    The parser is pure logic and doesn't depend on any agent state, so we
    bypass __init__ (which would try to build an HTTP client and require
    env vars) and call _parse_action directly.
    """
    return object.__new__(OpenClaw)  # type: ignore[return-value]


@pytest.mark.unit
class TestParseAction:
    def test_canonical_name(self) -> None:
        action = _bare_agent()._parse_action(_Msg('{"action":"ACTION1"}'), None)  # type: ignore[arg-type]
        assert action is GameAction.ACTION1

    def test_lowercase_name(self) -> None:
        action = _bare_agent()._parse_action(_Msg('{"action":"action3"}'), None)  # type: ignore[arg-type]
        assert action is GameAction.ACTION3

    def test_reset(self) -> None:
        action = _bare_agent()._parse_action(_Msg('{"action":"RESET"}'), None)  # type: ignore[arg-type]
        assert action is GameAction.RESET

    def test_integer_string_id(self) -> None:
        # Model sometimes emits {"action": "1"} instead of {"action": "ACTION1"}.
        action = _bare_agent()._parse_action(_Msg('{"action":"1"}'), None)  # type: ignore[arg-type]
        assert action is GameAction.ACTION1

    def test_raw_integer_id(self) -> None:
        # Or even {"action": 4}.
        action = _bare_agent()._parse_action(_Msg('{"action":4}'), None)  # type: ignore[arg-type]
        assert action is GameAction.ACTION4

    def test_action6_with_coords(self) -> None:
        action = _bare_agent()._parse_action(
            _Msg('{"action":"ACTION6","x":12,"y":34}'),
            None,  # type: ignore[arg-type]
        )
        assert action is GameAction.ACTION6
        assert action.action_data.x == 12  # type: ignore[union-attr]
        assert action.action_data.y == 34  # type: ignore[union-attr]

    def test_action6_with_string_coords(self) -> None:
        # Models sometimes emit numbers as strings.
        action = _bare_agent()._parse_action(
            _Msg('{"action":"ACTION6","x":"7","y":"8"}'),
            None,  # type: ignore[arg-type]
        )
        assert action is GameAction.ACTION6
        assert action.action_data.x == 7  # type: ignore[union-attr]
        assert action.action_data.y == 8  # type: ignore[union-attr]

    def test_action6_missing_coords_falls_back_to_center(self) -> None:
        action = _bare_agent()._parse_action(_Msg('{"action":"ACTION6"}'), None)  # type: ignore[arg-type]
        assert action is GameAction.ACTION6
        # Coordinates default to the center of the 64x64 grid.
        assert action.action_data.x == 32  # type: ignore[union-attr]
        assert action.action_data.y == 32  # type: ignore[union-attr]

    def test_markdown_fence_wrapper(self) -> None:
        action = _bare_agent()._parse_action(
            _Msg('```json\n{"action":"ACTION2"}\n```'),
            None,  # type: ignore[arg-type]
        )
        assert action is GameAction.ACTION2

    def test_leading_prose(self) -> None:
        # The regex extractor finds the first {...} containing "action".
        action = _bare_agent()._parse_action(
            _Msg('Sure thing! Here is the action: {"action":"ACTION5"}'),
            None,  # type: ignore[arg-type]
        )
        assert action is GameAction.ACTION5

    def test_unknown_action_falls_back(self) -> None:
        action = _bare_agent()._parse_action(_Msg('{"action":"FLY"}'), None)  # type: ignore[arg-type]
        assert action is GameAction.ACTION5

    def test_out_of_range_integer_falls_back(self) -> None:
        # Valid GameAction values are 0..7. 42 is not a member.
        action = _bare_agent()._parse_action(_Msg('{"action":42}'), None)  # type: ignore[arg-type]
        assert action is GameAction.ACTION5

    def test_missing_action_key_falls_back(self) -> None:
        action = _bare_agent()._parse_action(_Msg('{"foo":"bar"}'), None)  # type: ignore[arg-type]
        assert action is GameAction.ACTION5

    def test_unparseable_text_falls_back(self) -> None:
        action = _bare_agent()._parse_action(_Msg("totally not json"), None)  # type: ignore[arg-type]
        assert action is GameAction.ACTION5

    def test_empty_content_falls_back(self) -> None:
        action = _bare_agent()._parse_action(_Msg(""), None)  # type: ignore[arg-type]
        assert action is GameAction.ACTION5


@pytest.mark.unit
class TestActionNames:
    """Cover _action_names: it gets a list of available actions and must
    normalize ints, strings, and GameAction members to canonical names."""

    def test_handles_game_action_members(self) -> None:
        names = _bare_agent()._action_names([GameAction.ACTION1, GameAction.ACTION3])
        assert names == ["ACTION1", "ACTION3"]

    def test_handles_integer_ids(self) -> None:
        # ARC's FrameData.available_actions arrives as a list of ints.
        names = _bare_agent()._action_names([1, 2, 3, 4])
        assert names == ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]

    def test_handles_string_digits(self) -> None:
        names = _bare_agent()._action_names(["1", "6"])
        assert names == ["ACTION1", "ACTION6"]

    def test_handles_unknown_values_gracefully(self) -> None:
        names = _bare_agent()._action_names([99, "garbage"])
        assert names == ["99", "garbage"]

    def test_handles_none(self) -> None:
        names = _bare_agent()._action_names(None)
        assert names == []

    def test_handles_empty(self) -> None:
        names = _bare_agent()._action_names([])
        assert names == []
