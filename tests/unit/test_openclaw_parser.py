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
class TestExtractReasoning:
    """Cover _extract_reasoning: builds the four doc-listed fields with
    consistent shape regardless of what the model returned."""

    def test_happy_path(self) -> None:
        blob = {
            "action": "ACTION1",
            "thought": "Player is below the door; move up.",
            "confidence": 0.78,
            "alternatives_considered": ["ACTION4 to test wall", "ACTION5 to wait"],
        }
        assert OpenClaw._extract_reasoning(blob) == {
            "thought": "Player is below the door; move up.",
            "confidence": 0.78,
            "alternatives_considered": ["ACTION4 to test wall", "ACTION5 to wait"],
            "reasoning_tokens": 0,
        }

    def test_missing_fields_get_defaults(self) -> None:
        assert OpenClaw._extract_reasoning({"action": "ACTION1"}) == {
            "thought": "(no thought provided)",
            "confidence": 0.0,
            "alternatives_considered": [],
            "reasoning_tokens": 0,
        }

    def test_parse_failed_blob_marks_thought(self) -> None:
        # blob=None means the parser couldn't extract any JSON at all.
        out = OpenClaw._extract_reasoning(None)
        assert out["thought"] == "(parse failed)"

    def test_confidence_above_one_clamped(self) -> None:
        out = OpenClaw._extract_reasoning({"action": "ACTION1", "confidence": 1.7})
        assert out["confidence"] == 1.0

    def test_confidence_below_zero_clamped(self) -> None:
        out = OpenClaw._extract_reasoning({"action": "ACTION1", "confidence": -0.3})
        assert out["confidence"] == 0.0

    def test_confidence_non_numeric_falls_back_to_zero(self) -> None:
        out = OpenClaw._extract_reasoning({"action": "ACTION1", "confidence": "high"})
        assert out["confidence"] == 0.0

    def test_alternatives_non_list_falls_back_to_empty(self) -> None:
        out = OpenClaw._extract_reasoning(
            {"action": "ACTION1", "alternatives_considered": "not a list"}
        )
        assert out["alternatives_considered"] == []

    def test_alternatives_truncated_to_five_items(self) -> None:
        out = OpenClaw._extract_reasoning(
            {"action": "ACTION1", "alternatives_considered": list("abcdefgh")}
        )
        assert out["alternatives_considered"] == ["a", "b", "c", "d", "e"]

    def test_alternatives_items_coerced_to_str_and_truncated(self) -> None:
        long = "x" * 500
        out = OpenClaw._extract_reasoning(
            {"action": "ACTION1", "alternatives_considered": [42, long]}
        )
        assert out["alternatives_considered"][0] == "42"
        assert out["alternatives_considered"][1] == "x" * 200

    def test_thought_under_cap_passes_unchanged(self) -> None:
        # Below 16 KB the model's thought passes through verbatim — the
        # reviewer's trace-analysis use case relies on this.
        out = OpenClaw._extract_reasoning({"action": "ACTION1", "thought": "a" * 5000})
        assert out["thought"] == "a" * 5000

    def test_thought_over_cap_truncated_to_fit(self) -> None:
        import json as _json

        out = OpenClaw._extract_reasoning({"action": "ACTION1", "thought": "a" * 20000})
        # The full JSON payload (not just thought) must fit under the cap.
        size = len(_json.dumps(out, separators=(",", ":")).encode("utf-8"))
        assert size <= 16 * 1024
        # Thought is trimmed from the end, structure preserved.
        assert out["thought"].startswith("a")
        assert len(out["thought"]) < 20000


@pytest.mark.unit
class TestParseActionWithReasoningBlob:
    """The regex extractor must handle JSON with nested arrays
    (e.g. alternatives_considered) when buried in prose or fences."""

    def test_blob_with_alternatives_array(self) -> None:
        text = '{"action":"ACTION2","thought":"go down","confidence":0.5,"alternatives_considered":["a","b"]}'
        action = _bare_agent()._parse_action(_Msg(text), None)  # type: ignore[arg-type]
        assert action is GameAction.ACTION2

    def test_blob_with_alternatives_array_inside_prose(self) -> None:
        text = (
            'Here you go: {"action":"ACTION3","alternatives_considered":["x","y","z"]}'
        )
        action = _bare_agent()._parse_action(_Msg(text), None)  # type: ignore[arg-type]
        assert action is GameAction.ACTION3

    def test_markdown_fence_with_array(self) -> None:
        text = '```json\n{"action":"ACTION1","alternatives_considered":["a"]}\n```'
        action = _bare_agent()._parse_action(_Msg(text), None)  # type: ignore[arg-type]
        assert action is GameAction.ACTION1


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
