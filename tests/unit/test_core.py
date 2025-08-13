import pytest

from agents.structs import (
    ActionInput,
    Card,
    FrameData,
    GameAction,
    GameState,
    Scorecard,
)
from agents.templates.langgraph_random_agent import LangGraphRandom
from agents.templates.random_agent import Random


@pytest.mark.unit
class TestGameActionCore:
    @pytest.mark.parametrize(
        "action,data,expected",
        [
            (GameAction.ACTION1, {"game_id": "test"}, {"game_id": "test"}),
            (
                GameAction.ACTION6,
                {"game_id": "test", "x": 32, "y": 45},
                {"x": 32, "y": 45, "game_id": "test"},
            ),
        ],
    )
    def test_action_init(self, action, data, expected):
        action.set_data(data)

        assert action.action_data.game_id == expected["game_id"]
        if "x" in expected:
            assert action.action_data.x == expected["x"]
            assert action.action_data.y == expected["y"]

    @pytest.mark.parametrize(
        "action,invalid_data",
        [
            (GameAction.ACTION6, {"game_id": "test", "x": -1, "y": 0}),
            (GameAction.ACTION6, {"game_id": "test", "x": 0, "y": 64}),
            (GameAction.ACTION6, {"x": "not_a_number", "y": 10}),
        ],
    )
    def test_coordinate_validation(self, action, invalid_data):
        with pytest.raises(Exception):
            action.set_data(invalid_data)

    @pytest.mark.parametrize(
        "action_id,expected",
        [
            (0, GameAction.RESET),
            (6, GameAction.ACTION6),
        ],
    )
    def test_action_from_id(self, action_id, expected):
        action = GameAction.from_id(action_id)
        assert action == expected

        with pytest.raises(ValueError):
            GameAction.from_id(999)

    @pytest.mark.parametrize(
        "action_name,expected",
        [
            ("RESET", GameAction.RESET),
            ("action6", GameAction.ACTION6),
        ],
    )
    def test_action_from_name(self, action_name, expected):
        action = GameAction.from_name(action_name)
        assert action == expected

        with pytest.raises(ValueError):
            GameAction.from_name("INVALID_ACTION")

    def test_action_classification(self):
        simple_actions = GameAction.all_simple()
        complex_actions = GameAction.all_complex()

        assert GameAction.RESET in simple_actions
        assert GameAction.ACTION1 in simple_actions
        assert GameAction.ACTION6 in complex_actions

        assert GameAction.RESET.is_simple()
        assert not GameAction.RESET.is_complex()
        assert GameAction.ACTION6.is_complex()
        assert not GameAction.ACTION6.is_simple()


@pytest.mark.unit
class TestActionInput:
    def test_action_input_init(self):
        action_input = ActionInput()
        assert action_input.id == GameAction.RESET
        assert action_input.data == {}
        assert action_input.reasoning is None

        action_input = ActionInput(
            id=GameAction.ACTION6,
            data={"game_id": "test", "x": 10, "y": 20},
            reasoning={"model": "test", "tokens": 50},
        )

        assert action_input.id == GameAction.ACTION6
        assert action_input.data["x"] == 10
        assert action_input.reasoning["tokens"] == 50

    def test_reasoning_json_validation(self):
        action_input = ActionInput(reasoning={"key": "value", "number": 42})
        assert action_input.reasoning["key"] == "value"

        with pytest.raises(Exception):
            ActionInput(reasoning=lambda x: x)  # Functions are not JSON serializable


@pytest.mark.unit
class TestCard:
    def test_card_init(self):
        card = Card(game_id="test-game")
        assert card.game_id == "test-game"
        assert card.total_plays == 0
        assert not card.started
        assert card.score is None
        assert card.high_score == 0
        assert card.idx == -1

        card = Card(
            game_id="test-game",
            total_plays=2,
            scores=[10, 15],
            states=[GameState.GAME_OVER, GameState.WIN],
            actions=[50, 30],
            resets=[1, 0],
        )

        assert card.started
        assert card.score == 15
        assert card.high_score == 15
        assert card.state == GameState.WIN
        assert card.action_count == 30
        assert card.total_actions == 80
        assert card.idx == 1


@pytest.mark.unit
class TestScorecard:
    def test_scorecard_init(self):
        scorecard = Scorecard(card_id="test-card", api_key="test-key")
        assert scorecard.card_id == "test-card"

        card1 = Card(
            game_id="game1",
            total_plays=2,
            scores=[10, 20],
            states=[GameState.GAME_OVER, GameState.WIN],
        )
        card2 = Card(
            game_id="game2", total_plays=1, scores=[15], states=[GameState.GAME_OVER]
        )

        scorecard.cards = {"game1": card1, "game2": card2}

        assert scorecard.won == 1
        assert scorecard.played == 2
        assert scorecard.total_actions == 0

        all_cards = scorecard.get()
        assert len(all_cards) == 2
        assert "game1" in all_cards
        assert "game2" in all_cards

        specific_card = scorecard.get("game1")
        assert "game1" in specific_card

        json_data = scorecard.get_json_for("game1")
        assert "won" in json_data
        assert "played" in json_data
        assert "cards" in json_data
        assert json_data["cards"]["game1"]["game_id"] == "game1"


@pytest.mark.unit
class TestRandomAgent:
    def test_agent_init(self):
        agent = Random(
            card_id="test-card",
            game_id="test-game",
            agent_name="test-agent",
            ROOT_URL="https://example.com",
            record=False,
        )

        assert agent.game_id == "test-game"
        assert agent.card_id == "test-card"
        assert agent.MAX_ACTIONS == 80
        assert agent.action_counter == 0

        name = agent.name
        assert "test-game" in name
        assert "random" in name
        assert "80" in name

    def test_agent_action_logic(self, sample_frame):
        agent = Random(
            card_id="test-card",
            game_id="test-game",
            agent_name="test-agent",
            ROOT_URL="https://example.com",
            record=False,
        )

        sample_frame.state = GameState.NOT_PLAYED
        action = agent.choose_action([sample_frame], sample_frame)
        assert action == GameAction.RESET

        sample_frame.state = GameState.NOT_FINISHED
        action = agent.choose_action([sample_frame], sample_frame)
        assert action != GameAction.RESET
        assert isinstance(action, GameAction)

        sample_frame.state = GameState.WIN
        assert agent.is_done([sample_frame], sample_frame) is True

        sample_frame.state = GameState.NOT_FINISHED
        assert agent.is_done([sample_frame], sample_frame) is False


@pytest.mark.unit
class TestLangGraphRandomAgent:
    def test_agent_init(self):
        agent = LangGraphRandom(
            card_id="test-card",
            game_id="test-game",
            agent_name="test-agent",
            ROOT_URL="https://example.com",
            record=False,
        )

        assert agent.game_id == "test-game"
        assert agent.card_id == "test-card"
        assert agent.MAX_ACTIONS == 80
        assert agent.action_counter == 0

        # Test that workflow is properly initialized
        assert agent.workflow is not None

        name = agent.name
        assert "test-game" in name
        assert "langgraphrandom" in name.lower()
        assert "80" in name

    def test_agent_action_logic(self, sample_frame):
        agent = LangGraphRandom(
            card_id="test-card",
            game_id="test-game",
            agent_name="test-agent",
            ROOT_URL="https://example.com",
            record=False,
        )

        # NOT_PLAYED state -> RESET
        sample_frame.state = GameState.NOT_PLAYED
        action = agent.choose_action([sample_frame], sample_frame)
        assert action == GameAction.RESET

        # GAME_OVER state -> RESET
        sample_frame.state = GameState.GAME_OVER
        action = agent.choose_action([sample_frame], sample_frame)
        assert action == GameAction.RESET

        # active game state -> random action (not RESET)
        sample_frame.state = GameState.NOT_FINISHED
        action = agent.choose_action([sample_frame], sample_frame)
        assert action != GameAction.RESET
        assert isinstance(action, GameAction)

        sample_frame.state = GameState.WIN
        assert agent.is_done([sample_frame], sample_frame) is True

        sample_frame.state = GameState.NOT_FINISHED
        assert agent.is_done([sample_frame], sample_frame) is False


@pytest.mark.unit
class TestFrameData:
    def test_frame_init(self):
        frame = FrameData(
            game_id="test",
            frame=[[[1, 2], [3, 4]]],
            state=GameState.NOT_FINISHED,
            score=10,
        )

        assert frame.game_id == "test"
        assert frame.score == 10
        assert frame.state == GameState.NOT_FINISHED
        assert not frame.is_empty()

        frame = FrameData()
        assert frame.game_id == ""
        assert frame.frame == []
        assert frame.state == GameState.NOT_PLAYED
        assert frame.score == 0
        assert frame.is_empty()
        assert frame.guid is None
        assert frame.full_reset is False

    @pytest.mark.parametrize(
        "score,should_pass",
        [
            (0, True),
            (254, True),
            (-1, False),
            (255, False),
        ],
    )
    def test_score_validation(self, score, should_pass):
        if should_pass:
            frame = FrameData(score=score)
            assert frame.score == score
        else:
            with pytest.raises(Exception):
                FrameData(score=score)

    def test_frame_2(self):
        action_input = ActionInput(
            id=GameAction.ACTION1, data={"game_id": "test"}, reasoning={"model": "test"}
        )

        frame = FrameData(
            game_id="test",
            action_input=action_input,
            guid="test-guid-123",
            full_reset=True,
        )

        assert frame.action_input.id == GameAction.ACTION1
        assert frame.action_input.data["game_id"] == "test"
        assert frame.guid == "test-guid-123"
        assert frame.full_reset is True

        json_data = frame.model_dump()
        assert json_data["game_id"] == "test"
        assert json_data["guid"] == "test-guid-123"
        assert json_data["full_reset"] is True

    def test_frame_3(self):
        complex_frame = [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
        ]

        frame = FrameData(game_id="complex-test", frame=complex_frame, score=50)

        assert frame.frame == complex_frame
        assert not frame.is_empty()
        assert len(frame.frame) == 2
        assert len(frame.frame[0]) == 3
        assert len(frame.frame[0][0]) == 3
