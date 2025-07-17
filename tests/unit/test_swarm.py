from unittest.mock import Mock, patch

import pytest
import requests

from agents.structs import Card, GameState, Scorecard
from agents.swarm import Swarm
from agents.templates.random_agent import Random


@pytest.mark.unit
class TestSwarmInitialization:
    def test_swarm_init(self):
        with patch.dict("os.environ", {"ARC_API_KEY": "test-api-key"}):
            swarm = Swarm(
                agent="random", ROOT_URL="https://example.com", games=["game1", "game2"]
            )

            assert swarm.agent_name == "random"
            assert swarm.ROOT_URL == "https://example.com"
            assert swarm.GAMES == ["game1", "game2"]
            assert swarm.agent_class == Random
            assert len(swarm.threads) == 0
            assert len(swarm.agents) == 0

            assert swarm.headers["X-API-Key"] == "test-api-key"
            assert swarm.headers["Accept"] == "application/json"
            assert isinstance(swarm._session, requests.Session)
            assert swarm._session.headers["Accept"] == "application/json"


@pytest.mark.unit
class TestSwarmScorecard:
    @patch("agents.swarm.requests.Session.post")
    def test_open_scorecard(self, mock_post):
        mock_response = Mock()
        mock_response.json.return_value = {"card_id": "test-card-123"}
        mock_post.return_value = mock_response

        swarm = Swarm(agent="random", ROOT_URL="https://example.com", games=["game1"])

        card_id = swarm.open_scorecard()
        assert card_id == "test-card-123"

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "/api/scorecard/open" in call_args[0][0]

        json_data = call_args[1]["json"]
        tags = json_data["tags"]
        assert tags == []

        mock_post.reset_mock()
        mock_response.json.return_value = {
            "error": "API Error",
            "card_id": "error-card",
        }

        with patch("agents.swarm.logger") as mock_logger:
            card_id = swarm.open_scorecard()
            assert card_id == "error-card"
            mock_logger.warning.assert_called_once()

    @patch("agents.swarm.requests.Session.post")
    def test_close_scorecard(self, mock_post):
        card = Card(
            game_id="test-game",
            total_plays=2,
            scores=[10, 20],
            states=[GameState.GAME_OVER, GameState.WIN],
            actions=[50, 60],
        )

        mock_response = Mock()
        mock_response.json.return_value = {
            "card_id": "test-card-123",
            "cards": {"test-game": card.model_dump()},
        }
        mock_post.return_value = mock_response

        swarm = Swarm(agent="random", ROOT_URL="https://example.com", games=["game1"])

        scorecard = swarm.close_scorecard("test-card-123")
        assert isinstance(scorecard, Scorecard)
        assert scorecard.card_id == "test-card-123"
        assert swarm.card_id is None

        mock_post.reset_mock()
        mock_response.json.return_value = {"error": "Close error"}

        with patch("agents.swarm.logger") as mock_logger:
            scorecard = swarm.close_scorecard("test-card-123")
            mock_logger.warning.assert_called_once()


@pytest.mark.unit
class TestSwarmAgentManagement:
    @patch("agents.swarm.Swarm.open_scorecard")
    @patch("agents.swarm.Swarm.close_scorecard")
    @patch("agents.swarm.Thread")
    def test_agent_threading(self, mock_thread, mock_close, mock_open):
        mock_open.return_value = "test-card-123"
        mock_close.return_value = Scorecard()

        mock_thread_instances = [Mock() for _ in range(3)]
        mock_thread.side_effect = mock_thread_instances

        swarm = Swarm(
            agent="random",
            ROOT_URL="https://example.com",
            games=["game1", "game2", "game3"],
        )

        assert swarm.agent_name == "random"
        assert swarm.agent_class == Random
        assert swarm.GAMES == ["game1", "game2", "game3"]

        with patch.object(Random, "main") as mock_agent_main:
            mock_agent_main.return_value = None

            swarm.main()

            assert mock_thread.call_count == 3
            for mock_thread_instance in mock_thread_instances:
                mock_thread_instance.start.assert_called_once()
                mock_thread_instance.join.assert_called_once()


@pytest.mark.unit
class TestSwarmCleanup:
    def test_cleanup(self):
        swarm = Swarm(
            agent="random", ROOT_URL="https://example.com", games=["game1", "game2"]
        )

        mock_agent1 = Mock()
        mock_agent2 = Mock()
        swarm.agents = [mock_agent1, mock_agent2]

        mock_session = Mock()
        swarm._session = mock_session

        scorecard = Scorecard()
        swarm.cleanup(scorecard)

        mock_agent1.cleanup.assert_called_once_with(scorecard)
        mock_agent2.cleanup.assert_called_once_with(scorecard)

        mock_session.close.assert_called_once()

        mock_agent = Mock()
        swarm.agents = [mock_agent]

        swarm.cleanup()
        mock_agent.cleanup.assert_called_once_with(None)

        delattr(swarm, "_session")
        swarm.cleanup()


@pytest.mark.unit
class TestSwarmTags:
    @patch("agents.swarm.requests.Session.post")
    def test_open_scorecard_with_custom_tags(self, mock_post):
        """Test that custom tags are sent when opening a scorecard"""
        mock_response = Mock()
        mock_response.json.return_value = {"card_id": "test-card-123"}
        mock_post.return_value = mock_response

        custom_tags = ["experiment1", "version2", "test"]
        default_tags = ["agent", "random"]
        combined_tags = default_tags + custom_tags

        swarm = Swarm(
            agent="random",
            ROOT_URL="https://example.com",
            games=["game1"],
            tags=combined_tags,
        )

        card_id = swarm.open_scorecard()
        assert card_id == "test-card-123"

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        json_data = call_args[1]["json"]

        assert json_data["tags"] == combined_tags

    @patch("agents.swarm.requests.Session.post")
    def test_open_scorecard_with_empty_tags(self, mock_post):
        """Test that default tags are sent when no custom tags are provided"""
        mock_response = Mock()
        mock_response.json.return_value = {"card_id": "test-card-123"}
        mock_post.return_value = mock_response

        default_tags = ["agent", "random"]

        swarm = Swarm(
            agent="random",
            ROOT_URL="https://example.com",
            games=["game1"],
            tags=default_tags,
        )

        card_id = swarm.open_scorecard()
        assert card_id == "test-card-123"

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        json_data = call_args[1]["json"]

        assert json_data["tags"] == default_tags
        assert json_data["tags"] == ["agent", "random"]

    @patch("agents.swarm.requests.Session.post")
    def test_open_scorecard_with_default_and_custom_tags(self, mock_post):
        """Test that tags include both defaults and custom tags when set from main.py"""
        mock_response = Mock()
        mock_response.json.return_value = {"card_id": "test-card-123"}
        mock_post.return_value = mock_response

        default_tags = ["agent", "random"]
        custom_tags = ["experiment1", "version2"]
        combined_tags = default_tags + custom_tags

        swarm = Swarm(
            agent="random",
            ROOT_URL="https://example.com",
            games=["game1"],
            tags=combined_tags,
        )

        card_id = swarm.open_scorecard()
        assert card_id == "test-card-123"

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        json_data = call_args[1]["json"]
        assert json_data["tags"] == combined_tags
