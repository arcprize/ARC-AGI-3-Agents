"""
Unit tests for RLM Agent functionality.
"""

import json
import pytest
from unittest.mock import Mock, patch

from arcengine import FrameData, GameAction, GameState


# Mock the RLM module since it's not available in test environment
class MockRLM:
    def __init__(self, **kwargs):
        pass
    
    def completion(self, prompt):
        mock_result = Mock()
        mock_result.response = json.dumps({
            "action": "ACTION1",
            "reasoning": "Test reasoning",
            "hypothesis": "Test hypothesis", 
            "observation": "Test observation"
        })
        return mock_result


# Patch the import at module level
import sys
sys.modules['rlms'] = Mock()
sys.modules['rlms'].RLM = MockRLM

# Now import the agent
from agents.templates.rlm_agent import RLMAgent


@pytest.mark.unit
class TestRLMAgent:
    """Test RLM Agent functionality."""

    def test_agent_initialization(self):
        """Test RLM agent can be initialized."""
        with patch('rlms.RLM', MockRLM):
            agent = RLMAgent(
                card_id="test",
                game_id="test",
                agent_name="test",
                ROOT_URL="http://test.com",
                record=False,
                arc_env=Mock()
            )
            
            assert agent.BACKEND == "openrouter"
            assert agent.MODEL == "google/gemini-2.5-flash"
            assert agent.ENVIRONMENT == "local"
            assert agent.hypothesis == "Unknown game. Need to explore and discover the rules."
            assert agent.memory == []
            assert agent.turn_number == 0

    def test_choose_action_with_valid_response(self):
        """Test action selection with valid RLM response."""
        with patch('rlms.RLM', MockRLM):
            agent = RLMAgent(
                card_id="test",
                game_id="test",
                agent_name="test",
                ROOT_URL="http://test.com",
                record=False,
                arc_env=Mock()
            )
            
            sample_frame = FrameData(
                game_id="test",
                frame=[[[1, 2], [3, 4]]],
                state=GameState.NOT_FINISHED,
                levels_completed=0
            )
            
            action = agent.choose_action(sample_frame)
            
            assert action == GameAction.ACTION1
            assert agent.turn_number == 1
            assert len(agent.memory) == 1

    def test_parsing_various_response_formats(self):
        """Test parsing different response formats."""
        with patch('rlms.RLM', MockRLM):
            agent = RLMAgent(
                card_id="test",
                game_id="test",
                agent_name="test",
                ROOT_URL="http://test.com",
                record=False,
                arc_env=Mock()
            )
            
            sample_frame = FrameData(
                game_id="test",
                frame=[[[1, 2], [3, 4]]],
                state=GameState.NOT_FINISHED,
                levels_completed=0
            )
            
            test_cases = [
                ('{"action": "ACTION2", "reasoning": "Test"}', GameAction.ACTION2),
                ('ACTION3', GameAction.ACTION3),
                ('I think ACTION4 is best', GameAction.ACTION4),
                ('result = {"action": "ACTION5", "reasoning": "Test"}', GameAction.ACTION5),
            ]
            
            for response_text, expected_action in test_cases:
                mock_result = Mock()
                mock_result.response = response_text
                
                action, meta = agent._parse_rlm_result(mock_result, sample_frame)
                assert action == expected_action

    def test_fallback_action(self):
        """Test fallback action when parsing fails."""
        with patch('rlms.RLM', MockRLM):
            agent = RLMAgent(
                card_id="test",
                game_id="test",
                agent_name="test",
                ROOT_URL="http://test.com",
                record=False,
                arc_env=Mock()
            )
            
            action = agent._fallback_action()
            assert action == GameAction.ACTION5

    def test_memory_management(self):
        """Test memory recording and size limits."""
        with patch('rlms.RLM', MockRLM):
            agent = RLMAgent(
                card_id="test",
                game_id="test",
                agent_name="test",
                ROOT_URL="http://test.com",
                record=False,
                arc_env=Mock()
            )
            
            # Add many observations
            for i in range(60):
                agent._record_observation(f"ACTION{i%6+1}", f"Obs {i}", f"Reason {i}")
            
            # Should keep only last 50
            assert len(agent.memory) == 50
            assert agent.memory[-1]["observation"] == "Obs 59"

    def test_stuck_detection(self):
        """Test agent stuck state detection."""
        with patch('rlms.RLM', MockRLM):
            agent = RLMAgent(
                card_id="test",
                game_id="test",
                agent_name="test",
                ROOT_URL="http://test.com",
                record=False,
                arc_env=Mock()
            )
            
            frame = FrameData(
                game_id="test",
                frame=[[[1, 2], [3, 4]]],
                state=GameState.NOT_FINISHED,
                levels_completed=0
            )
            
            # Same grid multiple times should trigger stuck detection
            for i in range(6):
                is_stuck = agent._is_stuck(frame)
                if i >= 4:  # After 5 consecutive no changes
                    assert is_stuck
                else:
                    assert not is_stuck

    def test_exploratory_action(self):
        """Test exploratory action selection when stuck."""
        with patch('rlms.RLM', MockRLM):
            agent = RLMAgent(
                card_id="test",
                game_id="test",
                agent_name="test",
                ROOT_URL="http://test.com",
                record=False,
                arc_env=Mock()
            )
            
            action = agent._exploratory_action()
            assert action in [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5]

    def test_game_reset_handling(self):
        """Test game reset state handling."""
        with patch('rlms.RLM', MockRLM):
            agent = RLMAgent(
                card_id="test",
                game_id="test",
                agent_name="test",
                ROOT_URL="http://test.com",
                record=False,
                arc_env=Mock()
            )
            
            # Add some memory
            agent.memory.append({"test": "data"})
            agent.turn_number = 5
            agent.hypothesis = "Some hypothesis"
            
            # Handle reset
            frame = FrameData(
                game_id="test",
                frame=[[[1, 2], [3, 4]]],
                state=GameState.RESET,
                levels_completed=0
            )
            
            action = agent.choose_action(frame)
            
            assert action == GameAction.RESET
            assert len(agent.memory) == 0
            assert agent.turn_number == 0
            assert agent.hypothesis == "Game reset. Starting fresh exploration."
