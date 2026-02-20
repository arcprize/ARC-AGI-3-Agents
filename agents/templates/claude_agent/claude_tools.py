import json
import logging
from typing import Any, TYPE_CHECKING

from arcengine import FrameData
from claude_agent_sdk import tool, create_sdk_mcp_server

if TYPE_CHECKING:
    from .claude_agents import ClaudeCodeAgent

logger = logging.getLogger()


def format_frame_info(agent: "ClaudeCodeAgent") -> str:
    latest_frame = agent.current_frame if agent.current_frame else (agent.frames[-1] if agent.frames else None)
    
    if not latest_frame:
        return "No frame data available."
    
    available_actions_str = ", ".join(
        [f"ACTION{a}" if a > 0 else "RESET" for a in latest_frame.available_actions]
    )
    
    return json.dumps({
        "state": latest_frame.state.value,
        "levels_completed": latest_frame.levels_completed,
        "available_actions": available_actions_str,
    }, indent=2)


def create_arc_tools_server(agent: "ClaudeCodeAgent") -> Any:
    def check_action_available(action_value: int) -> tuple[bool, str]:
        latest_frame = agent.current_frame
        if not latest_frame or not latest_frame.available_actions:
            return True, ""
        
        if action_value in latest_frame.available_actions:
            return True, ""
        
        available = ", ".join([f"ACTION{a}" if a > 0 else "RESET" for a in latest_frame.available_actions])
        return False, f"ERROR: ACTION{action_value} is not available in this game. Available actions: {available}"
    
    @tool(
        "reset_game",
        "Reset the game to its initial state. Use this when you want to start over from the beginning.",
        {}
    )
    async def reset_game(args: dict[str, Any]) -> dict[str, Any]:
        is_valid, error_msg = check_action_available(0)
        if not is_valid:
            return {
                "content": [{
                    "type": "text",
                    "text": error_msg
                }],
                "is_error": True
            }
        
        return {
            "content": [{
                "type": "text",
                "text": f"RESET action will be executed. Current state:\n{format_frame_info(agent)}"
            }]
        }
    
    @tool(
        "action1_move_up",
        "Execute ACTION1 (typically move up).",
        {}
    )
    async def action1_move_up(args: dict[str, Any]) -> dict[str, Any]:
        is_valid, error_msg = check_action_available(1)
        if not is_valid:
            return {
                "content": [{
                    "type": "text",
                    "text": error_msg
                }],
                "is_error": True
            }
        
        return {
            "content": [{
                "type": "text",
                "text": f"ACTION1 will be executed. Current state:\n{format_frame_info(agent)}"
            }]
        }
    
    @tool(
        "action2_move_down",
        "Execute ACTION2 (typically move down).",
        {}
    )
    async def action2_move_down(args: dict[str, Any]) -> dict[str, Any]:
        is_valid, error_msg = check_action_available(2)
        if not is_valid:
            return {
                "content": [{
                    "type": "text",
                    "text": error_msg
                }],
                "is_error": True
            }
        
        return {
            "content": [{
                "type": "text",
                "text": f"ACTION2 will be executed. Current state:\n{format_frame_info(agent)}"
            }]
        }
    
    @tool(
        "action3_move_left",
        "Execute ACTION3 (typically move left).",
        {}
    )
    async def action3_move_left(args: dict[str, Any]) -> dict[str, Any]:
        is_valid, error_msg = check_action_available(3)
        if not is_valid:
            return {
                "content": [{
                    "type": "text",
                    "text": error_msg
                }],
                "is_error": True
            }
        
        return {
            "content": [{
                "type": "text",
                "text": f"ACTION3 will be executed. Current state:\n{format_frame_info(agent)}"
            }]
        }
    
    @tool(
        "action4_move_right",
        "Execute ACTION4 (typically move right).",
        {}
    )
    async def action4_move_right(args: dict[str, Any]) -> dict[str, Any]:
        is_valid, error_msg = check_action_available(4)
        if not is_valid:
            return {
                "content": [{
                    "type": "text",
                    "text": error_msg
                }],
                "is_error": True
            }
        
        return {
            "content": [{
                "type": "text",
                "text": f"ACTION4 will be executed. Current state:\n{format_frame_info(agent)}"
            }]
        }
    
    @tool(
        "action5_interact",
        "Execute ACTION5 (typically interact with environment).",
        {}
    )
    async def action5_interact(args: dict[str, Any]) -> dict[str, Any]:
        is_valid, error_msg = check_action_available(5)
        if not is_valid:
            return {
                "content": [{
                    "type": "text",
                    "text": error_msg
                }],
                "is_error": True
            }
        
        return {
            "content": [{
                "type": "text",
                "text": f"ACTION5 will be executed. Current state:\n{format_frame_info(agent)}"
            }]
        }
    
    @tool(
        "action6_click",
        "Execute ACTION6 with coordinates (x, y). Coordinates must be in range 0-63.",
        {
            "x": int,
            "y": int
        }
    )
    async def action6_click(args: dict[str, Any]) -> dict[str, Any]:
        is_valid, error_msg = check_action_available(6)
        if not is_valid:
            return {
                "content": [{
                    "type": "text",
                    "text": error_msg
                }],
                "is_error": True
            }
        
        x = args.get("x", 0)
        y = args.get("y", 0)
        
        if not (0 <= x <= 63 and 0 <= y <= 63):
            return {
                "content": [{
                    "type": "text",
                    "text": f"Invalid coordinates: x={x}, y={y}. Must be in range 0-63."
                }],
                "is_error": True
            }
        
        return {
            "content": [{
                "type": "text",
                "text": f"ACTION6 will be executed at ({x}, {y}). Current state:\n{format_frame_info(agent)}"
            }]
        }
    
    @tool(
        "action7_undo",
        "Execute ACTION7 (typically undo the previous action).",
        {}
    )
    async def action7_undo(args: dict[str, Any]) -> dict[str, Any]:
        is_valid, error_msg = check_action_available(7)
        if not is_valid:
            return {
                "content": [{
                    "type": "text",
                    "text": error_msg
                }],
                "is_error": True
            }
        
        return {
            "content": [{
                "type": "text",
                "text": f"ACTION7 (undo) will be executed. Current state:\n{format_frame_info(agent)}"
            }]
        }
    
    return create_sdk_mcp_server(
        name="arc-game-tools",
        version="1.0.0",
        tools=[
            reset_game,
            action1_move_up,
            action2_move_down,
            action3_move_left,
            action4_move_right,
            action5_interact,
            action6_click,
            action7_undo,
        ]
    )
