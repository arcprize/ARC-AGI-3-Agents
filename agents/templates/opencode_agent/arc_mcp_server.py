import asyncio
import logging
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("arc-mcp-server")

app = Server("arc-game-tools")


@app.list_tools()
async def list_tools() -> list[Tool]:
    logger.info("Listing ARC game action tools")
    return [
        Tool(
            name="reset_game",
            description="Reset the game to its initial state. Use this when you want to start over from the beginning.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="action1_move_up",
            description="Execute ACTION1 (typically move up).",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="action2_move_down",
            description="Execute ACTION2 (typically move down).",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="action3_move_left",
            description="Execute ACTION3 (typically move left).",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="action4_move_right",
            description="Execute ACTION4 (typically move right).",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="action5_interact",
            description="Execute ACTION5 (typically interact with environment).",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="action6_click",
            description="Execute ACTION6 with coordinates (x, y). x: horizontal coordinate (0 = left, 63 = right, range 0-63). y: vertical coordinate (0 = top, 63 = bottom, range 0-63)",
            inputSchema={
                "type": "object",
                "properties": {
                    "x": {
                        "type": "integer",
                        "description": "Horizontal coordinate (0-63)",
                        "minimum": 0,
                        "maximum": 63
                    },
                    "y": {
                        "type": "integer",
                        "description": "Vertical coordinate (0-63)",
                        "minimum": 0,
                        "maximum": 63
                    }
                },
                "required": ["x", "y"]
            }
        ),
        Tool(
            name="action7_undo",
            description="Execute ACTION7 (typically undo the previous action).",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    logger.info(f"Tool called: {name} with arguments: {arguments}")
    
    if name == "reset_game":
        return [TextContent(
            type="text",
            text="RESET action will be executed."
        )]
    
    elif name == "action1_move_up":
        return [TextContent(
            type="text",
            text="ACTION1 will be executed."
        )]
    
    elif name == "action2_move_down":
        return [TextContent(
            type="text",
            text="ACTION2 will be executed."
        )]
    
    elif name == "action3_move_left":
        return [TextContent(
            type="text",
            text="ACTION3 will be executed."
        )]
    
    elif name == "action4_move_right":
        return [TextContent(
            type="text",
            text="ACTION4 will be executed."
        )]
    
    elif name == "action5_interact":
        return [TextContent(
            type="text",
            text="ACTION5 will be executed."
        )]
    
    elif name == "action6_click":
        x = arguments.get("x", 0)
        y = arguments.get("y", 0)
        
        if not (0 <= x <= 63 and 0 <= y <= 63):
            return [TextContent(
                type="text",
                text=f"Invalid coordinates: x={x}, y={y}. Must be in range 0-63."
            )]
        
        return [TextContent(
            type="text",
            text=f"ACTION6 will be executed at ({x}, {y})."
        )]
    
    elif name == "action7_undo":
        return [TextContent(
            type="text",
            text="ACTION7 (undo) will be executed."
        )]
    
    else:
        return [TextContent(
            type="text",
            text=f"Unknown tool: {name}"
        )]


async def main():
    logger.info("Starting ARC MCP Server...")
    try:
        async with stdio_server() as (read_stream, write_stream):
            logger.info("MCP Server initialized, waiting for requests")
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options()
            )
    except Exception as e:
        logger.error(f"MCP Server error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("MCP Server shutting down")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
