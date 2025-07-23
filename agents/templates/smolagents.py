import logging
import os
import textwrap
import time
from typing import Any

from PIL import Image
from smolagents import (
    AgentImage,
    CodeAgent,
    OpenAIServerModel,
    Tool,
    ToolCallingAgent,
    tool,
)

from agents.structs import FrameData, GameAction, GameState
from agents.templates.llm_agents import LLM

from ..agent import Agent

logger = logging.getLogger()


class SmolCodingAgent(LLM, Agent):
    """An agent that uses CodeAgent from the smolagents library to play games."""

    MAX_ACTIONS: int = 100
    DO_OBSERVATION: bool = True

    MESSAGE_LIMIT: int = 10
    MODEL: str = os.environ.get("LLM_MODEL", "o4-mini")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def main(self) -> None:
        """The main function to initialize the agent and play the game until finished."""
        self.timer = time.time()
        model = OpenAIServerModel(self.MODEL, api_key=os.environ.get("LLM_API_KEY", ""), api_base=os.environ.get("LLM_BASE_URL", ""))

        # A CodeAgent calls and manipulates tools as Python functions which enables complex reasoning, algorithms etc.
        # Think BFS, DFS, A*, etc.
        agent = CodeAgent(
            model=model,
            planning_interval=10,
            tools=self.build_tools(),
            # Uncomment below to see the agent's raw outputs
            # verbosity_level=LogLevel.DEBUG,
        )

        # Reset the game at the start
        reset_frame = self.take_action(GameAction.RESET)
        if reset_frame:
            self.append_frame(reset_frame)

        # Start the agent
        prompt = self.build_initial_prompt(self.frames[-1])
        response = agent.run(prompt, max_steps=self.MAX_ACTIONS)
        print(response)

        self.cleanup()

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing or not."""
        return latest_frame.state is GameState.WIN

    def build_tools(self) -> list[Tool]:
        """Create smolagents tools for all available game actions.

        Returns:
            List of all game action tools
        """

        tools = []
        for action in GameAction:
            try:
                tool = self.create_smolagents_tool(action)
                tools.append(tool)
            except Exception as e:
                print(f"Failed to create tool for {action.name}: {e}")

        return tools

    def _execute_action(self, action: GameAction, action_description: str = "") -> str:
        """Helper method to execute an action and handle common logic.

        Args:
            action: The GameAction to execute
            action_description: Optional description for logging/responses

        Returns:
            String response describing the action result
        """
        if frame := self.take_action(action):
            self.append_frame(frame)
            logger.info(
                f"{self.game_id} - {action.name}: count {self.action_counter}, score {frame.score}, avg fps {self.fps})"
            )

            # Check if the game is won
            if self.is_done(self.frames, self.frames[-1]):
                return f"Action {action.name}{action_description} executed successfully! 🎉 GAME WON! The game is complete. Use the final_answer tool to end the run and report success."
            else:
                return self.build_func_resp_prompt(self.frames[-1])
        else:
            raise Exception(
                f"Action {action.name}{action_description} failed to execute properly."
            )

    def create_smolagents_tool(self, game_action: GameAction) -> Tool:
        """Converts GameAction dynamically into a smolagents tool.
        Actions in the game are called at the end of the tool execution.

        Args:
            game_action: The GameAction enum value to convert into a tool

        Returns:
            A smolagents Tool that can execute the specified game action
        """

        # This should be dynamic for each game, hardcoded for for the template
        llm_functions = self.build_functions()
        action_info = next(
            (f for f in llm_functions if f["name"] == game_action.name), None
        )

        if not action_info:
            raise ValueError(f"No function info found for {game_action.name}")

        description = action_info["description"]

        if game_action.is_simple():
            # Create a python function that is converted to a tool with the @tool decorator
            @tool  # type: ignore[misc]
            def simple_action() -> str:
                """Execute a simple game action."""
                return self._execute_action(game_action)

            # Update the tool's metadata to match the game action
            simple_action.name = game_action.name.lower()
            simple_action.description = description
            simple_action.inputs = {}
            simple_action.output_type = "string"

            return simple_action

        elif game_action.is_complex():
            # Create a python function that is converted to a tool with the @tool decorator with parameters
            @tool  # type: ignore[misc]
            def complex_action(x: int, y: int) -> str:
                """Execute a complex game action with coordinates.

                Args:
                    x: Coordinate X which must be Int<0,63>
                    y: Coordinate Y which must be Int<0,63>

                Returns:
                    String describing the action result and game state
                """
                if not (0 <= x <= 63):
                    return "Error: x coordinate must be between 0 and 63"
                if not (0 <= y <= 63):
                    return "Error: y coordinate must be between 0 and 63"

                # Create the action with coordinates
                action = game_action
                action.set_data({"x": x, "y": y})

                return self._execute_action(action, f" at coordinates ({x}, {y})")

            # Update the tool's metadata
            complex_action.name = game_action.name.lower()
            complex_action.description = description
            complex_action.inputs = {
                "x": {
                    "type": "integer",
                    "description": "Coordinate X which must be Int<0,63>",
                },
                "y": {
                    "type": "integer",
                    "description": "Coordinate Y which must be Int<0,63>",
                },
            }
            complex_action.output_type = "string"

            return complex_action

        else:
            raise ValueError(f"Unknown action type for {game_action.name}")

    def build_initial_prompt(self, latest_frame: FrameData) -> str:
        """Customize this method to provide instructions to the LLM."""
        return textwrap.dedent(
            """
# CONTEXT:
You are an agent playing an unknown dynamic game. Your objective is to
WIN and avoid GAME_OVER while minimizing actions.

One action produces one Frame. One Frame is made of one or more sequential
Grids. Each Grid is a matrix size INT<0,63> by INT<0,63> filled with
INT<0,15> values.


# Initial Game State:
Current State: {state}
Current Score: {score}

# Initial Frame:
{frame}

# INSTRUCTIONS:
First explore the game by taking actions and then determine the best strategy to WIN the game.
Use the available tools to take actions in the game. The game is already reset, so you can start taking other actions.
        """.format(
                state=latest_frame.state.name,
                score=latest_frame.score,
                frame=self.pretty_print_3d(latest_frame.frame),
            )
        )

    def build_func_resp_prompt(self, latest_frame: FrameData) -> str:
        return textwrap.dedent(
            """
# Game State:
{state}

# Score:
{score}

# Action Count:
{action_count}

# Current Frame:
{frame}
        """.format(
                state=latest_frame.state.name,
                score=latest_frame.score,
                action_count=len(self.frames),
                frame=self.pretty_print_3d(latest_frame.frame),
            )
        )


class SmolVisionAgent(LLM, Agent):
    """An agent that uses a multimodal model with the smolagents library to play games by seeing them."""

    MAX_ACTIONS: int = 100
    DO_OBSERVATION: bool = True

    MESSAGE_LIMIT: int = 10
    MODEL: str = "o4-mini"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def main(self) -> None:
        """The main agent loop. Play the game_id until finished, then exits."""
        self.timer = time.time()
        model = OpenAIServerModel(self.MODEL, api_key=os.environ.get("LLM_API_KEY", ""), api_base=os.environ.get("LLM_BASE_URL", ""))

        agent = ToolCallingAgent(
            model=model,
            tools=self.build_tools(),
            # Uncomment below to see the agent's raw outputs
            # verbosity_level=LogLevel.DEBUG,
            planning_interval=10,
        )

        # Reset the game at the start
        reset_frame = self.take_action(GameAction.RESET)
        if reset_frame:
            self.append_frame(reset_frame)

        # Start the agent
        prompt = self.build_initial_prompt(self.frames[-1])
        initial_image = self.grid_to_image(self.frames[-1].frame)
        agent.run(prompt, max_steps=self.MAX_ACTIONS, images=[initial_image])
        self.cleanup()

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing or not."""
        return latest_frame.state is GameState.WIN

    def build_tools(self) -> list[Tool]:
        """Create smolagents tools for all available game actions.

        Returns:
            List of all game action tools
        """
        tools = []
        for action in GameAction:
            try:
                tool = self.create_smolagents_tool(action)
                tools.append(tool)
            except Exception as e:
                print(f"Failed to create tool for {action.name}: {e}")
        return tools

    def _execute_action(
        self, action: GameAction, action_description: str = ""
    ) -> AgentImage:
        """Helper method to execute an action and handle common logic.

        Args:
            action: The GameAction to execute
            action_description: Optional description for logging/responses

        Returns:
            String response describing the action result, along with the new visual frame.
        """
        if frame := self.take_action(action):
            self.append_frame(frame)
            logger.info(
                f"{self.game_id} - {action.name}: count {self.action_counter}, score {frame.score}, avg fps {self.fps})"
            )

            image = self.grid_to_image(frame.frame)

            # Check if the game is won
            if self.is_done(self.frames, self.frames[-1]):
                return f"Action {action.name}{action_description} executed successfully! 🎉 GAME WON! The game is complete. Use the final_answer tool to end the run and report success."
            else:
                # The LLM will get the new frame from this response.
                return AgentImage(image)
        else:
            raise Exception(
                f"Action {action.name}{action_description} failed to execute properly."
            )

    def create_smolagents_tool(self, game_action: GameAction) -> Tool:
        """Universal function to convert any GameAction into a smolagents tool.

        Args:
            game_action: The GameAction enum value to convert into a tool

        Returns:
            A smolagents Tool that can execute the specified game action
        """

        # This should be dynamic for each game, hardcoded for for the template
        llm_functions = self.build_functions()
        action_info = next(
            (f for f in llm_functions if f["name"] == game_action.name), None
        )
        if not action_info:
            raise ValueError(f"No function info found for {game_action.name}")
        description = action_info["description"]

        if game_action.is_simple():
            # Create a python function that is converted to a tool with the decorator
            @tool  # type: ignore[misc]
            def simple_action() -> AgentImage:
                """Execute a simple game action."""
                return self._execute_action(game_action)

            # Update the tool's metadata to match the game action
            simple_action.name = game_action.name.lower()
            simple_action.description = description
            simple_action.inputs = {}
            simple_action.output_type = "image"

            return simple_action

        elif game_action.is_complex():
            # Create a python @tool function with parameters
            @tool  # type: ignore[misc]
            def complex_action(x: int, y: int) -> AgentImage:
                """Execute a complex game action with coordinates.

                Args:
                    x: Coordinate X which must be Int<0,63>
                    y: Coordinate Y which must be Int<0,63>

                Returns:
                    String describing the action result and game state
                """
                if not (0 <= x <= 63):
                    return "Error: x coordinate must be between 0 and 63"
                if not (0 <= y <= 63):
                    return "Error: y coordinate must be between 0 and 63"

                # Create the action with coordinates
                action = game_action
                action.set_data({"x": x, "y": y})

                return self._execute_action(action, f" at coordinates ({x}, {y})")

            # Update the tool's metadata to match the game action
            complex_action.name = game_action.name.lower()
            complex_action.description = description
            complex_action.inputs = {
                "x": {
                    "type": "integer",
                    "description": "Coordinate X which must be Int<0,63>",
                },
                "y": {
                    "type": "integer",
                    "description": "Coordinate Y which must be Int<0,63>",
                },
            }
            complex_action.output_type = "image"

            return complex_action

        else:
            raise ValueError(f"Unknown action type for {game_action.name}")

    def grid_to_image(self, grid: list[list[list[int]]]) -> Image.Image:
        """Converts a 3D grid of integers into an example PIL image, stacking grid layers horizontally."""
        color_map = [
            (0, 0, 0),
            (0, 0, 170),
            (0, 170, 0),
            (0, 170, 170),
            (170, 0, 0),
            (170, 0, 170),
            (170, 85, 0),
            (170, 170, 170),
            (85, 85, 85),
            (85, 85, 255),
            (85, 255, 85),
            (85, 255, 255),
            (255, 85, 85),
            (255, 85, 255),
            (255, 255, 85),
            (255, 255, 255),
        ]

        height = len(grid[0])
        width = len(grid[0][0])
        num_layers = len(grid)

        # Add a small separator between grids if there are multiple layers
        separator_width = 5 if num_layers > 1 else 0
        total_width = (width * num_layers) + (separator_width * (num_layers - 1))

        image = Image.new("RGB", (total_width, height), "white")
        pixels = image.load()

        for i, grid_layer in enumerate(grid):
            # Check if grid_layer is valid
            if len(grid_layer) != height or len(grid_layer[0]) != width:
                logger.warning(
                    f"Skipping inconsistent grid layer {i} in grid_to_image."
                )
                continue

            offset_x = i * (width + separator_width)
            for y in range(height):
                for x in range(width):
                    color_index = grid_layer[y][x] % 16
                    pixels[x + offset_x, y] = color_map[color_index]

        return image

    def build_initial_prompt(self, latest_frame: FrameData) -> str:
        """Customize this method to provide instructions to the LLM."""
        return textwrap.dedent(
            """
# CONTEXT:
You are an agent playing an unknown dynamic game by looking at images of the game. Your objective is to WIN and avoid GAME_OVER while never giving up.
You will be given an image of the current game state. You must call a tool to perform an action that updates the game state. The tool will return a description of the new game state.

# Initial Game State:
Current State: {state}
Current Score: {score}

# INSTRUCTIONS:
You can see the game state in the image. Analyze the image and the game state, then decide on the best action to take. The game is already reset, so you can start taking other actions.
# TURN:
Call exactly one action.
        """.format(
                state=latest_frame.state.name,
                score=latest_frame.score,
            )
        )
