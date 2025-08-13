import random
import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.pregel import Pregel

from ..agent import Agent
from ..structs import FrameData, GameAction, GameState


class RandomAgentState(TypedDict):
    """State for the LangGraph workflow."""

    latest_frame: FrameData


class RandomAgentOutput(TypedDict):
    """Output from the LangGraph workflow."""

    action: GameAction


class LangGraphRandom(Agent):
    """An agent that always selects actions at random, using a LangGraph workflow."""

    MAX_ACTIONS = 80

    workflow: Pregel[RandomAgentState, Any, RandomAgentState, RandomAgentOutput]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1_000_000) + hash(self.game_id) % 1_000_000
        random.seed(seed)

        self.workflow = self._build_workflow()

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for decision making."""

        def choose_action(state: RandomAgentState) -> RandomAgentOutput:
            latest_frame = state["latest_frame"]

            if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                # if game is not started (at init or after GAME_OVER) we need to reset
                # add a small delay before resetting after GAME_OVER to avoid timeout
                action = GameAction.RESET
                action.reasoning = "Game not started or over - need to reset"
            else:
                # Choose a random action that isn't reset
                available_actions = [a for a in GameAction if a is not GameAction.RESET]
                action = random.choice(available_actions)

                if action.is_simple():
                    action.reasoning = f"RNG told me to pick {action.value}"
                elif action.is_complex():
                    action.set_data(
                        {
                            "x": random.randint(0, 63),
                            "y": random.randint(0, 63),
                        }
                    )
                    action.reasoning = {
                        "desired_action": f"{action.value}",
                        "my_reason": "RNG said so!",
                    }

            return {"action": action}

        # Build the graph
        workflow = StateGraph(
            RandomAgentState,
            input_schema=RandomAgentState,
            output_schema=RandomAgentOutput,
        )

        workflow.add_node("choose_action", choose_action)

        workflow.add_edge(START, "choose_action")
        workflow.add_edge("choose_action", END)

        return workflow.compile()

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing or not."""
        return any(
            [
                latest_frame.state is GameState.WIN,
                # uncomment to only let the agent play one time
                # latest_frame.state is GameState.GAME_OVER,
            ]
        )

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Choose action using LangGraph workflow."""

        # Prepare state for the graph
        initial_state: RandomAgentState = {
            "latest_frame": latest_frame,
        }

        # Execute the workflow
        output: RandomAgentOutput = self.workflow.invoke(initial_state)

        # Return the selected action
        return output["action"]
