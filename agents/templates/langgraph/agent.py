import sqlite3
from typing import Any, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.sqlite import SqliteStore

from ...agent import Agent
from ...structs import FrameData, GameAction, GameState
from .nodes import act, analyze_frame_delta, check_key, init
from .schema import LLM, AgentState


class LangGraph(Agent):
    """A LangGraph agent, using a variety of tools to make decisions."""

    MAX_ACTIONS = 20

    agent_state: AgentState
    workflow: CompiledStateGraph[AgentState, AgentState, AgentState]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.agent_state = {
            "action": None,
            "context": [],
            "key_matches_door": False,
            "llm": kwargs.get("llm", LLM.OPENAI_GPT_41),
            "thoughts": [],
            "frames": [],
            "latest_frame": None,  # type: ignore[typeddict-item]
            "previous_frame": None,
        }
        self.workflow = self._build_workflow()

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for decision making."""

        # Build the graph
        workflow = StateGraph(
            AgentState,
            input_schema=AgentState,
            output_schema=AgentState,
        )

        workflow.add_node("init", init)
        workflow.add_node("analyze_frame_delta", analyze_frame_delta)
        workflow.add_node("check_key", check_key)
        workflow.add_node("act", act)

        workflow.add_edge(START, "init")

        workflow.add_conditional_edges(
            "init",
            lambda state: state["action"] is GameAction.RESET,
            {True: END, False: "check_key"},
        )
        workflow.add_edge("check_key", "analyze_frame_delta")

        # Act after analysis
        workflow.add_edge("analyze_frame_delta", "act")

        workflow.add_edge("act", END)

        return workflow.compile(
            store=SqliteStore(
                sqlite3.connect(
                    "memory.db",
                    check_same_thread=False,
                    isolation_level=None,  # autocommit mode
                )
            )
        )

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
        self.agent_state = {
            **self.agent_state,
            "context": [],
            "frames": frames,
            "latest_frame": latest_frame,
            "previous_frame": self.agent_state["latest_frame"],
        }

        # Execute the workflow
        output: AgentState = self.workflow.invoke(self.agent_state)

        self.agent_state = output

        # Return the selected action
        return cast(GameAction, output["action"])
