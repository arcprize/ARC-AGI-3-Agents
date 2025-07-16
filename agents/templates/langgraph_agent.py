import textwrap
import uuid
from dataclasses import dataclass
from typing import Any

from ..agent import Agent
from ..structs import FrameData, GameAction, GameState


@dataclass
class State:
    frames: list[FrameData]
    latest_frame: FrameData
    messages: list[dict[str, Any]]


SYS_PROMPT = """# CONTEXT:
You are an agent playing a dynamic game. Your objective is to
WIN and avoid GAME_OVER while minimizing actions.

One action produces one Frame. One Frame is made of one or more sequential
Grids. Each Grid is a matrix size INT<0,63> by INT<0,63> filled with
INT<0,15> values.

# TURN:
Call exactly one action.
"""


def build():
    import langsmith as ls
    from langchain.chat_models import init_chat_model
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.functions import entrypoint
    from langgraph.store.memory import InMemoryStore

    tools = _get_arc_tools()
    llm = init_chat_model("openai:o4-mini").bind_tools(tools, tool_choice="any")

    def format_frame(latest_frame: FrameData) -> str:
        lines = []
        for i, block in enumerate(latest_frame.frame):
            lines.append(f"Grid {i}:")
            for row in block:
                lines.append(f"  {row}")
            lines.append("")
        frame_txt = "\n".join(lines)
        return textwrap.dedent(
            """
    # State:
    {state}

    # Score:
    {score}

    # Frame:
    {latest_frame}

    # TURN:
    Reply with a few sentences of plain-text strategy observation about the frame to inform your next action.
        """.format(
                latest_frame=frame_txt,
                score=latest_frame.score,
                state=latest_frame.state.name,
            )
        )

    @ls.traceable("prompt")
    def prompt(latest_frame: FrameData, messages: list) -> str:
        """Build the user prompt for the LLM. Override this method to customize the prompt."""
        if len(messages) == 0:
            inbound = [{"role": "user", "content": format_frame(latest_frame)}]
        else:
            inbound = [
                {
                    "role": "tool",
                    "tool_call_id": messages[-1].tool_call_id,
                    "content": format_frame(latest_frame),
                },
                *messages,
            ]

        return [
            {"role": "system", "content": SYS_PROMPT},
            *messages,
            inbound,
        ]

    @entrypoint(checkpointer=InMemorySaver(), store=InMemoryStore())
    def agent(state: State) -> GameAction:
        sys_messages, *convo = prompt(state.latest_frame, state.messages)
        response = llm.invoke([sys_messages, *convo])
        return entrypoint.final(value=response, save=[*convo, response])

    return agent


# Required API

class LangGraphBase(Agent):
    """An agent that always selects actions at random."""

    MAX_ACTIONS = 80

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._thread_id = uuid.UUID(self.game_id)
        self.agent = build()

    @property
    def name(self) -> str:
        return f"LangGraphAgent.{super().name}.{self.MAX_ACTIONS}"

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
        return self.agent.invoke(
            {"frames": frames, "latest_frame": latest_frame},
            {"configurable": {"thread_id": self._thread_id}},
        )


## Copied from LLMAgent


def _get_functions() -> list[dict[str, Any]]:
    """Build JSON function description of game actions for LLM."""
    empty_params: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    functions: list[dict[str, Any]] = [
        {
            "name": GameAction.RESET.name,
            "description": "Start or restart a game. Must be called first when NOT_PLAYED or after GAME_OVER to play again.",
            "parameters": empty_params,
        },
        {
            "name": GameAction.ACTION1.name,
            "description": "Send this simple input action (1, A, Left).",
            "parameters": empty_params,
        },
        {
            "name": GameAction.ACTION2.name,
            "description": "Send this simple input action (2, D, Right).",
            "parameters": empty_params,
        },
        {
            "name": GameAction.ACTION3.name,
            "description": "Send this simple input action (3, W, Up).",
            "parameters": empty_params,
        },
        {
            "name": GameAction.ACTION4.name,
            "description": "Send this simple input action (4, S, Down).",
            "parameters": empty_params,
        },
        {
            "name": GameAction.ACTION5.name,
            "description": "Send this simple input action (5, Enter, Spacebar, Delete).",
            "parameters": empty_params,
        },
        {
            "name": GameAction.ACTION6.name,
            "description": "Send this complex input action (6, Click, Point).",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "string",
                        "description": "Coordinate X which must be Int<0,63>",
                    },
                    "y": {
                        "type": "string",
                        "description": "Coordinate Y which must be Int<0,63>",
                    },
                },
                "required": ["x", "y"],
                "additionalProperties": False,
            },
        },
    ]
    return functions


def _get_arc_tools() -> list[dict[str, Any]]:
    """Support models that expect tool_call format."""
    functions = _get_functions()
    tools: list[dict[str, Any]] = []
    for f in functions:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": f["name"],
                    "description": f["description"],
                    "parameters": f.get("parameters", {}),
                    "strict": True,
                },
            }
        )
    return tools
