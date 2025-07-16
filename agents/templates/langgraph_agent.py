import json
import logging
import struct
import textwrap
import uuid
from typing import Any, TypedDict

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage

from ..agent import Agent
from ..structs import FrameData, GameAction, GameState

logger = logging.getLogger(__name__)


class State(TypedDict):
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
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.func import entrypoint
    from langgraph.store.memory import InMemoryStore
    from langsmith.schemas import Attachment
    from langsmith.wrappers import wrap_openai

    tools = _get_arc_tools()
    openai_client = wrap_openai(OpenAI())

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

    @ls.traceable(run_type="prompt")
    def prompt(latest_frame: FrameData, messages: list) -> list:
        """Build the user prompt for the LLM. Override this method to customize the prompt."""
        if (rt := ls.get_current_run_tree()) and latest_frame.frame:
            rt.attachments["frame"] = Attachment(
                mime_type="image/bmp",
                data=_bmp(latest_frame.frame),
            )

        if len(messages) == 0:
            inbound = {"role": "user", "content": format_frame(latest_frame)}
        else:
            inbound = {
                "role": "tool",
                "tool_call_id": messages[-1].tool_call_id,
                "content": format_frame(latest_frame),
            }

        return [
            {"role": "system", "content": SYS_PROMPT},
            *messages,
            inbound,
        ]

    @entrypoint(checkpointer=InMemorySaver(), store=InMemoryStore())
    def agent(state: State) -> dict:
        # TODO: handle the frame bursts
        sys_messages, *convo = prompt(state["latest_frame"], state.get("messages", []))
        response = openai_client.chat.completions.create(
            model="o4-mini",
            messages=[sys_messages, *convo],
            tools=tools,
            tool_choice="required",
        )
        ai_msg = response.choices[0].message
        ai_msg.tool_calls = ai_msg.tool_calls[:1]
        return entrypoint.final(value=ai_msg, save={"messages": [*convo, ai_msg]})

    return agent


# Required API


class LangGraph(Agent):
    """An agent that always selects actions at random."""

    MAX_ACTIONS = 80

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._thread_id = uuid.uuid5(uuid.NAMESPACE_DNS, self.game_id)
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
        import langsmith as ls

        with ls.trace("choose_action") as rt:
            msg: ChatCompletionMessage = self.agent.invoke(
                {"frames": frames, "latest_frame": latest_frame},
                {"configurable": {"thread_id": self._thread_id}},
            )
            func = msg.tool_calls[0].function
            action = GameAction.from_name(func.name)
            try:
                args = json.loads(func.arguments) if func.arguments else {}
            except Exception as e:
                args = {}
                logger.warning(f"JSON parsing error on LLM function response: {e}")
            action.set_data(args)
            rt.end(outputs={"action": action})
        return action

    def main(self) -> None:
        import langsmith as ls

        with ls.trace(
            "LangGraph Agent",
            input={"state": self.state},
            metadata={
                "game_id": self.game_id,
                "card_id": self.card_id,
                "agent_name": self.agent_name,
                "thread_id": self._thread_id,
            },
        ) as rt:
            super().main()
            rt.end(outputs={"state": self.state})


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


def _bmp(b):
    while isinstance(b[0][0], (list, tuple)):
        b = b[0]
    h, w = len(b), len(b[0])
    p = b""
    for r in b[::-1]:
        p += b"".join(
            (v and b"\x00\x00\x00" or b"\xff\xff\xff") for v in r
        ) + b"\x00" * ((-3 * w) % 4)
    return (
        b"BM"
        + struct.pack("<IHHI", 14 + 40 + len(p), 0, 0, 14 + 40)
        + struct.pack("<IiiHHIIIIII", 40, w, h, 1, 24, 0, len(p), 0, 0, 0, 0)
        + p
    )
