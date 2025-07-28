from enum import Enum
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import BaseMessage

from ...structs import FrameData, GameAction


class LLM(Enum):
    OPENAI_GPT_41 = "openai:gpt-4.1"


class AgentState(TypedDict):
    """State for the LangGraph workflow."""

    action: Optional[GameAction]

    context: list[BaseMessage]
    """Additional context build up by the agent. Passed to the thinking node."""

    key_matches_door: bool

    frames: list[FrameData]
    latest_frame: FrameData
    previous_frame: Optional[FrameData]
    llm: LLM
    thoughts: list[str]


class KeyCheck(TypedDict):
    """Key check of the current state of the game."""

    shape_of_key: Annotated[
        str, ..., "Explain the shape of the key in the bottom-left corner of the frame."
    ]
    shape_of_exit_door: Annotated[
        str, ..., "Explain the shape of the exit door in the center of the frame."
    ]
    does_match: Annotated[
        bool,
        ...,
        "Whether the key is the correct shape to open the exit door. Return one of the following responses only: 'Match' or 'No Match'.",
    ]


class Observation(TypedDict):
    """
    An observation of the game's rules. Stored in long-term memory.
    """

    id: str
    observation: str
