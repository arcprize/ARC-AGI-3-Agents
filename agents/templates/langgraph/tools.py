import logging
import uuid
from typing import Literal, TypedDict, cast

from langchain_core.tools import tool

from langgraph.config import get_store

from ...structs import GameAction

log = logging.getLogger(__name__)


class GameActionSimple(TypedDict):
    type: Literal["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"]


class GameActionComplex(TypedDict):
    type: Literal["ACTION6"]
    x: int
    y: int


@tool  # type: ignore[misc]
def act(action: GameActionSimple | GameActionComplex) -> GameAction:
    """Perform an action in the game."""

    log.info(f"👉 {action}")

    if "x" in action:
        action = cast(GameActionComplex, action)
        act = GameAction.from_name(action["type"])
        act.set_data({"x": action["x"], "y": action["y"]})
        return act
    else:
        act = GameAction.from_name(action["type"])
        return act


@tool  # type: ignore[misc]
def think(thought: str) -> str:
    """
    Think about your next action or what is happening in the environment.

    This will not add an observation to your journal, so it is good for short-term thinking or reflection in the moment.
    """
    log.info(f"🤔 {thought}")
    return f"Thought: {thought}"


@tool  # type: ignore[misc]
def delete_observation(id: str) -> str:
    """Delete an observation from your journal. Useful if you think it no longer applies."""

    store = get_store()
    store.delete(("observations"), id)
    return f"Observation deleted with ID: {id}"


@tool  # type: ignore[misc]
def observe(observation: str) -> str:
    """
    Stores an observation about the game in your journal.

    These observations are long-lived and will persist between game sessions.

    Example: After confirming how ACTION1 works, it would be a good idea to store an observation about it for future reference.
    """
    store = get_store()
    id = uuid.uuid4()

    log.info(f"👀 {observation}")

    store.put(
        ("observations"),
        id,
        observation,
    )

    return f"Observation stored with ID: {id}"


all_tools = [act, delete_observation, observe, think]
