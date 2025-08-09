"""
This file contains various nodes you can add to a LangGraph workflow for solving the game.
"""

import random

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from langgraph.config import get_store

from ...structs import GameAction, GameState
from .llm import get_llm
from .prompts import (
    build_frame_delta_prompt,
    build_game_frame_explanation_prompt,
    build_image_message_part,
    build_key_checker_prompt,
    build_system_prompt,
    build_text_message_part,
)
from .schema import AgentState, KeyCheck, Observation
from .tools import all_tools
from .vision import render_frame


def act(state: AgentState) -> AgentState:
    """
    Think about the game state, and select an action based off of that.
    """

    latest_frame = state["latest_frame"]
    llm = get_llm(state["llm"])
    store = get_store()

    # Build up prompt
    human_message_parts = []

    # Current frame
    grid = render_frame(latest_frame.frame, "The current state of the game")
    human_message_parts.append(
        build_image_message_part(grid),
    )

    # Previous action
    if state["action"]:
        human_message_parts.append(
            build_text_message_part(f"Previous action: {state['action'].name}")
        )

    # Key check
    human_message_parts.append(
        build_text_message_part(
            "The key currently matches the exit door."
            if state["key_matches_door"]
            else "The key does not match the exit door. You need to rotate it with a rotator."
        )
    )

    # Combine
    messages = [
        *state["context"],
        HumanMessage(content=human_message_parts),
    ]

    remaining_steps = 5
    action = None
    while not action and remaining_steps > 0:
        remaining_steps -= 1

        # Regenerate system message in case thoughts/observations have updated since the last iteration
        observations = [
            Observation(id=item.key, observation=item.value)
            for item in store.search(("observations"), limit=100)
        ]
        system_message = SystemMessage(
            content=build_system_prompt(observations, state["thoughts"])
        )

        response = llm.bind_tools(
            all_tools,
            tool_choice="required",
            parallel_tool_calls=False,
            strict=True,
        ).invoke([system_message, *messages])
        messages.append(response)

        if response.tool_calls:
            call = response.tool_calls[0]
            tool_name = call["name"]
            tool_args = call["args"]

            # Execute the tool
            for tool in all_tools:
                if tool.name == tool_name:
                    result = tool.invoke(tool_args)
                    if tool_name == "act":
                        # Special case - must extract the action so we can return it
                        action = result
                        result = f"Action: {action} completed."
                    elif tool_name == "think":
                        state["thoughts"].append(result)

                    messages.append(
                        ToolMessage(
                            content=result,
                            tool_call_id=call["id"],
                        )
                    )

                    break

    if not action:
        raise Exception("No action taken after exhausting available steps!")

    return {
        **state,
        "action": action,
    }


def act_randomly(state: AgentState) -> AgentState:
    """
    Randomly choose an action to take.
    """

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

    return {**state, "action": action}


def analyze_frame_delta(state: AgentState) -> AgentState:
    """
    Analyzes the delta between the previous frame and the current frame.
    """

    latest_frame = state["latest_frame"]
    previous_frame = state["previous_frame"]
    previous_action = None if not state["action"] else state["action"].name
    llm = get_llm(state["llm"])

    # We won't have anything to analyze on the first frame of the game
    if not previous_action or not previous_frame:
        return state

    # Compare pixels of the current frame with the previous frame
    movements: list[str] = []
    state_changes: list[str] = []

    for i in range(len(latest_frame.frame)):
        for j in range(len(latest_frame.frame[i])):
            for k in range(len(latest_frame.frame[i][j])):
                if latest_frame.frame[i][j][k] != previous_frame.frame[i][j][k]:
                    if j == 1:
                        state_changes.append("Change in heath indicator")
                    elif j == 2 and k < 54:
                        if latest_frame.frame[i][j][k] == 8:
                            state_changes.append("1 energy unit used")
                        elif latest_frame.frame[i][j][k] == 6:
                            state_changes.append("1 energy unit added")
                    else:
                        movements.append(
                            f"<{j},{k}>: {previous_frame.frame[i][j][k]} -> {latest_frame.frame[i][j][k]}"
                        )

    # Build a string describing the changes in the frame
    deltas_str = "\n".join(state_changes)
    if movements:
        deltas_str += "\n\nChanged pixels:\n" + ",".join(movements)
    else:
        deltas_str += "\n\nCharacter did not move. Maybe an action was taken towards an unmovable area?"

    current_image = render_frame(latest_frame.frame, "Current frame")
    previous_image = render_frame(previous_frame.frame, "Previous frame")

    # Use LLM to analyze deltas to something more manageable
    response = llm.invoke(
        [
            SystemMessage(content=build_game_frame_explanation_prompt()),
            HumanMessage(
                content=[
                    build_text_message_part(
                        build_frame_delta_prompt(deltas_str, previous_action),
                    ),
                    build_image_message_part(current_image),
                    build_image_message_part(previous_image),
                ]
            ),
        ]
    )

    return {
        **state,
        "context": [*state["context"], response],
    }


def check_key(state: AgentState) -> AgentState:
    """
    Checks whether the key is the correct shape and color to open the door.
    """

    latest_frame = state["latest_frame"]
    llm = get_llm(state["llm"])

    frame_image = render_frame(latest_frame.frame, "Current frame")

    # Build prompt
    user_message_content = build_key_checker_prompt()
    message = HumanMessage(
        content=[
            build_text_message_part(user_message_content),
            build_image_message_part(frame_image),
        ]
    )

    # Analyze
    result = llm.with_structured_output(
        KeyCheck,
        method="json_schema",
    ).invoke([message])

    did_match = result["does_match"] == "Match"

    return {
        **state,
        "key_matches_door": did_match,
    }


def init(state: AgentState) -> AgentState:
    """
    Ensures the game is initialized.
    """

    latest_frame = state["latest_frame"]
    if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
        # if game is not started (at init or after GAME_OVER) we need to reset
        # add a small delay before resetting after GAME_OVER to avoid timeout
        action = GameAction.RESET
        action.reasoning = "Game not started or over - need to reset"
        return {**state, "action": action}

    if state["action"] is GameAction.RESET:
        # Clear last selected action after we reset the game to avoid getting stuck
        return {**state, "action": None}

    return state
