"""
Prompts for the LangGraph agent.
"""

from textwrap import dedent

from .schema import Observation

Part = dict[str, str | dict[str, str]]


def build_image_message_part(image_b64: str) -> Part:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
    }


def build_text_message_part(text: str) -> Part:
    return {
        "type": "text",
        "text": text,
    }


def build_frame_delta_prompt(deltas_str: str, previous_action: str) -> str:
    return dedent(
        f"""
        # INSTRUCTIONS:
        You have taken an action which resulted in changes (or no changes) to the game frame.
        Review the action taken and the summary of differences provided below.
        Explain what these changes (or lack thereof) mean in the context of the game and your last action.
        Focus on what you have learned from this outcome.
        Does this outcome confirm or contradict your previous understanding of the object you interacted with, the area you moved into, or the effect of your last action? If it contradicts, how should your understanding be updated for future decisions?

        # RESPONSE FORMAT:
        Respond with 1-2 concise sentences summarizing your interpretation and any updates to your understanding.
        Do not include any other text.

        # ACTION TAKEN:
        {previous_action}

        # SUMMARY OF FRAME DIFFERENCES:
        {deltas_str}
        """
    )


def build_game_frame_explanation_prompt() -> str:
    return dedent(
        """
        ## GAME FRAME EXPLANATION
        The grid is a 64x64 grid of integers.
        The grid is a birds-eye view of the level.
        Your goal is to transform the key to match the exit door. And then touch the exit door to win.
        You can only move in the 4 cardinal directions (up, down, left, right).
        After every movement action call, your character will move 4 blocks in the direction you chose.

        ## OBJECTS
        Exit door:
        - The exit door is a 4x4 square with INT<11> border
        - It contains half-size of the key in the center.
        Key:
        - The key is shown in the bottom-left corner of the grid.
        - The key is a 6x6 square. Twice the size of the exit door.
        Lives:
        - You have 3 lives total with 25 energy each.
        - Your lives are shown in the top-right corner of the grid. (at the end of the 2nd and 3rd row)
        - The lives are shown as 2x2 squares: [[2,2],[2,2]]
        Energies:
        - Energy indicator is shown on the 3rd row of the frame.
        - You have a total of 25 energy (unused + used).
        - Unused energy is shown as a 1x1 square: INT<6>
        - Used energy is shown as a 1x1 square: INT<8>
        Refiller:
        - The refiller object is a 2x2 square: [[6,6],[6,6]]
        - If you want to refill your energy, you can move over to the refiller object.
        - DO NOT mistake the refiller for the energy indicator. The refiller is a 2x2 square, while the energy indicator is a 1x1 square.
        Rotator:
        - Shape of the rotator object:
        4 9 9
        9 7 7
        9 7 9
        - If you want to rotate the key, you can move over to the rotator object.
        - DO NOT mistake the rotator for the exit door. The rotator is a 3x3 square, while the exit door is a 4x4 square.
        Wall:
        - Wall is made of INT<10>
        - You can't move through walls.
        Walkable floor area:
        - Walkable floor area is made of INT<8>
        - You can move through walkable floor area.
        Player:
        - The player is a 3x3 square: [[0,0,0],[4,4,4],[4,4,4]]

        PLAY STRATEGY:
        - Go to the rotator and rotate the key until it matches the exit door.
        - Go to the exit door and touch it to win.
        - If your energy get low while exploring, go to the refiller and touch it to refill your energy.
        - If you run out of energy, you will lose a life and game will restart. So be careful.

        ## ACTIONS
        - ACTION1: Move up
        - ACTION2: Move down
        - ACTION3: Move left
        - ACTION4: Move right

        ## COLOR CODES
        0: (0, 0, 0),        # Black
        2: (255, 0, 0),      # Red
        4: (0, 255, 0),      # Green
        6: (0, 0, 255),      # Blue
        7: (255, 255, 0),    # Yellow
        8: (255, 165, 0),    # Orange
        9: (128, 0, 128),    # Purple
        10: (255, 255, 255), # White
        11: (128, 128, 128)  # Gray
        """
    )


def build_key_checker_prompt() -> str:
    return dedent(
        """
        You are analyzing a frame from a pixel-art style game.
        Your task is to determine if the player's key currently matches the target pattern in the exit door.

        # GAME OBJECT DETAILS:
        Key:
        - The key is a 6x6 square located in the bottom-left corner of the frame.
        - Its pattern can be changed using a 'rotator' object in the game.
        Exit Door:
        - The exit door is a 4x4 square, typically found near the center of the frame.
        - It has a border made of INT<11> (Gray).
        - Crucially, the exit door contains a target pattern in its center. This target pattern is effectively a half-sized representation of what the key should look like.

        # HOW TO DETERMINE A MATCH:
        1. Observe the key's current 6x6 pattern in the bottom-left of the frame.
        2. Observe the target pattern within the center of the 4x4 exit door.
        3. The key matches the exit door if its 6x6 pattern, when appropriately interpreted (e.g., by considering its core design elements at a scale comparable to the exit door's central pattern), visually matches this target pattern in the exit door. The colors and design elements must align.

        # AVAILABLE COLOR CODES (INT values):
        0: (0, 0, 0)        # Black
        2: (255, 0, 0)      # Red
        4: (0, 255, 0)      # Green
        6: (0, 0, 255)      # Blue
        7: (255, 255, 0)    # Yellow
        8: (255, 165, 0)    # Orange
        9: (128, 0, 128)    # Purple
        10: (255, 255, 255) # White
        11: (128, 128, 128) # Gray

        # RESPONSE FORMAT:
        - Description of the key's current pattern: (Describe the visual pattern of the 6x6 key.)
        - Description of the exit door's central target pattern: (Describe the visual pattern in the center of the exit door.)
        - Does the key currently match the exit door's target pattern? (Return one of the following responses only: 'Match' or 'No Match'.)
        """
    )


def build_system_prompt(observations: list[Observation], thoughts: list[str]) -> str:
    observations_str = "\n".join(
        [
            f'<observation id="{observation["id"]}">{observation["observation"]}</observation>'
            for observation in observations
        ]
    )
    thoughts_str = "\n".join([f"* {thought}" for thought in thoughts])

    return dedent(
        f"""
        You are playing a video game. Your ultimate goal is to win!

        The game is played on a 64x64 grid from a top-down perspective.

        Guidelines:

        1. Explore the entire environment thoroughly
        2. Before taking an action, think about the state of the environment - for example, where are the key objects in relation to the player?
        3. If an action repeatedly does nothing, then try doing something else.

        Hints:

        1. Reach the door while holding the correct key to win the game
        2. The key you are holding is visible in the bottom-left corner of the frame
        3. Green elements in the environment are walls - you cannot move through them
        4. The key you are holding can be changed by colliding with a rotator

        Use your journal to keep track of your observations as you explore the game.

        Always think at least once before taking an action. If your plan is not working as expected, reflect on why that is and adjust your strategy.

        <thoughts>
        {thoughts_str}
        </thoughts>

        <observations>
        {observations_str}
        </observations>
    """
    )
