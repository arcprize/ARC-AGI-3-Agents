import random
from arcengine import GameAction

class Brain:

    def decide(self, state, memory):

        # 🧠 RULE 1: survival priority
        if state["danger"]:
            return GameAction.ACTION6, {"x": 5, "y": 5}

        # 🧠 RULE 2: goal chasing
        if state["goal_visible"]:
            return GameAction.ACTION2, {}

        # 🧠 RULE 3: memory influence
        if len(memory.recent()) > 5:
            return GameAction.ACTION3, {}

        # 🧠 fallback exploration
        return random.choice([
            (GameAction.ACTION1, {}),
            (GameAction.ACTION4, {}),
            (GameAction.ACTION5, {}),
        ])
