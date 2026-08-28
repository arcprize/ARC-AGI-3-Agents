import arc_agi
import random
from arcengine import GameAction


# ============================================================
# MEMORY SYSTEM
# ============================================================

class Memory:
    def __init__(self):
        self.history = []

    def store(self, state, action, reward):
        self.history.append({
            "state": state,
            "action": action,
            "reward": reward
        })

    def recent(self):
        return self.history[-10:]


# ============================================================
# BRAIN (decision system)
# ============================================================

class Brain:

    def decide(self, state, memory):

        # 🧠 Rule 1: survival priority
        if state.get("danger"):
            return GameAction.ACTION6, {"x": 5, "y": 5}

        # 🧠 Rule 2: goal chasing
        if state.get("goal_visible"):
            return GameAction.ACTION2, {}

        # 🧠 Rule 3: memory influence
        if len(memory.recent()) > 5:
            return GameAction.ACTION3, {}

        # 🧠 fallback exploration
        return random.choice([
            (GameAction.ACTION1, {}),
            (GameAction.ACTION4, {}),
            (GameAction.ACTION5, {}),
        ])


# ============================================================
# STATE ENCODER (simple perception layer)
# ============================================================

def encode(obs):
    return {
        "raw": str(obs),
        "danger": "enemy" in str(obs),
        "goal_visible": "goal" in str(obs)
    }


# ============================================================
# MAIN AGENT
# ============================================================

class LMLMAgent:

    def __init__(self):
        self.arc = arc_agi.Arcade()
        self.env = self.arc.make(
            "ab12-v1",
            seed=0,
            render_mode="terminal"
        )

        self.brain = Brain()
        self.memory = Memory()

    def run(self, episodes=1):

        for _ in range(episodes):

            obs = self.env.reset()
            done = False

            while not done:

                # 1. perceive
                state = encode(obs)

                # 2. decide
                action, data = self.brain.decide(state, self.memory)

                # 3. act
                obs, reward, done, info = self.env.step(
                    action,
                    data=data
                )

                # 4. learn (store experience)
                self.memory.store(state, (action, data), reward)

                print("reward:", reward)


# ============================================================
# RUNNER
# ============================================================

if __name__ == "__main__":
    agent = LMLMAgent()
    agent.run()
