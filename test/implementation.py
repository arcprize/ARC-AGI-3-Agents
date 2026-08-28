import arc_agi
import random
from arcengine import GameAction


# ============================================================
# MEMORY
# ============================================================

class Memory:
    def __init__(self):
        self.history = []

    def store(self, state, action, reward):
        self.history.append((state, action, reward))

    def recent(self, n=10):
        return self.history[-n:]


# ============================================================
# BRAIN (decision system)
# ============================================================

class Brain:

    def decide(self, state, memory):

        # 🧠 danger avoidance
        if state.get("danger"):
            return GameAction.ACTION6, {"x": 5, "y": 5}

        # 🧠 objective chasing
        if state.get("goal_visible"):
            return GameAction.ACTION2, {}

        # 🧠 memory-driven behavior
        if len(memory.recent()) > 5:
            return GameAction.ACTION3, {}

        # 🧠 exploration fallback
        return random.choice([
            (GameAction.ACTION1, {}),
            (GameAction.ACTION4, {}),
            (GameAction.ACTION5, {}),
        ])


# ============================================================
# STATE ENCODER
# ============================================================

def encode(obs):
    return {
        "raw": str(obs),
        "danger": "enemy" in str(obs),
        "goal_visible": "goal" in str(obs)
    }


# ============================================================
# AGENT
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

    def play_episode(self, max_steps=500):

        obs = self.env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:

            # 1. perceive
            state = encode(obs)

            # 2. decide
            action, data = self.brain.decide(state, self.memory)

            # 3. act
            obs, reward, done, info = self.env.step(
                action,
                data=data
            )

            # 4. learn
            self.memory.store(state, (action, data), reward)

            print(f"step={steps} reward={reward}")

            steps += 1

        print("Episode finished\n")

    def run(self, episodes=10):

        for i in range(episodes):
            print(f"\n=== EPISODE {i+1} ===")
            self.play_episode()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    agent = LMLMAgent()
    agent.run(episodes=5)
