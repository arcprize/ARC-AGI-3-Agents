import arc_agi
from arcengine import GameAction

from brain01 import Brain
from enc01 import encode
from mem01 import Memory
from rew01 import compute_reward

arc = arc_agi.Arcade()
env = arc.make("ab12-v1", seed=0, render_mode="terminal")

brain = Brain()
memory = Memory()

obs = env.reset()
done = False

while not done:

    # 🧠 1. perceive
    state = encode(obs)

    # 🧠 2. think
    action, data = brain.decide(state, memory)

    # 🧠 3. act
    obs, reward, done, info = env.step(action, data=data)

    # 🧠 4. learn (store experience)
    memory.store(state, (action, data), reward)

    print("reward:", reward)
