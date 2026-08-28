import random

class QBrain:
    def __init__(self):
        self.q = {}

    def get_q(self, state, action):
        return self.q.get((str(state), action), 0.0)

    def update(self, state, action, reward, lr=0.1):
        key = (str(state), action)
        self.q[key] = self.get_q(state, action) + lr * (reward - self.get_q(state, action))

    def decide(self, state, actions):
        best_action = None
        best_value = -999

        for a in actions:
            value = self.get_q(state, a)

            if value > best_value:
                best_value = value
                best_action = a

        return best_action
