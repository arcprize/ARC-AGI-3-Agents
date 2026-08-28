class Swarm:

    def __init__(self):
        self.brains = {
            "explorer": Brain(),
            "planner": Brain(),
            "critic": Brain()
        }

    def vote(self, state, memory):

        votes = []

        for name, brain in self.brains.items():
            action, data = brain.decide(state, memory)
            votes.append((action, data))

        return random.choice(votes)
