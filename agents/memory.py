# agents/memory.py

class MemoryBank:
    def __init__(self):
        self.patterns = []

    def store(self, input_grid, output_grid):
        self.patterns.append((input_grid, output_grid))

    def retrieve_similar(self, grid):
        # naive similarity search (upgrade later to embeddings)
        return self.patterns[:5]
