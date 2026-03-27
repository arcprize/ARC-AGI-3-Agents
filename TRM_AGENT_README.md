# TRM Agent Implementation

This implementation adds a Trace-based Reasoning Model (TRM) agent to the ARC-AGI-3-Agents repository. The TRM agent uses program synthesis and neural-guided search to solve ARC tasks.

## Modules Added

### 1. `memory.py` - Trace Memory System
A persistent memory system for storing and retrieving program execution traces and hypotheses.

**Key Features:**
- JSON-lines based persistent storage
- Program fingerprinting and failure tracking
- Trajectory recording for game states
- Hypothesis management with confidence scores

**Main Classes:**
- `MemoryEntry`: Data structure for memory entries with metadata
- `TraceMemory`: Main memory manager with persistence

### 2. `dsl.py` - Domain-Specific Language
A DSL for program generation, mutation, and interpretation.

**Key Features:**
- Token-based program representation (ACTION and CONTROL tokens)
- Program generation with mutation and crossover operators
- Beam search for program optimization
- Interpreter for executing programs in ARC environments

**Main Classes:**
- `Program`: Token sequence representation
- `ProgramGenerator`: Random sampling, mutation, and crossover
- `Interpreter`: Executes programs in ARC environment
- `beam_search`: Search function for finding optimal programs

**Token Types:**
- **Action Tokens**: `ACTION1`, `ACTION2`, `ACTION3`, `ACTION4`, `RESET`, `WAIT`
- **Control Tokens**: `IF_CHANGED`, `IF_BLOCKED`, `REPEAT2`, `REPEAT3`, `EXPLORE`, `END`

### 3. `agents/templates/trm_agent.py` - TRM Agent
The main agent implementation using neural models and program synthesis.

**Key Features:**
- Neural encoders for observations and error signals
- Transformer-based reasoning model (TRM)
- Two-phase strategy: curiosity-driven exploration then synthesis
- Beam search with heuristic scoring
- Memory-based program filtering

**Main Classes:**
- `ErrorEncoder`: Neural encoder for program tokens
- `ObservationEncoder`: Neural encoder for game frames
- `TRM`: Transformer-based reasoning model
- `TRMAgent`: Main agent class inheriting from `Agent`

## Usage

The TRM agent can be used like any other agent in the repository:

```bash
# Run with a specific game
uv run main.py --agent=trmagent --game=ls20

# Run on all available games
uv run main.py --agent=trmagent
```

## Configuration

The TRM agent has several configurable parameters:

- `CURIOSITY_STEPS`: Number of initial exploration steps (default: 12)
- `BEAM_WIDTH`: Width of beam search (default: 6)
- `BEAM_DEPTH`: Depth of beam search (default: 2)
- `TRM_TEMPERATURE`: Sampling temperature for neural model (default: 1.2)
- `memory_path`: Path to persistent memory file (default: "pmll_memory.jsonl")

## Dependencies

The implementation adds the following dependency:
- `torch>=2.0.0`: For neural network models

## Testing

Tests for the new modules are located in `tests/unit/test_trm.py` and cover:
- Memory persistence and operations
- DSL program generation and manipulation
- Agent registration and initialization

Run tests with:
```bash
pytest tests/unit/test_trm.py -v
```

## Architecture

The TRM agent follows a two-phase approach:

1. **Curiosity Phase** (first N steps): Generates simple random programs for exploration
2. **Synthesis Phase** (remaining steps):
   - Uses TRM neural model to generate candidate programs
   - Applies beam search with heuristics to find optimal programs
   - Filters out previously failed programs using memory
   - Executes best program and tracks results

The agent maintains a queue of actions from the synthesized program and executes them sequentially, providing more coherent multi-step behavior than single-action agents.

## Implementation Notes

- The TRM model is initialized in evaluation mode (no training)
- Programs are scored based on action diversity and past failures
- Memory persists across runs to avoid repeating failed strategies
- The agent uses PyTorch for neural components but doesn't require GPU
