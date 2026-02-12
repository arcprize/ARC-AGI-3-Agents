# RLM Agent - Recursive Language Model

The RLM Agent is a cutting-edge implementation for ARC-AGI-3 that uses recursive language models with Python REPL capabilities to analyze game states, form hypotheses, and make informed decisions.

## Features

- **Recursive Reasoning**: Uses the `rlms` library to create a recursive reasoning loop
- **Python REPL**: Built-in Python environment for grid analysis and pattern detection
- **Multi-Backend Support**: Compatible with OpenRouter, OpenAI, and Anthropic models
- **Hypothesis Evolution**: Maintains and updates hypotheses about game rules
- **Memory Management**: Episodic memory for learning from past observations
- **Stuck Detection**: Automatically detects when stuck and takes exploratory actions

## Quick Start

### 1. Installation

```bash
# Install dependencies
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### 2. Configuration

Add these to your `.env` file:

```bash
# OpenRouter (recommended for RLM)
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=google/gemini-2.5-flash

# RLM Agent Configuration
RLM_BACKEND=openrouter        # openrouter, openai, anthropic
RLM_ENVIRONMENT=local          # local, docker, modal, prime
RLM_VERBOSE=false              # enable verbose RLM logging
```

### 3. Run the Agent

```bash
# Run RLM agent on a specific game
uv run main.py --agent=rlmagent --game=ls20

# Run with debug logging
DEBUG=True uv run main.py --agent=rlmagent --game=ls20
```

## Architecture

### Core Components

1. **RLM Client**: Interfaces with recursive language model backends
2. **Grid Analysis Utilities**: Built-in functions for pattern detection
3. **Prompt Builder**: Creates comprehensive prompts with game context
4. **Result Parser**: Extracts actions from LLM responses with multiple fallback strategies
5. **Memory System**: Maintains episodic memory and hypothesis evolution

### Grid Analysis Functions

The RLM agent includes these pre-loaded utility functions:

- `summarize_grid(grid, 64)` - Comprehensive grid state analysis
- `diff_summary(prev, curr)` - Compact change detection
- `find_player(grid)` - Player character detection
- `find_door(grid)` - Door pattern detection
- `find_key(grid, 64)` - Key pattern detection
- `color_name(val)` - Human-readable color names

### Reasoning Loop

1. **Analyze**: Grid state and changes using utility functions
2. **Hypothesize**: Form theories about game rules and objectives
3. **Act**: Choose action based on current understanding
4. **Observe**: Record results and update hypotheses
5. **Repeat**: Continue recursive reasoning loop

## Supported Backends

### OpenRouter (Recommended)
- Models: `google/gemini-2.5-flash`, `anthropic/claude-3.5-sonnet`, etc.
- Cost-effective with good performance
- Requires: `OPENROUTER_API_KEY`

### OpenAI
- Models: `gpt-4`, `gpt-4-turbo`, etc.
- High-quality reasoning
- Requires: `OPENAI_API_KEY`

### Anthropic
- Models: `claude-3-sonnet-20240229`, etc.
- Strong analytical capabilities
- Requires: `ANTHROPIC_API_KEY`

## Performance Tips

### Model Selection
- **Speed**: `google/gemini-2.5-flash` (recommended)
- **Quality**: `anthropic/claude-3.5-sonnet`
- **Cost**: OpenRouter provides best value

### Optimization
- Enable `RLM_VERBOSE=true` for debugging
- Use `DEBUG=True` for detailed logging
- Monitor `rlm_calls_total` in reasoning metadata
- Adjust memory size if needed (default: 50 entries)

### Troubleshooting

#### Common Issues

1. **Parsing Failures**
   - Check response format in logs
   - Try different models
   - Enable verbose logging

2. **Slow Responses**
   - Use faster models (Gemini 2.5 Flash)
   - Check network connectivity
   - Monitor API rate limits

3. **Stuck Behavior**
   - Agent automatically detects stuck states
   - Takes exploratory actions after 5 unchanged frames
   - Can manually reset with RESET action

#### Debug Mode

```bash
# Enable full debug logging
DEBUG=True RLM_VERBOSE=true uv run main.py --agent=rlmagent --game=ls20
```

## Benchmarking

Run benchmarks to evaluate performance:

```bash
# Quick benchmark
uv run benchmark.py --agents rlmagent --games ls20 --max-actions 20

# Full benchmark
uv run benchmark.py --agents rlmagent --games ls20 --max-actions 80
```

## Contributing

### Development Setup

```bash
# Install development dependencies
uv sync --group dev

# Run tests
uv run pytest tests/unit/test_rlm_agent.py -v

# Run with coverage
uv run pytest tests/unit/test_rlm_agent.py --cov=agents/templates/rlm_agent
```

### Adding New Features

1. **Grid Analysis**: Add new utility functions to `rlm_agent.py`
2. **Backends**: Extend `_build_backend_kwargs()` for new providers
3. **Parsing**: Improve `_extract_result_dict()` for better response handling
4. **Memory**: Enhance `_record_observation()` for better learning

## License

This RLM agent implementation is part of the ARC-AGI-3 Agents project and follows the same license terms.

## Citation

If you use this RLM agent in your research, please cite:

```bibtex
@misc{arc-agi-3-agents,
  title={ARC-AGI-3 Agents: Recursive Language Model Implementation},
  author={ARC Prize Community},
  year={2026},
  url={https://github.com/arcprize/ARC-AGI-3-Agents}
}
```
