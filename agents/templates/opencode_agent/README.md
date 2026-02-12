## Overview

The OpenCode agent uses OpenRouter (GPT-5.2 by default) to play ARC-AGI-3 puzzle games through a local OpenCode server. The agent uses async messaging with early-abort optimization to minimize latency and cost, while maintaining detailed step recordings with retry tracking.

## Architecture

```
Game Loop (main.py)
       ↓
opencode_agent.py ←→ OpenCode Server (localhost:4096)
       ↓                     ↓
   Async Polling      OpenRouter API
       ↓                     ↓
   Action Detection → Abort Session (early exit)
       ↓
opencode_client.py
    ├─ MessageParser (extract reasoning, tools, cost)
    └─ Session Management
           ↓
    ARC-Game-Tools MCP Server
       ├─ Game Actions (reset, action1-7)
       └─ Memory (read, write, apply_patch for notes)
              ↓
       ./game_notes/{game_id}_{session}/notes.md
    
opencode_recorder.py → ./recordings/{game}_{agent}_{session}/step_XXX.json
```

## Components

### `opencode_agent.py`
Main agent class that manages the game loop with async messaging. Each turn:
1. Builds prompt with current grid state and structured notes instructions
2. Sends async prompt to OpenCode server
3. Polls for messages until game action detected
4. **Aborts session immediately** when action found (saves cost/latency)
5. Extracts reasoning, cost, and token usage from messages
6. Implements **retry mechanism** with feedback on failed actions
7. Records everything to JSON with error tracking

**Key Features:**
- Early abort optimization (stops generation after action detected)
- Retry with feedback (up to 3 attempts if no valid action)
- Cost tracking across multiple assistant messages per step
- Detailed error logging for traceability

### `opencode_client.py`
HTTP client for OpenCode server API:
- **Session management**: create, abort, delete sessions
- **Async messaging**: `send_message_async()` for non-blocking prompts
- **Message polling**: `get_messages()` with limit/slicing
- **MessageParser**: Extracts reasoning (text + reasoning parts), tool calls, and usage info

**Cost Extraction:**
- Parses `message["info"]["cost"]` and `message["info"]["tokens"]`
- Sums costs across multiple assistant messages within a step
- Handles incomplete messages (aborted before completion)

### `opencode_recorder.py`
Records each step to `recordings/{game_id}_{agent}_{session}/step_XXX.json`:
- Initial prompt sent
- Assistant messages (reasoning + tool calls)
- Parsed action with reasoning
- Cost per step (summed from all assistant messages)
- Retry metadata if errors occurred

**Recording includes:**
- `previous_error`: Details of initial failure if retry occurred
- `retry_success`: True if action succeeded after retry
- `retry_attempt`: Attempt number
- `error_feedback_sent`: Feedback message sent to LLM

### Retry Mechanism

When LLM fails to provide valid game action:
1. Log error and record failure details
2. Send error feedback with valid tool list
3. Poll for retry response (up to 3 attempts)
4. If retry succeeds: record with retry metadata
5. If max retries reached: raise RuntimeError

**No fallback actions** - the LLM must recover or the game stops.

## OpenCode Server

**Local server** running at `localhost:4096`:
- Manages conversation state and tool execution
- Routes requests to OpenRouter API
- Returns messages with cost/token information
- Supports session abort for early exit

**Configuration** (`.env`):
- `OPENCODE_SERVER_PORT=4096`
- `OPENROUTER_API_KEY=sk-or-v1-...`
- Model configurable in `opencode_agent.py` (default: `openai/gpt-5.2`)

## Memory Management

**Location**: `./game_notes/{game_id}_{session}/notes.md`

Agent uses standard file tools (read, write, apply_patch) to manage structured notes:
- **Game Mechanics**: Confirmed patterns across levels
- **Hypothesis**: Current theory about level mechanics (with confidence)
- **Key Positions**: Coordinates of important objects
- **Failed Approaches**: What didn't work (avoid repeating)
- **Current Plan**: Immediate action plan with cost estimate

**Note**: Instructions emphasize targeted edits (apply_patch) over full rewrites to maintain structure.

## Early Abort Optimization

**Why polling is necessary**: Unlike Claude's API (which supports `tool_choice` to limit tool calls), OpenCode/OpenRouter **does not provide a mechanism to restrict the LLM to calling exactly one tool**. Without this constraint, the LLM may:
- Call multiple game actions in sequence (corrupts game state)
- Generate extensive reasoning after the action (wastes tokens)
- Continue with unnecessary tool calls (increases cost)

**Solution**: 
1. Send async prompt to OpenCode (non-blocking)
2. Poll messages in real-time (0.1s intervals)
3. When **first game action detected** → abort session immediately
4. Sleep 0.5s to let OpenCode finalize cost calculation
5. Poll once more if cost not available (0.3s additional wait)
6. Extract reasoning + cost from partial response

**Savings**: ~50% faster, ~30% cheaper than waiting for completion.

**Trade-off**: This approach means we may miss reasoning that comes *after* the tool call. The 0.5s post-abort delay helps capture trailing reasoning, but it's not guaranteed.

## Cost Tracking

**Per-step cost** (stored in `step_XXX.json`):
- Sums costs from all assistant messages in that step
- Includes initial attempt + retry costs if applicable
- Extracted from `AssistantMessage.cost` in OpenCode API

**Cumulative cost** (logged during execution):
- Running total across all steps
- Updated after each step completes
- Logged as `${step_cost} / ${cumulative_cost}`

## Prompting Strategy

Each turn sends:
1. Game context (state, levels completed, available actions)
2. Current 64x64 grid visualization
3. MCP tool list (arc-game-tools server)
4. Structured notes instructions with sections
5. **Critical**: Only ONE action per turn (enforced)

**Reset rules** (prevents accidental game restart):
- Don't reset at level start
- Don't reset twice in a row
- Violation → game quit
