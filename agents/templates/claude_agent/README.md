## Overview

The Claude agent uses Anthropic's Claude Sonnet 4.5 model to play ARC-AGI-3 puzzle games through tool-calling and persistent session management. The agent maintains context across turns using session resumption and can store long-term insights in persistent notes files.

## Architecture

```
Game Loop (main.py)
       ↓
claude_agents.py ←→ Claude API (Sonnet 4.5, bypassPermissions)
       ↓
claude_tools.py (MCP Server)
    ├─ Game Actions (reset, action1-7)
    └─ Memory (read_notes, write_notes)
           ↓
    ./game_notes/{game_id}_notes.md
    
claude_recorder.py → ./recordings/{game}_{agent}/step_XXX.json
```

## Components

### `claude_agents.py`
Main agent class that manages the game loop. Each turn:
1. Builds prompt with current grid state
2. Calls Claude API with session resumption
3. Captures reasoning and tool calls
4. Parses action from tool response
5. Records everything for playback

### `claude_tools.py`
MCP server exposing game actions as tools:
- **Game actions**: `reset_game`, `action1-7` (move, interact, click, undo)
- **Memory tools**: `read_notes`, `write_notes` for persistent learning
- Validates actions before execution
- Returns game state feedback

### `claude_recorder.py`
Records each step to `recordings/{game_id}_{agent}/step_XXX.json`:
- Prompts sent to Claude
- Reasoning and tool calls
- Token usage and API costs
- Timestamps

Enables playback and analysis.

## Permissions

**Mode**: `bypassPermissions` (line 162 in `claude_agents.py`)
- Claude has unrestricted access to all MCP tools
- No user confirmation required
- Tools are scoped to game actions and notes directory only

## Persistent Memory

**Location**: `./game_notes/{game_id}_notes.md`

Claude can read/write notes via MCP tools to maintain memory across turns:
- Track patterns and strategies
- Remember what works/doesn't work
- Build hypotheses about puzzle mechanics
- `write_notes` overwrites entire file (full replacement)

## Prompting Strategy

Each turn sends Claude:
1. Game context (state, levels completed, available actions)
2. Current 64x64 grid visualization (values 0-15)
3. Tool list and instructions
4. Emphasis on using persistent notes

**Session continuity**: `session_id` is maintained across turns so Claude keeps full conversation history without re-explaining context.
