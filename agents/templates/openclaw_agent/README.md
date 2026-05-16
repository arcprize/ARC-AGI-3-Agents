# OpenClaw Agent

Routes ARC-AGI-3 actions through a local [OpenClaw](https://openclaw.ai/) Gateway
via its OpenAI-compatible HTTP API.

The Python class is a thin shim: the actual agent loop runs in the OpenClaw
daemon (Node, BYO LLM key). This satisfies "doesn't have to be Python only"
while keeping the agent runnable from the existing `Swarm` and agent router.

## How it fits

```
main.py --agent=openclaw --game=ls20
        │
        ▼
   Swarm (agents/swarm.py)
        │  one thread per game
        ▼
   OpenClaw (this file)
        │  HTTPS /v1/chat/completions
        ▼
   OpenClaw Gateway (Node, localhost:18789)
        │
        ▼
   Anthropic / OpenAI / Gemini / Ollama  (BYO key)
```

## Setup

**Recommended: run the gateway via Docker Compose.** The compose file uses
OpenClaw's published image, bind-mounts your host `~/.openclaw` config/workspace,
publishes `127.0.0.1:18789`, and rewrites the workspace path inside the
container so host-side OpenClaw onboarding works unchanged.

```bash
# 1. install the OpenClaw CLI and onboard with your provider key (one-time).
# Use --auth-choice openai-api-key (+ --openai-api-key sk-proj-...) instead
# if you plan to set OPENCLAW_USE_CODEX=1 in docker/.env; that toggle routes
# openai/* requests through OpenClaw's Codex harness.
npm install -g openclaw@latest
openclaw onboard --non-interactive --accept-risk \
  --auth-choice anthropic-api-key --anthropic-api-key sk-ant-...
openclaw config set gateway.http.endpoints.chatCompletions.enabled true
openclaw config set agents.defaults.model.primary anthropic/claude-haiku-4-5

# 2. start the gateway in a container
cd agents/templates/openclaw_agent/docker
cp .env.example .env  # then edit and set ANTHROPIC_API_KEY
docker compose up -d
cd ../../../..

# 3. wire the gateway token into the ARC .env (also has ARC_API_KEY)
cp .env.example .env
# edit .env and set OPENCLAW_GATEWAY_TOKEN to the value from:
#   python3 -c "import json; print(json.load(open('$HOME/.openclaw/openclaw.json'))['gateway']['auth']['token'])"
```

If you'd rather skip Docker, the gateway also runs natively
(`openclaw gateway run --port 18789`).

### Docker Gateway

Prerequisites: Docker Compose v2, an OpenClaw-supported provider key, and an
OpenClaw config created by the setup commands above.

```bash
cd agents/templates/openclaw_agent/docker
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY or OPENAI_API_KEY
docker compose up -d
docker compose logs -f openclaw-gateway
```

Verify the gateway:

```bash
curl -sf http://127.0.0.1:18789/healthz
TOKEN=$(python3 -c "import json; print(json.load(open('$HOME/.openclaw/openclaw.json'))['gateway']['auth']['token'])")
curl -sf -X POST -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  http://127.0.0.1:18789/v1/chat/completions \
  -d '{"model":"openclaw/default","messages":[{"role":"user","content":"reply OK"}]}'
```

Stop it with:

```bash
docker compose down
```

## Run

```bash
uv sync
uv run main.py --agent=openclaw --game=ls20
```

### Selecting the underlying model

By default the gateway uses whatever you set at
`agents.defaults.model.primary` in `~/.openclaw/openclaw.json` during
onboarding. To compare providers without editing that file, export
`OPENCLAW_MODEL` before running — the agent forwards it as the documented
`x-openclaw-model` header on each request:

```bash
OPENCLAW_MODEL=anthropic/claude-opus-4-7 uv run main.py --agent=openclaw --game=ls20
OPENCLAW_MODEL=openai/gpt-5              uv run main.py --agent=openclaw --game=ls20
OPENCLAW_MODEL=google/gemini-2.5-pro     uv run main.py --agent=openclaw --game=ls20
```

`OPENCLAW_AGENT` (default `openclaw/default`) selects the OpenClaw *agent
slug* (which tools/prompts it uses); `OPENCLAW_MODEL` overrides the
underlying *provider model* for that agent. The override is also folded
into the agent's recorder subdirectory name so per-model traces don't
collide. To enable a provider, drop its key into `docker/.env`
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`) and
`docker compose up -d --force-recreate`. The compose startup strips the
daemon's per-agent model allowlist so any provider with a key is
callable — no manual config edits needed.

## Notes

- **Why not OpenAI-style `tools`?** OpenClaw's `/v1/chat/completions` endpoint
  silently drops the `tools` field for some backends (verified 2026-05 against
  the Anthropic provider — the upstream model never sees the schema). This
  agent uses a **JSON-in-text protocol** instead: the prompt asks the model to
  reply with one JSON object naming the action, and we parse it from
  `message.content`. Tolerant of stray markdown fences.
- **Session memory.** Each game passes `x-openclaw-session-key:
  arc:<card_id>:<game_id>:<run-id>`. OpenClaw retains conversation history
  across the game's 80-action budget under that key — the main edge over a
  stateless LLM call. The `<run-id>` suffix is a random per-process value
  (overridable with `OPENCLAW_RUN_ID=<name>`), so each fresh `uv run main.py`
  starts with blank server-side memory while turns within one run still
  share state. Old sessions accumulate server-side; periodically run
  `openclaw sessions cleanup --enforce` to evict them.
- **No new Python deps.** The existing `openai` SDK talks to OpenClaw's
  endpoint directly.
- **Vision.** OpenClaw's compat API does not document image input, so the grid
  is serialized as hex text. If you want a multimodal variant later, you'd add
  `image_url` content blocks and verify the configured underlying model
  accepts them.

## Reasoning fields

Each turn's JSON reply must include the four fields described in the [ARC
toolkit reasoning-logs docs][reasoning-docs] alongside the action:

```json
{
  "action": "ACTION1",
  "thought": "Player is below the door; moving up should advance.",
  "confidence": 0.8,
  "alternatives_considered": ["ACTION4 to test right wall"]
}
```

`alternatives_considered` is clipped to 5 items × 200 chars each, and
`confidence` is clamped to `[0,1]`. `thought` passes through verbatim; the
agent only trims it if the full JSON payload would exceed `arcengine`'s 16 KB
cap (`MAX_REASONING_BYTES`), preserving as much justification as possible for
trace analysis.

`reasoning_tokens` reads `response.usage.completion_tokens_details.reasoning_tokens`.
OpenClaw's compat layer doesn't surface that telemetry today (verified against
v2026.5.7's `normalizeUsage`, which has no reasoning/thinking slot), so the
field reports `0` for OpenClaw replies. It will populate automatically once
the gateway forwards the upstream provider's reasoning-token count.

[reasoning-docs]: https://docs.arcprize.org/toolkit/submit-action#including-reasoning-logs
