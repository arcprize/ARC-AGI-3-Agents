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
# 1. install the OpenClaw CLI and onboard with your provider key (one-time)
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

## Notes

- **Why not OpenAI-style `tools`?** OpenClaw's `/v1/chat/completions` endpoint
  silently drops the `tools` field for some backends (verified 2026-05 against
  the Anthropic provider — the upstream model never sees the schema). This
  agent uses a **JSON-in-text protocol** instead: the prompt asks the model to
  reply with one JSON object naming the action, and we parse it from
  `message.content`. Tolerant of stray markdown fences.
- **Session memory.** Each game passes `x-openclaw-session-key:
  arc:<card_id>:<game_id>`. OpenClaw uses this to retain conversation history
  across the game's 80-action budget — the main edge over a stateless LLM call.
- **No new Python deps.** The existing `openai` SDK talks to OpenClaw's
  endpoint directly.
- **Vision.** OpenClaw's compat API does not document image input, so the grid
  is serialized as hex text. If you want a multimodal variant later, you'd add
  `image_url` content blocks and verify the configured underlying model
  accepts them.
