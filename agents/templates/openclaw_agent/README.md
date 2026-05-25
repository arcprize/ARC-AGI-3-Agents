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

**Run the gateway via Docker Compose — no host OpenClaw install needed.** The
compose file pulls OpenClaw's published image, keeps config/workspace in a
docker-managed volume, publishes `127.0.0.1:18789`, and **onboards your provider
in-container on first boot**. You just drop a provider key in `.env`.

```bash
# 1. configure the gateway
cd agents/templates/openclaw_agent/docker
cp .env.example .env
# edit docker/.env and set ONE provider key — the first present is onboarded:
#   ANTHROPIC_API_KEY=sk-ant-...     (or OPENAI_API_KEY / GEMINI_API_KEY)
# For the Codex harness instead: set OPENAI_API_KEY + OPENCLAW_USE_CODEX=1
#   (see "Codex harness" below)

# 2. start it — first boot onboards the provider inside the container
docker compose up -d
docker compose logs -f openclaw-gateway   # wait for "[gateway] ready", then Ctrl-C
cd ../../../..

# 3. set up the ARC .env. OPENCLAW_GATEWAY_TOKEN ships as a shared local-dev
#    default already matching docker/.env, so you only set ARC_API_KEY here.
cp .env.example .env
# edit .env and set ARC_API_KEY
```

To reuse an existing host `~/.openclaw` (or onboard on the host yourself), set
`OPENCLAW_HOST_CONFIG_DIR=$HOME/.openclaw` in `docker/.env` to bind-mount it
instead of the managed volume. The gateway can also run natively without Docker
(`npm install -g openclaw@latest && openclaw gateway run --port 18789`).

### Docker Gateway

Prerequisites: Docker Compose v2 and a provider API key in `docker/.env`. The
container onboards on first boot, so no pre-existing OpenClaw config is needed.

```bash
cd agents/templates/openclaw_agent/docker
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY (or OPENAI_API_KEY / GEMINI_API_KEY)
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

### Codex harness (`OPENCLAW_USE_CODEX=1`)

Routes `openai/*` calls through OpenClaw's [Codex harness][codex-harness]
(openai-only; auth via `OPENAI_API_KEY`). Fully container-managed — no host
`openclaw`/`npm` install. Just set the two vars and start:

```bash
cd agents/templates/openclaw_agent/docker
cp .env.example .env        # set OPENAI_API_KEY=sk-proj-... and OPENCLAW_USE_CODEX=1
docker compose up -d
```

On first boot the entrypoint onboards the openai-api-key auth inside the
container (using `OPENAI_API_KEY` and `OPENCLAW_GATEWAY_TOKEN` from `.env`),
then loads the image's **bundled** codex plugin. Because that plugin ships in
the image it's always on the same OpenClaw version as the runtime — avoiding the
load failure a host-installed `@openclaw/codex` newer than the image would cause
(`Cannot find module '…/plugin-sdk/…/codex-mcp-projection'`, after which codex
silently drops out). The repo-root `.env` ships the same default gateway token,
so just set `ARC_API_KEY` (step 3) and run:

```bash
OPENCLAW_MODEL=openai/gpt-5.5 uv run main.py --agent=openclaw --game=ls20
```

[codex-harness]: https://docs.openclaw.ai/plugins/codex-harness

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
