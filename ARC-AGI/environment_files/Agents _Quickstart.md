> ## Documentation Index
> Fetch the complete documentation index at: https://docs.arcprize.org/llms.txt
> Use this file to discover all available pages before exploring further.

> # Agents Quickstart

> Build AI agents for ARC-AGI-3 environments

Get started with the [ARC-AGI-3 Agents repo](https://github.com/arcprize/ARC-AGI-3-Agents).

Watch the [Agent Quickstart tutorial video](https://www.youtube.com/watch?v=xEVg9dcJMkw).

> ## Step 1: Clone the Repo

```bash theme={null}
git clone https://github.com/arcprize/ARC-AGI-3-Agents.git
cd ARC-AGI-3-Agents
```

Make sure you have your `ARC_API_KEY` populated in your environment variables. See [API Keys](/api-keys) for setup instructions.

> ## Step 2: Run an Agent

Run the random agent on the `ls20` game:

```bash theme={null}
uv run main.py --agent=random --game=ls20
```

The [replay](/recordings) of your agent is available at the end of the run.

## Next Steps

* [Create Your Own Agent](/create-agent) — Build a custom agent
* [LLM Agents](/llm_agents) — Build agents powered by LLMs
* [Partner Templates](/partner_templates/agentops) — Start from partner-provided templates
