> # ARC-AGI-3-Agents

> ## Quickstart

> Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if not already installed.

1. Clone the ARC-AGI-3-Agents repo and enter the directory.

```bash
git clone https://github.com/arcprize/ARC-AGI-3-Agents.git
cd ARC-AGI-3-Agents
```

2. Copy `.env-example` to `.env` and add your API keys.

```bash
cp .env-example .env
```

3. Get an API key from the
4. >  [ARC-AGI-3 website](https://three.arcprize.org/) and set it in `.env` (or export it):

```shell
export ARC_API_KEY="KGAT_9403c244a51a06c4e57a2ce504923339"
echo 'ARC_API_KEY=your-api-key-here' > .env
```
> ## agentops
```bash
export KAGGLE_API_TOKEN=KGAT_9403c244a51a06c4e57a2ce504923339
```

4. Run the random agent against the locksmith game:

```bash
uv run main.py --agent=random --game=locksmith
```

For more details see the full docs at <https://three.arcprize.org/docs>.

---

> ## Observability (Optional)

[AgentOps](https://agentops.ai/) provides real-time monitoring and debugging.

```bash
# Install the optional extra
uv sync --extra agentops

pip install ruff pre-commit
pre-commit install

# Or with pip
pip install -U agentops
```

> Add your key to `.env`:

```env
AGENTOPS_API_KEY="b51954a5-c92d-4055-a796-2a3ef528700e"
```

Run an agent and you’ll get a session replay link in the console.

---

> ## Contest Submission

Submit your agent via this form: <https://forms.gle/wMLZrEFGDh33DhzV9>

---

> ## Contributing

1. Fork the repo & create a branch.
2. Make changes and ensure tests pass.
3. Set up `ruff` pre-commit hooks for linting & formatting:

```bash
   pip install pre-commit
   pre-commit install
```
4. Commit with clear messages and open a PR.

Feel free to open an issue if you need help.

---

> ## Tests

Run the Python test-suite with `pytest`:

```bash
pytest
```

See <https://three.arcprize.org/docs#testing> for more info.

---

> ## Agent System

The core agent framework lives in [`agents/`](https://github.com/arcprize/ARC-AGI-3-Agents/tree/main/agents). It contains:

- `agent.py` – the base `Agent` class and utilities.
- `swarm.py` – orchestration for running many agents in parallel.
- `recorder.py` – JSONL gameplay recording utilities.
- `structs.py` – typed data structures (`FrameData`, `GameAction`, etc.).

See the concise README in that folder or jump straight to the online docs → > [Agent Quick-Start](https://three.arcprize.org/docs#agent-quickstart).

> ## Agent Templates

Ready-made templates live in [`agents/templates/`](https://github.com/arcprize/ARC-AGI-3-Agents/tree/main/agents/templates):

• **Starter templates** – random agents, basic LLM agents, guided & reasoning agents.
• **Third-party integrations** – HuggingFace *smolagents*, AgentOps tracing & reasoning agent, LangChain/LangGraph agents.

Browse them on GitHub or read the docs:
- Standard templates: <https://three.arcprize.org/docs#building-agents>
> - Third-party templates: <https://three.arcprize.org/docs#third-party-templates>

---

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
