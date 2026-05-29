from typing import Type, cast

from dotenv import load_dotenv

from .agent import Agent, Playback
from .recorder import Recorder
from .swarm import Swarm
from .templates.langgraph_functional_agent import LangGraphFunc, LangGraphTextOnly
from .templates.langgraph_random_agent import LangGraphRandom
from .templates.langgraph_thinking import LangGraphThinking
from .templates.llm_agents import LLM, FastLLM, GuidedLLM, ReasoningLLM
from .templates.multimodal import MultiModalLLM
from .templates.openclaw_agent import OpenClaw
from .templates.random_agent import Random
from .templates.reasoning_agent import ReasoningAgent
from .templates.smolagents import SmolCodingAgent, SmolVisionAgent
from .templates.solver_agent import SolverAgent

load_dotenv()

AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    cls.__name__.lower(): cast(Type[Agent], cls)
    for cls in Agent.__subclasses__()
    if cls.__name__ != "Playback"
}

# add all the recording files as valid agent names
for rec in Recorder.list():
    AVAILABLE_AGENTS[rec] = Playback

# explicitly register agent aliases/overrides for subclasses not auto-discovered
AVAILABLE_AGENTS["reasoningagent"] = ReasoningAgent

__all__ = [
    "Swarm",
    "Random",
    "LangGraphFunc",
    "LangGraphTextOnly",
    "LangGraphThinking",
    "LangGraphRandom",
    "LLM",
    "FastLLM",
    "ReasoningLLM",
    "GuidedLLM",
    "ReasoningAgent",
    "SmolCodingAgent",
    "SmolVisionAgent",
    "Agent",
    "Recorder",
    "Playback",
    "AVAILABLE_AGENTS",
    "MultiModalLLM",
    "SolverAgent",
    "OpenClaw",
]
