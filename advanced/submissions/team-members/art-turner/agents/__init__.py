"""Multi-agent system for ScholarAI Advanced Track."""

from .topic_splitter import TopicSplitterAgent, create_topic_splitter
from .researcher_agent import ResearcherAgent, create_researcher_agent
from .synthesizer_agent import SynthesizerAgent, create_synthesizer_agent
from .critic_agent import CriticAgent, create_critic_agent
from .orchestrator import MultiAgentOrchestrator, create_orchestrator

__all__ = [
    "TopicSplitterAgent",
    "create_topic_splitter",
    "ResearcherAgent",
    "create_researcher_agent",
    "SynthesizerAgent",
    "create_synthesizer_agent",
    "CriticAgent",
    "create_critic_agent",
    "MultiAgentOrchestrator",
    "create_orchestrator",
]
