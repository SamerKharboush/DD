"""
Multi-Agent Orchestration for CellType-Agent Phase 3.

Implements specialist agents with adversarial critique:
- Executor agents (chemist, biologist, etc.)
- Critic agent for adversarial review
- Orchestrator for coordination
"""

from ct.agents.base_agent import BaseAgent
from ct.agents.specialist_agents import (
    ChemistAgent,
    BiologistAgent,
    ToxicologistAgent,
    StatisticianAgent,
)
from ct.agents.critic_agent import CriticAgent
from ct.agents.orchestrator import AgentOrchestrator

__all__ = [
    "BaseAgent",
    "ChemistAgent",
    "BiologistAgent",
    "ToxicologistAgent",
    "StatisticianAgent",
    "CriticAgent",
    "AgentOrchestrator",
]