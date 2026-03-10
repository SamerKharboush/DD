"""
Agent Orchestrator for Multi-Agent Coordination.

Implements the orchestration layer that:
- Coordinates specialist agents
- Manages shared workspace
- Resolves conflicts between agents
- Synthesizes final conclusions
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ct.agents.base_agent import AgentRole, AgentMessage
from ct.agents.specialist_agents import (
    ChemistAgent,
    BiologistAgent,
    ToxicologistAgent,
    StatisticianAgent,
)
from ct.agents.critic_agent import CriticAgent

logger = logging.getLogger("ct.agents.orchestrator")


class OrchestrationMode(Enum):
    """Mode of agent orchestration."""
    SEQUENTIAL = "sequential"  # Agents run in sequence
    PARALLEL = "parallel"  # Agents run in parallel
    DEBATE = "debate"  # Agents debate until consensus
    HIERARCHICAL = "hierarchical"  # Orchestrator delegates to specialists


@dataclass
class AgentFinding:
    """Finding from a specific agent."""
    agent_role: str
    summary: str
    confidence: float
    details: dict = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


@dataclass
class Conflict:
    """A conflict between agent findings."""
    agents: list[str]
    issue: str
    severity: str  # critical, major, minor
    resolution: Optional[str] = None


@dataclass
class Workspace:
    """Shared workspace for agent collaboration."""
    query: str
    context: dict = field(default_factory=dict)
    findings: dict[str, AgentFinding] = field(default_factory=dict)
    conflicts: list[Conflict] = field(default_factory=list)
    debate_round: int = 0
    consensus_reached: bool = False
    final_conclusion: Optional[str] = None

    def add_finding(self, finding: AgentFinding) -> None:
        """Add an agent finding."""
        self.findings[finding.agent_role] = finding

    def detect_conflicts(self) -> list[Conflict]:
        """Detect conflicts between agent findings."""
        conflicts = []

        # Check for direct contradictions
        roles = list(self.findings.keys())

        for i, role1 in enumerate(roles):
            for role2 in roles[i+1:]:
                finding1 = self.findings[role1]
                finding2 = self.findings[role2]

                # Check for contradictory issues
                for issue in finding1.issues:
                    if any(self._contradicts(issue, i) for i in finding2.recommendations):
                        conflicts.append(Conflict(
                            agents=[role1, role2],
                            issue=f"Contradiction: {issue}",
                            severity="major",
                        ))

        self.conflicts = conflicts
        return conflicts

    def _contradicts(self, statement1: str, statement2: str) -> bool:
        """Check if two statements contradict."""
        # Simple heuristic - look for negation words
        negation_words = ["not", "avoid", "don't", "shouldn't", "cannot"]
        statement1_lower = statement1.lower()
        statement2_lower = statement2.lower()

        has_negation_1 = any(neg in statement1_lower for neg in negation_words)
        has_negation_2 = any(neg in statement2_lower for neg in negation_words)

        # If one has negation and other doesn't on similar topic
        if has_negation_1 != has_negation_2:
            # Check for topic overlap
            words1 = set(statement1_lower.split())
            words2 = set(statement2_lower.split())
            overlap = words1 & words2
            if len(overlap) >= 2:
                return True

        return False


@dataclass
class OrchestrationResult:
    """Result of multi-agent orchestration."""
    query: str
    final_conclusion: str
    agent_findings: dict[str, AgentFinding]
    conflicts_resolved: list[Conflict]
    confidence: float
    total_rounds: int
    total_time_seconds: float
    consensus_reached: bool


class AgentOrchestrator:
    """
    Orchestrates multi-agent collaboration.

    Usage:
        orchestrator = AgentOrchestrator()
        result = orchestrator.run(
            query="Design a KRAS G12C inhibitor with good BBB penetration",
            context={"target": "KRAS_G12C_SEQUENCE"},
            mode=OrchestrationMode.DEBATE,
        )
    """

    ORCHESTRATOR_SYSTEM_PROMPT = """You are the orchestrator for a multi-agent drug discovery team.

Your role is to:
1. Synthesize findings from specialist agents
2. Resolve conflicts between agents
3. Make final decisions when agents disagree
4. Produce coherent, actionable conclusions

SPECIALIST AGENTS:
- Chemist: Molecular design and ADMET optimization
- Biologist: Target biology and pathway analysis
- Toxicologist: Safety assessment (acts as critic)
- Statistician: Data analysis and validation
- Critic: Adversarial review and quality control

CONFLICT RESOLUTION RULES:
1. Safety concerns (Toxicologist) take precedence
2. Biological plausibility (Biologist) grounds chemical designs
3. Statistical rigor (Statistician) validates claims
4. Critic's issues must be addressed before finalizing

SYNTHESIS APPROACH:
1. Summarize each agent's key findings
2. Identify areas of agreement and disagreement
3. Weigh evidence by agent expertise and confidence
4. Make explicit any remaining uncertainties
5. Provide actionable next steps

OUTPUT FORMAT:
- Clear conclusion (2-3 sentences)
- Key supporting evidence (bullet points)
- Remaining uncertainties
- Recommended actions
- Confidence level with justification"""

    def __init__(
        self,
        mode: OrchestrationMode = OrchestrationMode.SEQUENTIAL,
        max_debate_rounds: int = 3,
        consensus_threshold: float = 0.7,
    ):
        """
        Initialize orchestrator.

        Args:
            mode: Orchestration mode
            max_debate_rounds: Maximum rounds in debate mode
            consensus_threshold: Threshold for consensus
        """
        self.mode = mode
        self.max_debate_rounds = max_debate_rounds
        self.consensus_threshold = consensus_threshold

        # Initialize agents
        self.agents = {
            AgentRole.CHEMIST: ChemistAgent(),
            AgentRole.BIOLOGIST: BiologistAgent(),
            AgentRole.TOXICOLOGIST: ToxicologistAgent(),
            AgentRole.STATISTICIAN: StatisticianAgent(),
            AgentRole.CRITIC: CriticAgent(),
        }

        self._llm_client = None

    @property
    def llm_client(self):
        """Lazy-load LLM client."""
        if self._llm_client is None:
            from ct.models.llm import get_llm_client
            self._llm_client = get_llm_client()
        return self._llm_client

    def run(
        self,
        query: str,
        context: Optional[dict] = None,
        agents_to_use: Optional[list[AgentRole]] = None,
    ) -> OrchestrationResult:
        """
        Run multi-agent orchestration.

        Args:
            query: User query
            context: Additional context
            agents_to_use: Specific agents to use (None = all)

        Returns:
            OrchestrationResult
        """
        start_time = time.time()
        context = context or {}

        # Initialize workspace
        workspace = Workspace(query=query, context=context)

        # Determine agents to use
        if agents_to_use is None:
            agents_to_use = [
                AgentRole.CHEMIST,
                AgentRole.BIOLOGIST,
                AgentRole.TOXICOLOGIST,
                AgentRole.CRITIC,
            ]

        # Run based on mode
        if self.mode == OrchestrationMode.SEQUENTIAL:
            self._run_sequential(workspace, agents_to_use)
        elif self.mode == OrchestrationMode.PARALLEL:
            self._run_parallel(workspace, agents_to_use)
        elif self.mode == OrchestrationMode.DEBATE:
            self._run_debate(workspace, agents_to_use)
        elif self.mode == OrchestrationMode.HIERARCHICAL:
            self._run_hierarchical(workspace, agents_to_use)

        # Synthesize final conclusion
        final_conclusion = self._synthesize_conclusion(workspace)

        # Calculate overall confidence
        confidence = self._calculate_overall_confidence(workspace)

        duration = time.time() - start_time

        return OrchestrationResult(
            query=query,
            final_conclusion=final_conclusion,
            agent_findings=workspace.findings,
            conflicts_resolved=[c for c in workspace.conflicts if c.resolution],
            confidence=confidence,
            total_rounds=workspace.debate_round,
            total_time_seconds=duration,
            consensus_reached=workspace.consensus_reached,
        )

    def _run_sequential(
        self,
        workspace: Workspace,
        agents: list[AgentRole],
    ) -> None:
        """Run agents sequentially."""
        for role in agents:
            if role in self.agents:
                agent = self.agents[role]
                response = agent.analyze(workspace.context, self._workspace_to_dict(workspace))

                finding = AgentFinding(
                    agent_role=role.value,
                    summary=response.message.content[:500],
                    confidence=response.message.confidence,
                    details=response.message.findings,
                    issues=response.issues_found,
                    recommendations=response.suggested_actions,
                )
                workspace.add_finding(finding)

        # Detect conflicts after all agents have run
        workspace.detect_conflicts()

        # Resolve conflicts
        self._resolve_conflicts(workspace)

    def _run_parallel(
        self,
        workspace: Workspace,
        agents: list[AgentRole],
    ) -> None:
        """Run agents in parallel (simulated)."""
        # In practice, would use async/threading
        # For now, run sequentially but note it's designed for parallel
        self._run_sequential(workspace, agents)

    def _run_debate(
        self,
        workspace: Workspace,
        agents: list[AgentRole],
    ) -> None:
        """Run agents with debate rounds until consensus."""
        # Initial round
        self._run_sequential(workspace, agents)
        workspace.debate_round = 1

        # Check for consensus
        while not workspace.consensus_reached and workspace.debate_round < self.max_debate_rounds:
            # Run critic to identify issues
            critic_response = self.agents[AgentRole.CRITIC].analyze(
                workspace.context,
                self._workspace_to_dict(workspace),
            )

            if not critic_response.issues_found:
                workspace.consensus_reached = True
                break

            # Re-run agents with critic feedback
            for role in agents:
                if role == AgentRole.CRITIC:
                    continue

                agent = self.agents[role]
                response = agent.analyze(
                    workspace.context,
                    self._workspace_to_dict(workspace),
                )

                # Update finding
                finding = AgentFinding(
                    agent_role=role.value,
                    summary=response.message.content[:500],
                    confidence=response.message.confidence,
                    details=response.message.findings,
                    issues=response.issues_found,
                    recommendations=response.suggested_actions,
                )
                workspace.add_finding(finding)

            workspace.debate_round += 1

        workspace.consensus_reached = True  # After max rounds, force consensus

    def _run_hierarchical(
        self,
        workspace: Workspace,
        agents: list[AgentRole],
    ) -> None:
        """Run with hierarchical delegation."""
        # Orchestrator decides which agents to call
        # For now, simplify to sequential
        self._run_sequential(workspace, agents)

    def _resolve_conflicts(self, workspace: Workspace) -> None:
        """Resolve conflicts between agents."""
        for conflict in workspace.conflicts:
            resolution = self._determine_resolution(conflict, workspace)
            conflict.resolution = resolution

    def _determine_resolution(self, conflict: Conflict, workspace: Workspace) -> str:
        """Determine resolution for a conflict."""
        # Safety takes precedence
        if AgentRole.TOXICOLOGIST.value in conflict.agents:
            return "Toxicologist's safety concern takes precedence"

        # Biological plausibility grounds chemical designs
        if AgentRole.BIOLOGIST.value in conflict.agents and AgentRole.CHEMIST.value in conflict.agents:
            return "Biologist's mechanistic insight informs chemist's design"

        # Default: require verification
        return f"Requires verification: {conflict.issue}"

    def _synthesize_conclusion(self, workspace: Workspace) -> str:
        """Synthesize final conclusion from all findings."""
        # Build summary of findings
        findings_summary = []
        for role, finding in workspace.findings.items():
            findings_summary.append(f"**{role}**: {finding.summary[:200]}")

        # Build prompt for synthesis
        prompt = f"""Synthesize the following agent findings into a coherent conclusion.

Query: {workspace.query}

Agent Findings:
{chr(10).join(findings_summary)}

Conflicts Resolved:
{chr(10).join(f"- {c.issue}: {c.resolution}" for c in workspace.conflicts if c.resolution) if workspace.conflicts else "None"}

Provide:
1. Main conclusion (2-3 sentences)
2. Key evidence supporting the conclusion
3. Remaining uncertainties
4. Recommended next steps"""

        try:
            response = self.llm_client.chat(
                messages=[
                    {"role": "system", "content": self.ORCHESTRATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                model="claude-opus-4-6",
                temperature=0.3,
                max_tokens=1024,
            )
            return response.get("content", "Unable to synthesize conclusion")
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return self._fallback_synthesis(workspace)

    def _fallback_synthesis(self, workspace: Workspace) -> str:
        """Fallback synthesis when LLM fails."""
        summaries = [f.summary for f in workspace.findings.values()]
        return " | ".join(summaries[:3]) if summaries else "No conclusions available"

    def _calculate_overall_confidence(self, workspace: Workspace) -> float:
        """Calculate overall confidence in the result."""
        if not workspace.findings:
            return 0.0

        # Average confidence across agents
        confidences = [f.confidence for f in workspace.findings.values()]
        avg_confidence = sum(confidences) / len(confidences)

        # Penalize for conflicts
        conflict_penalty = len(workspace.conflicts) * 0.1

        # Boost for consensus
        consensus_boost = 0.1 if workspace.consensus_reached else 0

        final_confidence = avg_confidence - conflict_penalty + consensus_boost
        return max(0.1, min(1.0, final_confidence))

    def _workspace_to_dict(self, workspace: Workspace) -> dict:
        """Convert workspace to dictionary for agent context."""
        return {
            "query": workspace.query,
            "context": workspace.context,
            "findings": {
                role: {
                    "summary": f.summary,
                    "confidence": f.confidence,
                    "issues": f.issues,
                }
                for role, f in workspace.findings.items()
            },
            "conflicts": [
                {"agents": c.agents, "issue": c.issue}
                for c in workspace.conflicts
            ],
            "debate_round": workspace.debate_round,
        }

    def get_agent_summary(self) -> dict:
        """Get summary of available agents."""
        return {
            role.value: {
                "tools": agent.get_available_tools(),
                "role": role.value,
            }
            for role, agent in self.agents.items()
        }


def run_multi_agent_analysis(
    query: str,
    context: Optional[dict] = None,
    mode: str = "sequential",
) -> dict:
    """
    Run multi-agent analysis.

    Args:
        query: User query
        context: Additional context
        mode: Orchestration mode (sequential, parallel, debate)

    Returns:
        Dictionary with analysis results
    """
    mode_map = {
        "sequential": OrchestrationMode.SEQUENTIAL,
        "parallel": OrchestrationMode.PARALLEL,
        "debate": OrchestrationMode.DEBATE,
    }

    orchestrator = AgentOrchestrator(
        mode=mode_map.get(mode, OrchestrationMode.SEQUENTIAL),
    )

    result = orchestrator.run(query, context)

    return {
        "summary": result.final_conclusion,
        "query": result.query,
        "confidence": result.confidence,
        "consensus_reached": result.consensus_reached,
        "total_rounds": result.total_rounds,
        "total_time_seconds": result.total_time_seconds,
        "agent_findings": {
            role: {
                "summary": f.summary,
                "confidence": f.confidence,
                "issues": f.issues[:3],
                "recommendations": f.recommendations[:3],
            }
            for role, f in result.agent_findings.items()
        },
        "conflicts_resolved": len(result.conflicts_resolved),
    }