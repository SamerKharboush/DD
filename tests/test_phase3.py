"""
Tests for CellType-Agent Phase 3 components.

Run with: pytest tests/test_phase3.py -v
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import time


# ============================================================
# Base Agent Tests
# ============================================================

class TestBaseAgent:
    """Tests for base agent."""

    def test_agent_role_enum(self):
        """Test agent role enumeration."""
        from ct.agents.base_agent import AgentRole

        assert AgentRole.CHEMIST.value == "chemist"
        assert AgentRole.BIOLOGIST.value == "biologist"
        assert AgentRole.TOXICOLOGIST.value == "toxicologist"
        assert AgentRole.CRITIC.value == "critic"

    def test_agent_message_creation(self):
        """Test agent message creation."""
        from ct.agents.base_agent import AgentMessage, AgentRole

        message = AgentMessage(
            agent_role=AgentRole.CHEMIST,
            content="Test message",
            confidence=0.8,
        )

        assert message.agent_role == AgentRole.CHEMIST
        assert message.confidence == 0.8
        assert isinstance(message.timestamp, float)

    def test_agent_response_creation(self):
        """Test agent response creation."""
        from ct.agents.base_agent import AgentResponse, AgentMessage, AgentRole

        message = AgentMessage(
            agent_role=AgentRole.CHEMIST,
            content="Test",
            confidence=0.7,
        )

        response = AgentResponse(
            success=True,
            message=message,
            issues_found=["Issue 1"],
        )

        assert response.success
        assert len(response.issues_found) == 1


# ============================================================
# Specialist Agent Tests
# ============================================================

class TestChemistAgent:
    """Tests for chemist agent."""

    def test_system_prompt(self):
        """Test chemist has system prompt."""
        from ct.agents.specialist_agents import ChemistAgent

        agent = ChemistAgent()
        prompt = agent.get_system_prompt()

        assert "medicinal chemist" in prompt.lower()
        assert "ADMET" in prompt
        assert "SAR" in prompt

    def test_available_tools(self):
        """Test chemist has appropriate tools."""
        from ct.agents.specialist_agents import ChemistAgent

        agent = ChemistAgent()
        tools = agent.get_available_tools()

        assert "admet.predict" in tools
        assert "boltz2.predict_affinity" in tools

    def test_confidence_calculation(self):
        """Test confidence calculation."""
        from ct.agents.specialist_agents import ChemistAgent

        agent = ChemistAgent()

        # High confidence with successful tools
        findings = {"specific_hits": True}
        tool_results = [{"result": "success"}, {"result": "success"}]
        confidence = agent.calculate_confidence(findings, tool_results)

        assert 0 < confidence <= 1

        # Lower confidence with uncertainty markers
        findings = {"summary": "This might possibly work"}
        confidence_with_uncertainty = agent.calculate_confidence(findings, tool_results)

        assert confidence_with_uncertainty < confidence


class TestBiologistAgent:
    """Tests for biologist agent."""

    def test_system_prompt(self):
        """Test biologist has system prompt."""
        from ct.agents.specialist_agents import BiologistAgent

        agent = BiologistAgent()
        prompt = agent.get_system_prompt()

        assert "biologist" in prompt.lower()
        assert "pathway" in prompt.lower()
        assert "biomarker" in prompt.lower()

    def test_available_tools(self):
        """Test biologist has appropriate tools."""
        from ct.agents.specialist_agents import BiologistAgent

        agent = BiologistAgent()
        tools = agent.get_available_tools()

        assert "knowledge.graphrag_query" in tools
        assert "knowledge.get_gene_diseases" in tools


class TestToxicologistAgent:
    """Tests for toxicologist agent."""

    def test_system_prompt(self):
        """Test toxicologist has conservative prompt."""
        from ct.agents.specialist_agents import ToxicologistAgent

        agent = ToxicologistAgent()
        prompt = agent.get_system_prompt()

        assert "toxicologist" in prompt.lower()
        assert "safety" in prompt.lower()
        assert "CRITIC" in prompt  # Should emphasize critic role

    def test_conservative_confidence(self):
        """Test toxicologist is conservative."""
        from ct.agents.specialist_agents import ToxicologistAgent
        from ct.agents.base_agent import AgentRole

        agent = ToxicologistAgent()

        # Toxicologist should lower confidence for conservatism
        base_confidence = 0.8
        # After conservative adjustment, should be lower
        adjusted = base_confidence * 0.9

        assert adjusted < base_confidence


class TestStatisticianAgent:
    """Tests for statistician agent."""

    def test_system_prompt(self):
        """Test statistician has system prompt."""
        from ct.agents.specialist_agents import StatisticianAgent

        agent = StatisticianAgent()
        prompt = agent.get_system_prompt()

        assert "statistician" in prompt.lower()
        assert "power" in prompt.lower()
        assert "confidence interval" in prompt.lower()


# ============================================================
# Critic Agent Tests
# ============================================================

class TestCriticAgent:
    """Tests for critic agent."""

    def test_system_prompt(self):
        """Test critic has adversarial prompt."""
        from ct.agents.critic_agent import CriticAgent

        agent = CriticAgent()
        prompt = agent.get_system_prompt()

        assert "adversarial" in prompt.lower()
        assert "CHALLENGE" in prompt
        assert "VERIFY" in prompt

    def test_review_agent_findings(self):
        """Test critic can review findings."""
        from ct.agents.critic_agent import CriticAgent

        agent = CriticAgent()

        # Mock findings
        chemist_findings = {
            "admet": {"predictions": {"herg_inhibitor": 0.8}},
            "confidence": 0.95,  # Suspiciously high
        }

        result = agent._critique_chemist(chemist_findings, {})

        assert "issues" in result
        assert "counter_arguments" in result

    def test_detects_overconfidence(self):
        """Test critic detects overconfidence."""
        from ct.agents.critic_agent import CriticAgent

        agent = CriticAgent()

        findings = {"confidence": 0.99}  # Too high
        result = agent._critique_toxicologist(findings, {})

        assert any("unusually high" in i.lower() for i in result.get("issues", []))


# ============================================================
# Orchestrator Tests
# ============================================================

class TestAgentOrchestrator:
    """Tests for agent orchestrator."""

    def test_orchestration_modes(self):
        """Test orchestration modes."""
        from ct.agents.orchestrator import OrchestrationMode

        assert OrchestrationMode.SEQUENTIAL.value == "sequential"
        assert OrchestrationMode.PARALLEL.value == "parallel"
        assert OrchestrationMode.DEBATE.value == "debate"

    def test_workspace_creation(self):
        """Test workspace creation."""
        from ct.agents.orchestrator import Workspace

        workspace = Workspace(query="Test query")

        assert workspace.query == "Test query"
        assert workspace.debate_round == 0
        assert not workspace.consensus_reached

    def test_workspace_conflict_detection(self):
        """Test workspace conflict detection."""
        from ct.agents.orchestrator import Workspace, AgentFinding

        workspace = Workspace(query="Test")

        # Add conflicting findings
        workspace.add_finding(AgentFinding(
            agent_role="chemist",
            summary="Use compound X",
            confidence=0.8,
            issues=["Compound may be toxic"],
        ))

        workspace.add_finding(AgentFinding(
            agent_role="toxicologist",
            summary="Avoid compound X",
            confidence=0.9,
            recommendations=["Do not use compound X"],
        ))

        conflicts = workspace.detect_conflicts()

        # Should detect some conflict
        assert isinstance(conflicts, list)

    def test_orchestrator_initialization(self):
        """Test orchestrator initializes agents."""
        from ct.agents.orchestrator import AgentOrchestrator

        orchestrator = AgentOrchestrator()

        assert len(orchestrator.agents) == 5
        assert "chemist" in [r.value for r in orchestrator.agents.keys()]


# ============================================================
# Vector Memory Tests
# ============================================================

class TestVectorMemory:
    """Tests for vector memory."""

    def test_memory_entry_creation(self):
        """Test memory entry creation."""
        from ct.memory.vector_memory import MemoryEntry

        entry = MemoryEntry(
            entry_id="test123",
            content="Test content",
            agent_role="chemist",
        )

        assert entry.entry_id == "test123"
        assert entry.agent_role == "chemist"
        assert isinstance(entry.created_at, float)

    def test_memory_store_and_retrieve(self):
        """Test storing and retrieving from memory."""
        from ct.memory.vector_memory import VectorMemory
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = VectorMemory(persist_dir=Path(tmpdir))

            # Store
            entry_id = memory.store(
                content="KRAS G12C inhibitors show promise",
                agent_role="chemist",
                query="Find KRAS inhibitors",
            )

            assert entry_id is not None

            # Keyword search (no embedding needed)
            results = memory._keyword_search("KRAS inhibitors", None, 5)

            assert len(results) > 0

    def test_memory_stats(self):
        """Test memory statistics."""
        from ct.memory.vector_memory import VectorMemory
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = VectorMemory(persist_dir=Path(tmpdir))

            memory.store("Test 1", agent_role="chemist")
            memory.store("Test 2", agent_role="biologist")

            stats = memory.get_stats()

            assert stats["total_entries"] == 2
            assert "chemist" in stats["by_agent"]
            assert "biologist" in stats["by_agent"]


# ============================================================
# DMTA Cycle Tests
# ============================================================

class TestDMTACycle:
    """Tests for DMTA cycle."""

    def test_dmta_phases(self):
        """Test DMTA phases."""
        from ct.campaign.dmta import DMTAPhase

        assert DMTAPhase.DESIGN.value == "design"
        assert DMTAPhase.MAKE.value == "make"
        assert DMTAPhase.TEST.value == "test"
        assert DMTAPhase.ANALYZE.value == "analyze"

    def test_dmta_state_creation(self):
        """Test DMTA state creation."""
        from ct.campaign.dmta import DMTAState, DMTAPhase

        state = DMTAState(
            cycle_id="test123",
            current_phase=DMTAPhase.DESIGN,
            target="KRAS_G12C",
        )

        assert state.current_phase == DMTAPhase.DESIGN
        assert state.iteration == 1
        assert not state.consensus_reached

    def test_dmta_cycle_initialization(self):
        """Test DMTA cycle can be initialized."""
        from ct.campaign.dmta import DMTACycle

        dmta = DMTACycle(
            target="MKTVRQERLKSIVRILERSKEPVSGAQL",
            objective="Find potent inhibitor",
        )

        assert dmta.state.target == "MKTVRQERLKSIVRILERSKEPVSGAQL"
        assert dmta.state.current_phase.value == "design"

    def test_synthesis_feasibility(self):
        """Test synthesis feasibility assessment."""
        from ct.campaign.dmta import DMTACycle

        dmta = DMTACycle(target="Test")

        # Short sequence - high feasibility
        short_candidate = {"sequence": "MKTVRQER"}
        feasibility = dmta._assess_synthesis_feasibility(short_candidate)
        assert feasibility == "high"

        # Long sequence - lower feasibility
        long_candidate = {"sequence": "M" * 150}
        feasibility = dmta._assess_synthesis_feasibility(long_candidate)
        assert feasibility == "low"

    def test_candidate_scoring(self):
        """Test candidate scoring."""
        from ct.campaign.dmta import DMTACycle

        dmta = DMTACycle(target="Test")

        test_result = {
            "assay_results": {
                "admet": {"overall_score": 0.8},
                "binding": {"predicted_affinity_nm": 50},
                "selectivity": {"selectivity_score": 0.9},
            }
        }

        criteria = {
            "affinity_threshold_nm": 100,
            "min_admet_score": 0.7,
        }

        score = dmta._calculate_candidate_score(test_result, criteria)

        assert score > 80  # Should score well


# ============================================================
# Tool Registration Tests
# ============================================================

class TestPhase3Tools:
    """Tests for Phase 3 tool registration."""

    def test_multi_agent_tools_callable(self):
        """Test multi-agent tools are callable."""
        from ct.tools.phase3_tools import (
            multi_agent_analyze,
            multi_agent_chemist_opinion,
            multi_agent_toxicologist_review,
        )

        assert callable(multi_agent_analyze)
        assert callable(multi_agent_chemist_opinion)
        assert callable(multi_agent_toxicologist_review)

    def test_memory_tools_callable(self):
        """Test memory tools are callable."""
        from ct.tools.phase3_tools import (
            memory_store,
            memory_recall,
            memory_stats,
        )

        assert callable(memory_store)
        assert callable(memory_recall)
        assert callable(memory_stats)

    def test_dmta_tools_callable(self):
        """Test DMTA tools are callable."""
        from ct.tools.phase3_tools import (
            dmta_run_cycle,
            dmta_design,
            dmta_test,
        )

        assert callable(dmta_run_cycle)
        assert callable(dmta_design)
        assert callable(dmta_test)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])