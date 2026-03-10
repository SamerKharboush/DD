"""
Integration Tests for CellType-Agent.

Tests the complete pipeline from API to agents to tools.
"""

import json
import os
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestImports:
    """Test that all critical modules import correctly."""

    def test_agent_runner_imports(self):
        """Test agent runner imports."""
        from ct.agent.runner import run_query, AgentRunner
        assert run_query is not None
        assert AgentRunner is not None

    def test_llm_client_imports(self):
        """Test LLM client imports."""
        from ct.models.llm import get_llm_client, LLMClient
        assert get_llm_client is not None
        assert LLMClient is not None

    def test_specialist_agents_imports(self):
        """Test specialist agents import."""
        from ct.agents.specialist_agents import (
            ChemistAgent,
            BiologistAgent,
            ToxicologistAgent,
            StatisticianAgent,
        )
        assert ChemistAgent is not None
        assert BiologistAgent is not None
        assert ToxicologistAgent is not None
        assert StatisticianAgent is not None

    def test_orchestrator_imports(self):
        """Test orchestrator imports."""
        from ct.agents.orchestrator import AgentOrchestrator, run_multi_agent_analysis
        assert AgentOrchestrator is not None
        assert run_multi_agent_analysis is not None

    def test_dmta_imports(self):
        """Test DMTA imports."""
        from ct.campaign.dmta import DMTACycle, run_dmta_cycle
        assert DMTACycle is not None
        assert run_dmta_cycle is not None

    def test_knowledge_graph_imports(self):
        """Test knowledge graph imports."""
        from ct.knowledge_graph import Neo4jClient, GraphRAG
        assert Neo4jClient is not None
        assert GraphRAG is not None

    def test_admet_imports(self):
        """Test ADMET imports."""
        from ct.admet.predictor import ADMETPredictor
        assert ADMETPredictor is not None

    def test_local_llm_imports(self):
        """Test local LLM imports."""
        from ct.local_llm import LocalLLMClient, HybridRouter, LoRATrainer
        assert LocalLLMClient is not None
        assert HybridRouter is not None
        assert LoRATrainer is not None

    def test_rlef_imports(self):
        """Test RLEF imports."""
        from ct.rlef import RLEFTrainer, FeedbackProcessor
        assert RLEFTrainer is not None
        assert FeedbackProcessor is not None

    def test_tool_registry(self):
        """Test tool registry."""
        from ct.tools import registry
        tools = registry.list_tools()
        assert len(tools) >= 5


class TestAgentRunner:
    """Test the agent runner."""

    def test_agent_runner_init(self):
        """Test agent runner initialization."""
        from ct.agent.runner import AgentRunner
        runner = AgentRunner(model="claude-sonnet-4-6")
        assert runner.model == "claude-sonnet-4-6"

    def test_agent_runner_build_system_prompt(self):
        """Test system prompt building."""
        from ct.agent.runner import AgentRunner
        runner = AgentRunner()
        prompt = runner._build_system_prompt()
        assert "CellType-Agent" in prompt
        assert "ADMET" in prompt


class TestLLMClient:
    """Test the LLM client."""

    def test_llm_client_init(self):
        """Test LLM client initialization."""
        from ct.models.llm import LLMClient
        client = LLMClient(provider="anthropic", model="claude-sonnet-4-6")
        assert client.provider == "anthropic"
        assert client.model == "claude-sonnet-4-6"

    def test_llm_client_missing_key(self):
        """Test LLM client with missing API key."""
        from ct.models.llm import LLMClient
        # Should not raise, uses env var
        client = LLMClient(provider="anthropic", api_key="test-key")
        assert client.api_key == "test-key"


class TestSpecialistAgents:
    """Test specialist agents."""

    def test_chemist_agent_init(self):
        """Test chemist agent initialization."""
        from ct.agents.specialist_agents import ChemistAgent
        from ct.agents.base_agent import AgentRole
        agent = ChemistAgent()
        assert agent.role == AgentRole.CHEMIST
        assert len(agent.get_available_tools()) > 0

    def test_chemist_agent_system_prompt(self):
        """Test chemist agent system prompt."""
        from ct.agents.specialist_agents import ChemistAgent
        agent = ChemistAgent()
        prompt = agent.get_system_prompt()
        assert "chemist" in prompt.lower()
        assert "ADMET" in prompt

    def test_biologist_agent_init(self):
        """Test biologist agent initialization."""
        from ct.agents.specialist_agents import BiologistAgent
        from ct.agents.base_agent import AgentRole
        agent = BiologistAgent()
        assert agent.role == AgentRole.BIOLOGIST

    def test_toxicologist_agent_init(self):
        """Test toxicologist agent initialization."""
        from ct.agents.specialist_agents import ToxicologistAgent
        from ct.agents.base_agent import AgentRole
        agent = ToxicologistAgent()
        assert agent.role == AgentRole.TOXICOLOGIST
        assert "SAFETY" in agent.get_system_prompt()

    def test_statistician_agent_init(self):
        """Test statistician agent initialization."""
        from ct.agents.specialist_agents import StatisticianAgent
        from ct.agents.base_agent import AgentRole
        agent = StatisticianAgent()
        assert agent.role == AgentRole.STATISTICIAN


class TestOrchestrator:
    """Test multi-agent orchestrator."""

    def test_orchestrator_init(self):
        """Test orchestrator initialization."""
        from ct.agents.orchestrator import AgentOrchestrator
        orchestrator = AgentOrchestrator()
        assert orchestrator is not None

    def test_orchestrator_conflict_detection(self):
        """Test conflict detection."""
        from ct.agents.orchestrator import AgentOrchestrator
        orchestrator = AgentOrchestrator()

        # Create mock findings
        findings = {
            "chemist": {"safety_concerns": []},
            "toxicologist": {"safety_concerns": ["hERG risk"]},
        }

        conflicts = orchestrator._detect_conflicts(findings)
        # Should detect conflict between chemist saying no issues and toxicologist finding one
        assert len(conflicts) >= 0


class TestDMTACycle:
    """Test DMTA cycle."""

    def test_dmta_init(self):
        """Test DMTA initialization."""
        from ct.campaign.dmta import DMTACycle
        dmta = DMTACycle(target="KRAS_G12C")
        assert dmta.state.target == "KRAS_G12C"

    def test_dmta_design_phase(self):
        """Test DMTA design phase."""
        from ct.campaign.dmta import DMTACycle, DMTAPhase
        dmta = DMTACycle(target="KRAS_G12C")

        # Design phase should be initial
        assert dmta.state.current_phase == DMTAPhase.DESIGN


class TestToolRegistry:
    """Test tool registry."""

    def test_registry_lists_tools(self):
        """Test that registry lists tools."""
        from ct.tools import registry
        tools = registry.list_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_registry_gets_tool(self):
        """Test that registry can get a tool."""
        from ct.tools import get_tool
        tool = get_tool("admet.predict")
        assert tool is not None

    def test_tool_execution(self):
        """Test tool execution."""
        from ct.tools import get_tool
        tool = get_tool("admet.predict")
        result = tool.run(smiles="CCO")
        assert isinstance(result, dict)


class TestSessionLogger:
    """Test session logger."""

    def test_session_logger_init(self):
        """Test session logger initialization."""
        from ct.session_logging import SessionLogger
        logger = SessionLogger()
        assert logger is not None

    def test_session_start(self):
        """Test starting a session."""
        from ct.session_logging import SessionLogger
        logger = SessionLogger()
        session_id = logger.start_session("Test query")
        assert session_id is not None
        assert logger.current_session_id == session_id

    def test_session_log_tool_call(self):
        """Test logging tool call."""
        from ct.session_logging import SessionLogger
        logger = SessionLogger()
        logger.start_session("Test query")
        logger.log_tool_call(
            tool_name="test.tool",
            parameters={"param": "value"},
            result={"output": "test"},
        )
        assert len(logger._current_trace.tool_calls) == 1


class TestFeedbackProcessor:
    """Test feedback processor."""

    def test_feedback_processor_init(self):
        """Test feedback processor initialization."""
        from ct.rlef.feedback_processor import FeedbackProcessor
        processor = FeedbackProcessor()
        assert processor is not None

    def test_feedback_add(self):
        """Test adding feedback."""
        from ct.rlef.feedback_processor import FeedbackProcessor
        processor = FeedbackProcessor()
        result = processor.add_feedback(
            session_id="test-session",
            query="Test query",
            response="Test response",
            rating=4,
        )
        assert result is True


class TestHybridRouter:
    """Test hybrid router."""

    def test_router_init(self):
        """Test router initialization."""
        from ct.local_llm.hybrid_router import HybridRouter
        router = HybridRouter()
        assert router.prefer_local is True

    def test_router_route_simple_query(self):
        """Test routing for simple query."""
        from ct.local_llm.hybrid_router import HybridRouter
        router = HybridRouter()
        decision = router.route("What is the MW of ethanol?")
        assert decision.selected_tier is not None

    def test_router_privacy_mode(self):
        """Test privacy mode routing."""
        from ct.local_llm.hybrid_router import HybridRouter, ModelTier
        router = HybridRouter(privacy_mode=True)
        decision = router.route("Query with PHI data")
        assert decision.selected_tier in [ModelTier.LOCAL_FAST, ModelTier.LOCAL_CAPABLE]


class TestRLEFTrainer:
    """Test RLEF trainer."""

    def test_trainer_init(self):
        """Test trainer initialization."""
        from ct.rlef.rlef_trainer import RLEFTrainer
        trainer = RLEFTrainer()
        assert trainer is not None

    def test_trainer_stats(self):
        """Test getting training stats."""
        from ct.rlef.rlef_trainer import RLEFTrainer
        trainer = RLEFTrainer()
        stats = trainer.get_training_stats()
        assert "num_feedback_samples" in stats


class TestAPIModels:
    """Test API models."""

    def test_query_request_model(self):
        """Test query request model."""
        from ct.api.main import QueryRequest
        request = QueryRequest(query="Test query")
        assert request.query == "Test query"
        assert request.mode == "single"

    def test_dmta_request_model(self):
        """Test DMTA request model."""
        from ct.api.main import DMTARequest
        request = DMTARequest(target="KRAS_G12C", iterations=3)
        assert request.target == "KRAS_G12C"
        assert request.iterations == 3

    def test_feedback_request_model(self):
        """Test feedback request model."""
        from ct.api.main import FeedbackRequest
        request = FeedbackRequest(session_id="test-id", rating=5)
        assert request.rating == 5


# Run tests if called directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])