"""
Base Agent Class for CellType-Agent Multi-Agent System.

Provides the foundation for specialist agents with:
- Tool access
- Memory integration
- Message handling
- Confidence scoring
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger("ct.agents.base")


class AgentRole(Enum):
    """Agent role types."""
    ORCHESTRATOR = "orchestrator"
    CHEMIST = "chemist"
    BIOLOGIST = "biologist"
    TOXICOLOGIST = "toxicologist"
    STATISTICIAN = "statistician"
    CRITIC = "critic"


@dataclass
class AgentMessage:
    """A message from an agent."""
    agent_role: AgentRole
    content: str
    confidence: float
    tool_calls: list = field(default_factory=list)
    findings: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentResponse:
    """Response from an agent."""
    success: bool
    message: AgentMessage
    suggested_actions: list[str] = field(default_factory=list)
    issues_found: list[str] = field(default_factory=list)


class BaseAgent(ABC):
    """
    Base class for all specialist agents.

    Each agent has:
    - A specific domain expertise
    - Access to relevant tools
    - Memory for context
    - Confidence scoring

    Subclasses must implement:
    - analyze(): Main analysis method
    - get_system_prompt(): Agent-specific system prompt
    """

    def __init__(
        self,
        role: AgentRole,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ):
        """
        Initialize base agent.

        Args:
            role: Agent role/type
            model: LLM model to use
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
        """
        self.role = role
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._llm_client = None
        self._memory = None
        self._tool_registry = None

    @property
    def llm_client(self):
        """Lazy-load LLM client."""
        if self._llm_client is None:
            from ct.models.llm import get_llm_client
            self._llm_client = get_llm_client()
        return self._llm_client

    @property
    def memory(self):
        """Lazy-load memory."""
        if self._memory is None:
            from ct.memory.vector_memory import get_agent_memory
            self._memory = get_agent_memory()
        return self._memory

    @property
    def tool_registry(self):
        """Lazy-load tool registry."""
        if self._tool_registry is None:
            from ct.tools import registry
            self._tool_registry = registry
        return self._tool_registry

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get agent-specific system prompt."""
        pass

    @abstractmethod
    def analyze(self, context: dict, workspace: dict) -> AgentResponse:
        """
        Perform domain-specific analysis.

        Args:
            context: Query context and parameters
            workspace: Shared workspace with other agents' findings

        Returns:
            AgentResponse with findings and recommendations
        """
        pass

    def get_available_tools(self) -> list[str]:
        """Get list of tools this agent can use."""
        return []

    def call_tool(self, tool_name: str, **kwargs) -> dict:
        """
        Call a tool by name.

        Args:
            tool_name: Name of the tool
            **kwargs: Tool parameters

        Returns:
            Tool result dictionary
        """
        tool = self.tool_registry.get_tool(tool_name)
        if tool is None:
            return {"error": f"Tool '{tool_name}' not found"}

        try:
            result = tool.run(**kwargs)
            return result if isinstance(result, dict) else {"result": result}
        except Exception as e:
            logger.error(f"Tool call failed: {tool_name} - {e}")
            return {"error": str(e)}

    def build_context(self, query: str, workspace: dict) -> str:
        """
        Build context string for the agent.

        Args:
            query: User query
            workspace: Shared workspace

        Returns:
            Context string
        """
        context_parts = [
            f"User Query: {query}",
            "",
            "Workspace State:",
        ]

        if workspace.get("findings"):
            for role, finding in workspace["findings"].items():
                context_parts.append(f"  {role}: {finding.get('summary', 'No summary')}")

        if workspace.get("conflicts"):
            context_parts.append("")
            context_parts.append("Identified Conflicts:")
            for conflict in workspace["conflicts"]:
                context_parts.append(f"  - {conflict}")

        return "\n".join(context_parts)

    def calculate_confidence(self, findings: dict, tool_results: list[dict]) -> float:
        """
        Calculate confidence score for findings.

        Args:
            findings: Agent's findings
            tool_results: Results from tool calls

        Returns:
            Confidence score (0.0 - 1.0)
        """
        confidence = 0.5  # Base confidence

        # Boost for successful tool calls
        successful_tools = sum(1 for r in tool_results if not r.get("error"))
        total_tools = len(tool_results) if tool_results else 1
        tool_success_rate = successful_tools / total_tools
        confidence += tool_success_rate * 0.2

        # Boost for specific findings
        if findings.get("specific_hits"):
            confidence += 0.1

        # Reduce for uncertainty markers
        content = findings.get("summary", "") + str(findings.get("details", ""))
        uncertainty_markers = ["unclear", "uncertain", "might", "possibly", "unknown"]
        for marker in uncertainty_markers:
            if marker in content.lower():
                confidence -= 0.05

        return max(0.1, min(1.0, confidence))

    def generate_response(
        self,
        query: str,
        workspace: dict,
        additional_context: Optional[str] = None,
    ) -> str:
        """
        Generate a response using the LLM.

        Args:
            query: User query
            workspace: Shared workspace
            additional_context: Optional additional context

        Returns:
            Generated response
        """
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": self.build_context(query, workspace)},
        ]

        if additional_context:
            messages.append({"role": "user", "content": additional_context})

        try:
            response = self.llm_client.chat(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.get("content", "")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"Error generating response: {e}"

    def store_finding(self, finding: dict) -> None:
        """Store a finding in agent memory."""
        if self._memory:
            self._memory.store(
                agent_role=self.role.value,
                finding=finding,
                metadata={"timestamp": time.time()},
            )

    def retrieve_relevant(self, query: str, limit: int = 5) -> list[dict]:
        """Retrieve relevant past findings."""
        if self._memory:
            return self._memory.search(
                query=query,
                agent_role=self.role.value,
                limit=limit,
            )
        return []


class SimpleAgent(BaseAgent):
    """Simple agent implementation for basic tasks."""

    def __init__(self, role: AgentRole, system_prompt: str, tools: list[str] = None):
        super().__init__(role)
        self._system_prompt = system_prompt
        self._tools = tools or []

    def get_system_prompt(self) -> str:
        return self._system_prompt

    def get_available_tools(self) -> list[str]:
        return self._tools

    def analyze(self, context: dict, workspace: dict) -> AgentResponse:
        query = context.get("query", "")

        # Generate response
        response = self.generate_response(query, workspace)

        # Create message
        message = AgentMessage(
            agent_role=self.role,
            content=response,
            confidence=0.7,
            findings={"summary": response[:200]},
        )

        return AgentResponse(
            success=True,
            message=message,
        )