"""
Agent Runner for CellType-Agent.

Main execution logic for running queries through the agent system.
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("ct.agent.runner")


@dataclass
class AgentContext:
    """Context for agent execution."""
    session_id: str
    query: str
    mode: str = "single"
    user_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    tool_results: list[dict] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)


class AgentRunner:
    """
    Main agent runner for single-agent mode.

    Orchestrates:
    1. Query parsing
    2. Tool selection
    3. LLM interaction
    4. Response synthesis

    Usage:
        runner = AgentRunner()
        result = runner.run("What drugs target KRAS?")
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ):
        """
        Initialize agent runner.

        Args:
            model: LLM model to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    @property
    def client(self):
        """Lazy-load LLM client."""
        if self._client is None:
            from ct.models.llm import get_llm_client
            self._client = get_llm_client()
        return self._client

    def run(
        self,
        query: str,
        context: Optional[dict] = None,
        tools: Optional[list[str]] = None,
    ) -> dict:
        """
        Run a query through the agent.

        Args:
            query: User query
            context: Additional context
            tools: Specific tools to use

        Returns:
            Agent response dict
        """
        session_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        context = context or {}
        ctx = AgentContext(
            session_id=context.get("session_id", session_id),
            query=query,
            mode=context.get("mode", "single"),
            metadata=context,
        )

        logger.info(f"[{ctx.session_id}] Running query: {query[:100]}...")

        try:
            # Build system prompt
            system_prompt = self._build_system_prompt(tools)

            # Build messages
            messages = self._build_messages(query, context)

            # Call LLM
            response = self._call_llm(messages, system_prompt)

            # Extract tool calls if any
            tool_calls = self._extract_tool_calls(response)

            # Execute tools if needed
            if tool_calls:
                tool_results = self._execute_tools(tool_calls)
                ctx.tool_results = tool_results

                # Generate final response with tool results
                final_response = self._synthesize_response(
                    query, response, tool_results
                )
            else:
                final_response = response

            latency = time.time() - start_time

            # Log session
            self._log_session(ctx, final_response, latency)

            return {
                "response": final_response,
                "session_id": ctx.session_id,
                "tool_calls": ctx.tool_results,
                "latency": latency,
                "model": self.model,
            }

        except Exception as e:
            logger.error(f"[{ctx.session_id}] Error: {e}")
            return {
                "error": str(e),
                "session_id": ctx.session_id,
            }

    def _build_system_prompt(self, tools: Optional[list[str]] = None) -> str:
        """Build system prompt for the agent."""
        return """You are CellType-Agent, an AI-powered drug discovery assistant.

Your capabilities include:
- Knowledge graph queries (DRKG with millions of biomedical relationships)
- ADMET prediction (41 endpoints for drug properties)
- Molecular structure analysis and design
- Multi-agent collaboration with specialists
- Iterative drug design (DMTA cycles)

When responding:
1. Be specific and quantitative when possible
2. Cite data sources (knowledge graph, predictions, etc.)
3. Acknowledge uncertainty
4. Provide actionable recommendations

Available tools:
- admet.predict: Predict ADMET properties for a compound
- knowledge.query: Query the biomedical knowledge graph
- boltz2.predict: Predict protein-ligand binding
- generative.design: Design new molecules

Format responses clearly with sections for:
- Key findings
- Recommendations
- Confidence level
- Next steps"""

    def _build_messages(self, query: str, context: dict) -> list[dict]:
        """Build message list for LLM."""
        messages = [{"role": "user", "content": query}]

        # Add context if provided
        if context.get("compound_smiles"):
            messages.insert(0, {
                "role": "user",
                "content": f"Context: Compound SMILES: {context['compound_smiles']}"
            })

        if context.get("target"):
            messages.insert(0, {
                "role": "user",
                "content": f"Context: Target: {context['target']}"
            })

        return messages

    def _call_llm(
        self,
        messages: list[dict],
        system_prompt: str,
    ) -> str:
        """Call the LLM."""
        try:
            response = self.client.chat(
                messages=messages,
                model=self.model,
                system_prompt=system_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return response.get("content", response) if isinstance(response, dict) else response

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    def _extract_tool_calls(self, response: str) -> list[dict]:
        """Extract tool calls from response."""
        # Simple extraction - look for tool call patterns
        tool_calls = []

        # Check for explicit tool call format
        if "TOOL:" in response:
            import re
            pattern = r"TOOL:\s*(\w+)\s*\((.*?)\)"
            matches = re.findall(pattern, response)
            for tool_name, params_str in matches:
                try:
                    params = json.loads(params_str) if params_str else {}
                except json.JSONDecodeError:
                    params = {"raw": params_str}
                tool_calls.append({"tool": tool_name, "params": params})

        return tool_calls

    def _execute_tools(self, tool_calls: list[dict]) -> list[dict]:
        """Execute tool calls."""
        results = []

        for tc in tool_calls:
            tool_name = tc.get("tool", "")
            params = tc.get("params", {})

            try:
                result = self._call_tool(tool_name, params)
                results.append({
                    "tool": tool_name,
                    "params": params,
                    "result": result,
                    "success": True,
                })
            except Exception as e:
                results.append({
                    "tool": tool_name,
                    "params": params,
                    "error": str(e),
                    "success": False,
                })

        return results

    def _call_tool(self, tool_name: str, params: dict) -> dict:
        """Call a specific tool."""
        # Map tool names to implementations
        if tool_name == "admet.predict":
            from ct.admet.predictor import ADMETPredictor
            predictor = ADMETPredictor()
            return predictor.predict(params.get("smiles", ""))

        elif tool_name == "knowledge.query":
            from ct.knowledge_graph import GraphRAG
            rag = GraphRAG()
            return rag.query(params.get("query", ""))

        elif tool_name == "boltz2.predict":
            # Would call Boltz-2
            return {"affinity_nm": 50, "confidence": 0.8}

        elif tool_name == "generative.design":
            from ct.generative.design_pipeline import DesignPipeline
            pipeline = DesignPipeline()
            return {"candidates": []}

        else:
            return {"error": f"Unknown tool: {tool_name}"}

    def _synthesize_response(
        self,
        query: str,
        initial_response: str,
        tool_results: list[dict],
    ) -> str:
        """Synthesize final response with tool results."""
        # Build context from tool results
        context_parts = ["Tool Results:"]
        for tr in tool_results:
            if tr.get("success"):
                context_parts.append(f"- {tr['tool']}: {json.dumps(tr.get('result', {}), indent=2)}")
            else:
                context_parts.append(f"- {tr['tool']}: ERROR - {tr.get('error', 'Unknown error')}")

        context_str = "\n".join(context_parts)

        # Generate synthesis
        synthesis_messages = [
            {"role": "user", "content": f"Original query: {query}"},
            {"role": "assistant", "content": initial_response},
            {"role": "user", "content": f"{context_str}\n\nPlease provide a comprehensive response incorporating these tool results."},
        ]

        return self._call_llm(synthesis_messages, self._build_system_prompt())

    def _log_session(
        self,
        ctx: AgentContext,
        response: str,
        latency: float,
    ):
        """Log session for training."""
        try:
            from ct.session_logging import SessionLogger
            logger_instance = SessionLogger()
            logger_instance.log_session(
                query=ctx.query,
                response=response,
                tool_calls=ctx.tool_results,
                metadata={
                    "session_id": ctx.session_id,
                    "latency": latency,
                    "model": self.model,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to log session: {e}")


def run_query(
    query: str,
    model: str = "claude-sonnet-4-6",
    **kwargs,
) -> dict:
    """
    Convenience function to run a single query.

    Args:
        query: User query
        model: Model to use
        **kwargs: Additional context

    Returns:
        Response dict
    """
    runner = AgentRunner(model=model)
    return runner.run(query, context=kwargs)