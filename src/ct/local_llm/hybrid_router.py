"""
Hybrid Router for intelligent routing between local and cloud models.

Routes queries based on:
- Query complexity
- Cost optimization
- Data privacy requirements
- Model capabilities
"""

import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger("ct.local_llm.router")


class ModelTier(Enum):
    """Model tier levels."""
    LOCAL_FAST = "local_fast"  # Local 7B model
    LOCAL_CAPABLE = "local_capable"  # Local 70B model
    CLOUD_FAST = "cloud_fast"  # Claude Sonnet
    CLOUD_CAPABLE = "cloud_capable"  # Claude Opus


@dataclass
class RoutingDecision:
    """Result of routing decision."""
    selected_tier: ModelTier
    reason: str
    estimated_cost: float
    estimated_latency: float


class HybridRouter:
    """
    Intelligent router between local and cloud models.

    Routing strategy:
    1. Simple lookups → local 7B
    2. Standard analysis → local 70B
    3. Complex reasoning → cloud Sonnet
    4. Expert-level → cloud Opus

    Usage:
        router = HybridRouter()
        decision = router.route(query)
        response = router.execute(query, decision)
    """

    # Cost per 1K tokens (approximate)
    COST_PER_1K_TOKENS = {
        ModelTier.LOCAL_FAST: 0.0001,  # Electricity
        ModelTier.LOCAL_CAPABLE: 0.0005,
        ModelTier.CLOUD_FAST: 0.003,
        ModelTier.CLOUD_CAPABLE: 0.015,
    }

    # Latency estimates (seconds)
    LATENCY_ESTIMATES = {
        ModelTier.LOCAL_FAST: 1.0,
        ModelTier.LOCAL_CAPABLE: 3.0,
        ModelTier.CLOUD_FAST: 2.0,
        ModelTier.CLOUD_CAPABLE: 5.0,
    }

    def __init__(
        self,
        prefer_local: bool = True,
        cost_threshold: float = 0.10,
        privacy_mode: bool = False,
    ):
        """
        Initialize hybrid router.

        Args:
            prefer_local: Prefer local models when possible
            cost_threshold: Maximum cost per query
            privacy_mode: Always use local for sensitive data
        """
        self.prefer_local = prefer_local
        self.cost_threshold = cost_threshold
        self.privacy_mode = privacy_mode

        self._local_client = None
        self._cloud_client = None

    @property
    def local_client(self):
        """Lazy-load local client."""
        if self._local_client is None:
            from ct.local_llm.local_client import LocalLLMClient
            self._local_client = LocalLLMClient()
        return self._local_client

    @property
    def cloud_client(self):
        """Lazy-load cloud client."""
        if self._cloud_client is None:
            from ct.models.llm import get_llm_client
            self._cloud_client = get_llm_client()
        return self._cloud_client

    def route(
        self,
        query: str,
        context: Optional[dict] = None,
    ) -> RoutingDecision:
        """
        Determine the best model for a query.

        Args:
            query: User query
            context: Additional context

        Returns:
            RoutingDecision
        """
        context = context or {}

        # Check privacy mode
        if self.privacy_mode or context.get("contains_phi"):
            return RoutingDecision(
                selected_tier=ModelTier.LOCAL_CAPABLE,
                reason="Privacy mode - local only",
                estimated_cost=self.COST_PER_1K_TOKENS[ModelTier.LOCAL_CAPABLE] * 50,
                estimated_latency=self.LATENCY_ESTIMATES[ModelTier.LOCAL_CAPABLE],
            )

        # Analyze query complexity
        complexity = self._analyze_complexity(query, context)

        # Simple queries
        if complexity < 0.3:
            if self.prefer_local and self.local_client.is_available():
                return RoutingDecision(
                    selected_tier=ModelTier.LOCAL_FAST,
                    reason="Simple query - local fast model sufficient",
                    estimated_cost=self.COST_PER_1K_TOKENS[ModelTier.LOCAL_FAST] * 20,
                    estimated_latency=self.LATENCY_ESTIMATES[ModelTier.LOCAL_FAST],
                )

        # Standard queries
        if complexity < 0.6:
            if self.prefer_local and self.local_client.is_available():
                return RoutingDecision(
                    selected_tier=ModelTier.LOCAL_CAPABLE,
                    reason="Standard query - local capable model",
                    estimated_cost=self.COST_PER_1K_TOKENS[ModelTier.LOCAL_CAPABLE] * 50,
                    estimated_latency=self.LATENCY_ESTIMATES[ModelTier.LOCAL_CAPABLE],
                )
            return RoutingDecision(
                selected_tier=ModelTier.CLOUD_FAST,
                reason="Standard query - cloud fast model",
                estimated_cost=self.COST_PER_1K_TOKENS[ModelTier.CLOUD_FAST] * 50,
                estimated_latency=self.LATENCY_ESTIMATES[ModelTier.CLOUD_FAST],
            )

        # Complex queries
        if complexity < 0.85:
            return RoutingDecision(
                selected_tier=ModelTier.CLOUD_FAST,
                reason="Complex query - cloud fast model",
                estimated_cost=self.COST_PER_1K_TOKENS[ModelTier.CLOUD_FAST] * 100,
                estimated_latency=self.LATENCY_ESTIMATES[ModelTier.CLOUD_FAST],
            )

        # Expert-level queries
        return RoutingDecision(
            selected_tier=ModelTier.CLOUD_CAPABLE,
            reason="Expert-level query - cloud capable model",
            estimated_cost=self.COST_PER_1K_TOKENS[ModelTier.CLOUD_CAPABLE] * 100,
            estimated_latency=self.LATENCY_ESTIMATES[ModelTier.CLOUD_CAPABLE],
        )

    def _analyze_complexity(self, query: str, context: dict) -> float:
        """
        Analyze query complexity.

        Returns:
            Complexity score (0.0 - 1.0)
        """
        complexity = 0.0

        # Length factor
        query_len = len(query)
        if query_len > 500:
            complexity += 0.2
        elif query_len > 200:
            complexity += 0.1

        # Keyword complexity
        complex_keywords = [
            "design", "generate", "optimize", "analyze", "compare",
            "predict", "synthesize", "validate", "validate", "validate",
        ]
        for kw in complex_keywords:
            if kw in query.lower():
                complexity += 0.1

        # Tool requirements
        if context.get("requires_gpu"):
            complexity += 0.2

        if context.get("requires_multi_agent"):
            complexity += 0.2

        # Domain complexity
        complex_domains = ["generative", "de novo", "multi-target", "off-target"]
        for domain in complex_domains:
            if domain in query.lower():
                complexity += 0.15

        return min(1.0, complexity)

    def execute(
        self,
        query: str,
        decision: Optional[RoutingDecision] = None,
        context: Optional[dict] = None,
    ) -> dict:
        """
        Execute query with selected model.

        Args:
            query: User query
            decision: Pre-computed routing decision
            context: Additional context

        Returns:
            Response dict
        """
        decision = decision or self.route(query, context)
        context = context or {}

        start_time = time.time()

        try:
            if decision.selected_tier in (ModelTier.LOCAL_FAST, ModelTier.LOCAL_CAPABLE):
                response = self.local_client.chat(query)
            else:
                model = "claude-sonnet-4-6" if decision.selected_tier == ModelTier.CLOUD_FAST else "claude-opus-4-6"
                response = self.cloud_client.chat(
                    messages=[{"role": "user", "content": query}],
                    model=model,
                )
                response = response.get("content", "")

            return {
                "response": response,
                "tier": decision.selected_tier.value,
                "reason": decision.reason,
                "latency": time.time() - start_time,
                "cost": decision.estimated_cost,
            }

        except Exception as e:
            logger.error(f"Execution failed on {decision.selected_tier}: {e}")
            # Fallback to cloud
            if decision.selected_tier != ModelTier.CLOUD_CAPABLE:
                return self.execute(
                    query,
                    RoutingDecision(
                        selected_tier=ModelTier.CLOUD_CAPABLE,
                        reason="Fallback after failure",
                        estimated_cost=self.COST_PER_1K_TOKENS[ModelTier.CLOUD_CAPABLE] * 50,
                        estimated_latency=self.LATENCY_ESTIMATES[ModelTier.CLOUD_CAPABLE],
                    ),
                    context,
                )
            return {"error": str(e)}

    def get_status(self) -> dict:
        """Get router status."""
        return {
            "local_available": self.local_client.is_available() if self._local_client else False,
            "prefer_local": self.prefer_local,
            "privacy_mode": self.privacy_mode,
            "cost_threshold": self.cost_threshold,
        }


def estimate_query_cost(query: str, mode: str = "hybrid") -> dict:
    """
    Estimate cost for a query.

    Args:
        query: Query string
        mode: Execution mode

    Returns:
        Cost estimate
    """
    router = HybridRouter()
    decision = router.route(query)

    return {
        "estimated_cost": decision.estimated_cost,
        "estimated_latency": decision.estimated_latency,
        "recommended_tier": decision.selected_tier.value,
        "reason": decision.reason,
    }