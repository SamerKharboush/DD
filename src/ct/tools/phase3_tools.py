"""
Phase 3 Tools for CellType-Agent.

Registers new tools for multi-agent orchestration:
- Multi-agent analysis
- Agent-specific queries
- Vector memory operations
- DMTA cycle commands
"""

from ct.tools import registry


# ============================================================
# MULTI-AGENT TOOLS
# ============================================================

@registry.register(
    name="multi_agent.analyze",
    description="Run comprehensive multi-agent analysis with specialist agents (chemist, biologist, toxicologist) and adversarial critique",
    category="analysis",
    parameters={
        "query": "Research question to analyze",
        "mode": "Orchestration mode: sequential, parallel, or debate (default: sequential)",
    },
    estimated_cost=2.00,
)
def multi_agent_analyze(query: str, mode: str = "sequential", **kwargs) -> dict:
    """Run multi-agent analysis."""
    from ct.agents.orchestrator import run_multi_agent_analysis

    return run_multi_agent_analysis(query=query, mode=mode)


@registry.register(
    name="multi_agent.chemist_opinion",
    description="Get the chemist agent's perspective on a molecular design question",
    category="analysis",
    parameters={
        "query": "Chemistry question",
        "compound_smiles": "Optional compound SMILES to analyze",
    },
    estimated_cost=0.30,
)
def multi_agent_chemist_opinion(
    query: str,
    compound_smiles: str = "",
    **kwargs,
) -> dict:
    """Get chemist agent's analysis."""
    from ct.agents.specialist_agents import ChemistAgent

    agent = ChemistAgent()
    context = {
        "query": query,
        "compound_smiles": compound_smiles,
    }
    response = agent.analyze(context, {})

    return {
        "summary": response.message.content[:500],
        "confidence": response.message.confidence,
        "issues": response.issues_found[:5],
        "recommendations": response.suggested_actions[:5],
    }


@registry.register(
    name="multi_agent.biologist_opinion",
    description="Get the biologist agent's perspective on a target/disease question",
    category="analysis",
    parameters={
        "query": "Biology question",
        "gene": "Optional gene symbol",
        "disease": "Optional disease name",
    },
    estimated_cost=0.30,
)
def multi_agent_biologist_opinion(
    query: str,
    gene: str = "",
    disease: str = "",
    **kwargs,
) -> dict:
    """Get biologist agent's analysis."""
    from ct.agents.specialist_agents import BiologistAgent

    agent = BiologistAgent()
    context = {
        "query": query,
        "gene": gene,
        "disease": disease,
    }
    response = agent.analyze(context, {})

    return {
        "summary": response.message.content[:500],
        "confidence": response.message.confidence,
        "recommendations": response.suggested_actions[:5],
    }


@registry.register(
    name="multi_agent.toxicologist_review",
    description="Get the toxicologist agent's safety assessment (adversarial review)",
    category="safety",
    parameters={
        "query": "Safety question",
        "compound_smiles": "Compound SMILES to assess",
    },
    estimated_cost=0.30,
)
def multi_agent_toxicologist_review(
    query: str,
    compound_smiles: str = "",
    **kwargs,
) -> dict:
    """Get toxicologist's safety review."""
    from ct.agents.specialist_agents import ToxicologistAgent

    agent = ToxicologistAgent()
    context = {
        "query": query,
        "compound_smiles": compound_smiles,
    }
    response = agent.analyze(context, {})

    return {
        "summary": response.message.content[:500],
        "confidence": response.message.confidence,
        "critical_issues": response.issues_found,
        "recommendations": response.suggested_actions[:5],
        "passed": response.success,
    }


@registry.register(
    name="multi_agent.adversarial_review",
    description="Run adversarial critique of a proposed compound or hypothesis",
    category="analysis",
    parameters={
        "query": "Hypothesis or compound to critique",
        "context": "Additional context (JSON string)",
    },
    estimated_cost=0.50,
)
def multi_agent_adversarial_review(
    query: str,
    context: str = "{}",
    **kwargs,
) -> dict:
    """Run adversarial critique."""
    from ct.agents.critic_agent import CriticAgent
    import json

    agent = CriticAgent()
    try:
        context_dict = json.loads(context) if context else {}
    except json.JSONDecodeError:
        context_dict = {}

    context_dict["query"] = query
    response = agent.analyze(context_dict, {})

    return {
        "summary": response.message.content[:500],
        "confidence": response.message.confidence,
        "issues_found": response.issues_found,
        "recommendations": response.suggested_actions,
    }


# ============================================================
# MEMORY TOOLS
# ============================================================

@registry.register(
    name="memory.store",
    description="Store an important finding in persistent memory for later retrieval",
    category="memory",
    parameters={
        "content": "Content to remember",
        "session_id": "Optional session ID",
        "metadata": "Optional metadata (JSON string)",
    },
)
def memory_store(
    content: str,
    session_id: str = "",
    metadata: str = "{}",
    **kwargs,
) -> dict:
    """Store in memory."""
    from ct.memory.vector_memory import get_agent_memory
    import json

    memory = get_agent_memory()
    try:
        metadata_dict = json.loads(metadata) if metadata else {}
    except json.JSONDecodeError:
        metadata_dict = {}

    entry_id = memory.store(
        content=content,
        session_id=session_id if session_id else None,
        metadata=metadata_dict,
    )

    return {
        "summary": f"Stored memory entry: {entry_id}",
        "entry_id": entry_id,
    }


@registry.register(
    name="memory.recall",
    description="Search memory for relevant past findings",
    category="memory",
    parameters={
        "query": "Search query",
        "agent_role": "Optional filter by agent role (chemist, biologist, etc.)",
        "limit": "Maximum results (default: 5)",
    },
)
def memory_recall(
    query: str,
    agent_role: str = "",
    limit: int = 5,
    **kwargs,
) -> dict:
    """Recall from memory."""
    from ct.memory.vector_memory import get_agent_memory

    memory = get_agent_memory()
    results = memory.search(
        query=query,
        agent_role=agent_role if agent_role else None,
        limit=limit,
    )

    return {
        "summary": f"Found {len(results)} relevant memories",
        "results": [
            {
                "content": r.entry.content[:200],
                "score": r.score,
                "agent": r.entry.agent_role,
                "created_at": r.entry.created_at,
            }
            for r in results
        ],
    }


@registry.register(
    name="memory.stats",
    description="Get memory statistics",
    category="memory",
    parameters={},
)
def memory_stats(**kwargs) -> dict:
    """Get memory statistics."""
    from ct.memory.vector_memory import get_agent_memory

    memory = get_agent_memory()
    stats = memory.get_stats()

    return {
        "summary": f"Memory contains {stats['total_entries']} entries",
        "stats": stats,
    }


# ============================================================
# DMTA CYCLE TOOLS
# ============================================================

@registry.register(
    name="dmta.run_cycle",
    description="Run a complete Design-Make-Test-Analyze cycle for drug discovery",
    category="workflow",
    parameters={
        "target_sequence": "Target protein sequence",
        "num_candidates": "Number of candidates per cycle (default: 10)",
        "iterations": "Number of DMTA iterations (default: 1)",
    },
    estimated_cost=5.00,
)
def dmta_run_cycle(
    target_sequence: str,
    num_candidates: int = 10,
    iterations: int = 1,
    **kwargs,
) -> dict:
    """Run DMTA cycle."""
    from ct.campaign.dmta import run_dmta_cycle

    return run_dmta_cycle(
        target=target_sequence,
        num_candidates=num_candidates,
        iterations=iterations,
    )


@registry.register(
    name="dmta.design",
    description="Design phase - generate candidate molecules",
    category="workflow",
    parameters={
        "target_sequence": "Target protein sequence",
        "num_candidates": "Number of candidates (default: 10)",
        "method": "Design method: boltzgen, esm3, or hybrid (default: hybrid)",
    },
    requires_gpu=True,
    gpu_profile="structure",
    estimated_cost=2.00,
)
def dmta_design(
    target_sequence: str,
    num_candidates: int = 10,
    method: str = "hybrid",
    **kwargs,
) -> dict:
    """Run design phase only."""
    from ct.campaign.dmta import DMTACycle

    dmta = DMTACycle(target=target_sequence)
    return dmta.design(method=method, num_candidates=num_candidates)


@registry.register(
    name="dmta.test",
    description="Test phase - predict assay results for candidates",
    category="workflow",
    parameters={
        "target_sequence": "Target protein sequence",
        "candidates": "Comma-separated candidate sequences",
        "assays": "Comma-separated assays: binding, admet, selectivity (default: all)",
    },
    estimated_cost=1.00,
)
def dmta_test(
    target_sequence: str,
    candidates: str = "",
    assays: str = "",
    **kwargs,
) -> dict:
    """Run test phase."""
    from ct.campaign.dmta import DMTACycle

    dmta = DMTACycle(target=target_sequence)

    candidate_list = [c.strip() for c in candidates.split("|") if c.strip()]
    assay_list = [a.strip() for a in assays.split(",") if a.strip()] if assays else None

    # If no candidates provided, design first
    if not candidate_list:
        design_result = dmta.design()
        candidate_list = design_result.get("candidates", [])

    return dmta.test(candidates=candidate_list, assays=assay_list)


# Register all tools
logger = __import__("logging").getLogger("ct.tools.phase3")
logger.info("Phase 3 tools registered: multi_agent, memory, dmta")