"""
Phase 1 Tools for CellType-Agent.

Registers new tools for:
- GraphRAG knowledge graph queries
- ADMET prediction
- GPU-accelerated structure prediction
- Session logging and feedback
"""

from ct.tools import registry


# ============================================================
# KNOWLEDGE GRAPH TOOLS
# ============================================================

@registry.register(
    name="knowledge.search_entities",
    description="Search the biological knowledge graph for entities (genes, drugs, diseases, pathways)",
    category="knowledge",
    parameters={
        "search_term": "Entity name to search for",
        "entity_type": "Optional filter: Gene, Drug, Disease, Pathway, SideEffect",
        "limit": "Maximum results to return (default: 20)",
    },
)
def knowledge_search_entities(
    search_term: str,
    entity_type: str = "",
    limit: int = 20,
    **kwargs,
) -> dict:
    """Search the knowledge graph for biological entities."""
    from ct.knowledge_graph.neo4j_client import get_neo4j_client

    try:
        client = get_neo4j_client()

        entity_types = [entity_type] if entity_type else None
        results = client.search_entities(search_term, entity_types, limit)

        if not results:
            return {
                "summary": f"No entities found matching '{search_term}'",
                "results": [],
            }

        summary = f"Found {len(results)} entities matching '{search_term}'"
        return {
            "summary": summary,
            "results": results,
            "count": len(results),
        }

    except Exception as e:
        return {
            "summary": f"Knowledge graph search failed: {str(e)}",
            "error": str(e),
        }


@registry.register(
    name="knowledge.get_drug_targets",
    description="Get all known protein targets for a drug from the knowledge graph",
    category="knowledge",
    parameters={
        "drug_name": "Drug name or ID (e.g., 'imatinib', 'DB00619')",
    },
)
def knowledge_get_drug_targets(drug_name: str, **kwargs) -> dict:
    """Get drug targets from knowledge graph."""
    from ct.knowledge_graph.neo4j_client import get_neo4j_client

    try:
        client = get_neo4j_client()
        results = client.get_drug_targets(drug_name)

        if not results:
            return {
                "summary": f"No targets found for drug '{drug_name}'",
                "targets": [],
            }

        targets = [r["target"] for r in results if r.get("target")]
        summary = f"Found {len(targets)} targets for {drug_name}: {', '.join(targets[:10])}"

        return {
            "summary": summary,
            "drug": drug_name,
            "targets": targets,
            "relations": results,
        }

    except Exception as e:
        return {
            "summary": f"Failed to get drug targets: {str(e)}",
            "error": str(e),
        }


@registry.register(
    name="knowledge.get_gene_diseases",
    description="Get diseases associated with a gene from the knowledge graph",
    category="knowledge",
    parameters={
        "gene_name": "Gene symbol (e.g., 'TP53', 'KRAS')",
        "limit": "Maximum results (default: 50)",
    },
)
def knowledge_get_gene_diseases(gene_name: str, limit: int = 50, **kwargs) -> dict:
    """Get gene-disease associations from knowledge graph."""
    from ct.knowledge_graph.neo4j_client import get_neo4j_client

    try:
        client = get_neo4j_client()
        results = client.get_disease_genes(gene_name, limit)

        if not results:
            return {
                "summary": f"No diseases found associated with gene '{gene_name}'",
                "diseases": [],
            }

        diseases = [r["gene"] for r in results if r.get("gene")]
        summary = f"Found {len(diseases)} diseases associated with {gene_name}"

        return {
            "summary": summary,
            "gene": gene_name,
            "diseases": diseases,
            "associations": results,
        }

    except Exception as e:
        return {
            "summary": f"Failed to get gene diseases: {str(e)}",
            "error": str(e),
        }


@registry.register(
    name="knowledge.find_path",
    description="Find the shortest path between two entities in the knowledge graph",
    category="knowledge",
    parameters={
        "start_entity": "Starting entity ID",
        "end_entity": "Target entity ID",
        "max_depth": "Maximum path depth (default: 4)",
    },
)
def knowledge_find_path(
    start_entity: str,
    end_entity: str,
    max_depth: int = 4,
    **kwargs,
) -> dict:
    """Find path between entities in knowledge graph."""
    from ct.knowledge_graph.neo4j_client import get_neo4j_client

    try:
        client = get_neo4j_client()
        results = client.find_path(start_entity, end_entity, max_depth)

        if not results:
            return {
                "summary": f"No path found between '{start_entity}' and '{end_entity}'",
                "path": None,
            }

        # Format path
        path_str = " -> ".join(
            [r.get("name", r.get("id", "?")) for r in results[0].get("nodes", [])]
        )

        return {
            "summary": f"Path found: {path_str}",
            "path": results[0],
        }

    except Exception as e:
        return {
            "summary": f"Failed to find path: {str(e)}",
            "error": str(e),
        }


@registry.register(
    name="knowledge.graphrag_query",
    description="Execute a natural language query on the biological knowledge graph",
    category="knowledge",
    parameters={
        "query": "Natural language query (e.g., 'What drugs target KRAS?')",
    },
)
def knowledge_graphrag_query(query: str, **kwargs) -> dict:
    """Execute natural language query on knowledge graph."""
    from ct.knowledge_graph.text_to_cypher import TextToCypher

    try:
        translator = TextToCypher()
        results, method = translator.translate_with_fallback(query)

        if not results:
            return {
                "summary": f"No results found for query: '{query}'",
                "results": [],
                "method": method,
            }

        summary = f"Found {len(results)} results using {method}"
        return {
            "summary": summary,
            "query": query,
            "results": results[:50],  # Limit output
            "method": method,
        }

    except Exception as e:
        return {
            "summary": f"GraphRAG query failed: {str(e)}",
            "error": str(e),
        }


# ============================================================
# ADMET TOOLS
# ============================================================

@registry.register(
    name="admet.predict",
    description="Predict ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) properties for a compound",
    category="safety",
    parameters={
        "smiles": "SMILES string of the compound",
        "endpoints": "Optional: specific endpoints to predict (comma-separated)",
    },
    estimated_cost=0.01,
)
def admet_predict(
    smiles: str,
    endpoints: str = "",
    **kwargs,
) -> dict:
    """Predict ADMET properties for a compound."""
    from ct.admet.predictor import ADMETPredictor

    try:
        predictor = ADMETPredictor()

        endpoint_list = [e.strip() for e in endpoints.split(",")] if endpoints else None

        result = predictor.predict(smiles, endpoint_list)

        # Format output
        critical_issues = predictor.get_critical_issues(result)

        summary_parts = []
        if result.flags:
            summary_parts.append(f"{len(result.flags)} issues detected")
        if critical_issues:
            summary_parts.append(f"{len(critical_issues)} critical issues")
        summary_parts.append(f"confidence={result.confidence:.0%}")

        return {
            "summary": f"ADMET prediction: {', '.join(summary_parts)}",
            "smiles": smiles,
            "predictions": {
                k: v for k, v in result.predictions.items()
                if not k.startswith("_")
            },
            "uncertainties": result.uncertainties,
            "flags": result.flags,
            "critical_issues": critical_issues,
            "confidence": result.confidence,
            "prediction_time_ms": result.prediction_time_ms,
        }

    except Exception as e:
        return {
            "summary": f"ADMET prediction failed: {str(e)}",
            "error": str(e),
        }


@registry.register(
    name="admet.batch_predict",
    description="Predict ADMET properties for multiple compounds",
    category="safety",
    parameters={
        "smiles_list": "Comma-separated list of SMILES strings",
    },
    estimated_cost=0.05,
)
def admet_batch_predict(smiles_list: str, **kwargs) -> dict:
    """Batch ADMET prediction."""
    from ct.admet.predictor import ADMETPredictor

    try:
        smiles_items = [s.strip() for s in smiles_list.split(",") if s.strip()]

        if not smiles_items:
            return {
                "summary": "No valid SMILES provided",
                "error": "empty_input",
            }

        predictor = ADMETPredictor()
        result = predictor.predict_batch(smiles_items)

        # Count issues
        total_issues = sum(len(r.flags) for r in result.results)
        avg_confidence = sum(r.confidence for r in result.results) / len(result.results)

        return {
            "summary": (
                f"Predicted ADMET for {len(smiles_items)} compounds. "
                f"Average confidence: {avg_confidence:.0%}. "
                f"Total issues: {total_issues}"
            ),
            "results": [
                {
                    "smiles": r.smiles,
                    "flags": r.flags,
                    "confidence": r.confidence,
                }
                for r in result.results
            ],
            "summary_stats": result.summary,
            "total_time_ms": result.total_time_ms,
        }

    except Exception as e:
        return {
            "summary": f"Batch ADMET prediction failed: {str(e)}",
            "error": str(e),
        }


@registry.register(
    name="admet.compare_compounds",
    description="Compare ADMET profiles of multiple compounds and rank by safety",
    category="safety",
    parameters={
        "smiles_list": "Comma-separated list of SMILES strings to compare",
    },
    estimated_cost=0.05,
)
def admet_compare_compounds(smiles_list: str, **kwargs) -> dict:
    """Compare ADMET profiles across compounds."""
    from ct.admet.predictor import ADMETPredictor

    try:
        smiles_items = [s.strip() for s in smiles_list.split(",") if s.strip()]

        if len(smiles_items) < 2:
            return {
                "summary": "Need at least 2 compounds to compare",
                "error": "insufficient_input",
            }

        predictor = ADMETPredictor()
        comparisons = predictor.compare_compounds(smiles_items)

        # Format ranked results
        ranked = []
        for i, comp in enumerate(comparisons, 1):
            ranked.append({
                "rank": i,
                "smiles": comp["smiles"],
                "score": comp["score"],
                "issues": comp["critical_issues"],
            })

        best = comparisons[0] if comparisons else None

        return {
            "summary": (
                f"Ranked {len(smiles_items)} compounds by ADMET profile. "
                f"Best: {best['smiles'][:20]}... (score: {best['score']:.1f})"
                if best else "Comparison failed"
            ),
            "ranked_compounds": ranked,
            "recommendation": (
                f"Compound {best['smiles'][:30]} has the best ADMET profile"
                if best else None
            ),
        }

    except Exception as e:
        return {
            "summary": f"ADMET comparison failed: {str(e)}",
            "error": str(e),
        }


# ============================================================
# GPU INFRASTRUCTURE TOOLS
# ============================================================

@registry.register(
    name="gpu.status",
    description="Check GPU availability and status",
    category="compute",
    parameters={},
)
def gpu_status(**kwargs) -> dict:
    """Check GPU status."""
    from ct.gpu_infrastructure.resource_manager import GPUResourceManager

    try:
        manager = GPUResourceManager()
        summary = manager.get_summary()

        if summary["gpu_count"] == 0:
            return {
                "summary": "No GPUs detected",
                "available": False,
            }

        available_gpus = [
            g for g in summary["gpus"]
            if g["status"] == "available"
        ]

        return {
            "summary": (
                f"{summary['gpu_count']} GPUs detected, "
                f"{len(available_gpus)} available, "
                f"{summary['free_vram_gb']:.1f}GB free VRAM"
            ),
            "available": len(available_gpus) > 0,
            "gpu_count": summary["gpu_count"],
            "available_count": len(available_gpus),
            "total_vram_gb": summary["total_vram_gb"],
            "free_vram_gb": summary["free_vram_gb"],
            "gpus": summary["gpus"],
        }

    except Exception as e:
        return {
            "summary": f"GPU status check failed: {str(e)}",
            "available": False,
            "error": str(e),
        }


@registry.register(
    name="boltz2.predict_affinity",
    description="Predict binding affinity for a protein-ligand pair using Boltz-2",
    category="structure",
    parameters={
        "protein_sequence": "Protein amino acid sequence (FASTA or plain)",
        "ligand_smiles": "Ligand SMILES string",
    },
    requires_gpu=True,
    gpu_profile="structure",
    estimated_cost=0.10,
    min_vram_gb=24,
)
def boltz2_predict_affinity(
    protein_sequence: str,
    ligand_smiles: str,
    **kwargs,
) -> dict:
    """Predict binding affinity using Boltz-2."""
    from ct.gpu_infrastructure.boltz2_optimizer import predict_binding_affinity

    return predict_binding_affinity(protein_sequence, ligand_smiles)


@registry.register(
    name="boltz2.virtual_screen",
    description="Virtual screen a library of ligands against a protein target",
    category="structure",
    parameters={
        "protein_sequence": "Target protein amino acid sequence",
        "ligand_smiles_list": "Comma-separated list of ligand SMILES strings",
        "top_k": "Number of top hits to return (default: 10)",
    },
    requires_gpu=True,
    gpu_profile="structure",
    estimated_cost=1.00,
    min_vram_gb=24,
)
def boltz2_virtual_screen(
    protein_sequence: str,
    ligand_smiles_list: str,
    top_k: int = 10,
    **kwargs,
) -> dict:
    """Virtual screen ligands using Boltz-2."""
    from ct.gpu_infrastructure.boltz2_optimizer import virtual_screen_library

    smiles_items = [s.strip() for s in ligand_smiles_list.split(",") if s.strip()]

    if not smiles_items:
        return {
            "summary": "No valid ligand SMILES provided",
            "error": "empty_input",
        }

    return virtual_screen_library(protein_sequence, smiles_items, top_k)


# ============================================================
# SESSION LOGGING TOOLS
# ============================================================

@registry.register(
    name="session.feedback",
    description="Provide feedback on the current session for learning",
    category="claude",
    parameters={
        "rating": "Rating from 1-5 stars",
        "outcome": "One of: validated, partially_validated, refuted, inconclusive",
        "feedback": "Optional text feedback",
    },
)
def session_feedback(
    rating: str = "",
    outcome: str = "",
    feedback: str = "",
    **kwargs,
) -> dict:
    """Record user feedback for the session."""
    from ct.session_logging.logger import SessionLogger

    try:
        logger = SessionLogger()

        # Parse inputs
        rating_int = int(rating) if rating else None

        feedback_data = {}
        if rating_int:
            feedback_data["rating"] = rating_int
        if outcome:
            feedback_data["outcome"] = outcome
        if feedback:
            feedback_data["feedback_text"] = feedback

        if not feedback_data:
            return {
                "summary": "No feedback provided. Use rating, outcome, or feedback parameters.",
            }

        # Get session ID from context
        session_id = kwargs.get("_session", {}).get("session_id")

        if session_id:
            logger.add_feedback(
                session_id=session_id,
                rating=rating_int,
                feedback=feedback,
                outcome=outcome,
            )

        return {
            "summary": f"Feedback recorded: {feedback_data}",
            "thank_you": "Your feedback helps improve the agent!",
        }

    except Exception as e:
        return {
            "summary": f"Failed to record feedback: {str(e)}",
            "error": str(e),
        }


@registry.register(
    name="session.stats",
    description="Get session logging statistics",
    category="claude",
    parameters={},
)
def session_stats(**kwargs) -> dict:
    """Get session statistics."""
    from ct.session_logging.logger import SessionLogger
    from ct.session_logging.feedback_collector import FeedbackCollector

    try:
        logger = SessionLogger()
        stats = logger.get_stats()

        collector = FeedbackCollector()
        feedback_stats = collector.get_stats()

        return {
            "summary": (
                f"{stats['total_sessions']} sessions logged, "
                f"{stats['training_ready']} ready for training, "
                f"{feedback_stats['validation_rate']:.0%} validation rate"
            ),
            "total_sessions": stats["total_sessions"],
            "total_tool_calls": stats["total_tool_calls"],
            "training_ready": stats["training_ready"],
            "with_feedback": stats["with_feedback"],
            "feedback_rate": stats["feedback_rate"],
            "outcome_distribution": stats["outcomes"],
        }

    except Exception as e:
        return {
            "summary": f"Failed to get session stats: {str(e)}",
            "error": str(e),
        }


# Register all tools by importing
logger = __import__("logging").getLogger("ct.tools.phase1")
logger.info("Phase 1 tools registered: knowledge, admet, gpu, session")