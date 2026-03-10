"""
Phase 2 Tools for CellType-Agent.

Registers new tools for generative chemistry:
- BoltzGen de novo binder design
- ESM3 conditional protein generation
- Design pipeline orchestration
- Protein validation
"""

from ct.tools import registry


# ============================================================
# GENERATIVE CHEMISTRY TOOLS
# ============================================================

@registry.register(
    name="generative.design_binder",
    description="Design de novo protein binders against a target using BoltzGen",
    category="design",
    parameters={
        "target_sequence": "Target protein sequence",
        "target_name": "Optional target identifier",
        "num_candidates": "Number of candidates to generate (default: 10)",
        "pocket_residues": "Optional comma-separated residue indices for binding pocket",
    },
    requires_gpu=True,
    gpu_profile="structure",
    estimated_cost=2.00,
    min_vram_gb=40,
)
def generative_design_binder(
    target_sequence: str,
    target_name: str = "",
    num_candidates: int = 10,
    pocket_residues: str = "",
    **kwargs,
) -> dict:
    """Design protein binders against a target."""
    from ct.generative.design_pipeline import design_binder

    pocket_list = [int(r.strip()) for r in pocket_residues.split(",") if r.strip()] if pocket_residues else None

    return design_binder(
        target_sequence=target_sequence,
        target_name=target_name,
        num_candidates=num_candidates,
        pocket_residues=pocket_list,
    )


@registry.register(
    name="generative.design_molecular_glue",
    description="Design molecular glue degraders with off-target avoidance",
    category="design",
    parameters={
        "e3_ligase_sequence": "E3 ligase sequence (e.g., CRBN)",
        "target_protein_sequence": "Target protein to degrade",
        "avoid_offtargets": "Comma-separated list of proteins to avoid (e.g., SALL4,IKZF1)",
    },
    requires_gpu=True,
    gpu_profile="structure",
    estimated_cost=3.00,
    min_vram_gb=40,
)
def generative_design_molecular_glue(
    e3_ligase_sequence: str,
    target_protein_sequence: str,
    avoid_offtargets: str = "",
    **kwargs,
) -> dict:
    """Design molecular glue degraders."""
    from ct.generative.design_pipeline import design_molecular_glue

    offtargets = [t.strip() for t in avoid_offtargets.split(",") if t.strip()]

    return design_molecular_glue(
        e3_ligase=e3_ligase_sequence,
        target_protein=target_protein_sequence,
        avoid_offtargets=offtargets,
    )


@registry.register(
    name="generative.generate_protein",
    description="Generate a protein with specified function using ESM3",
    category="design",
    parameters={
        "function_prompt": "Function description (e.g., 'E3 ligase substrate for CRBN')",
        "length_min": "Minimum sequence length (default: 50)",
        "length_max": "Maximum sequence length (default: 200)",
    },
    estimated_cost=0.50,
)
def generative_generate_protein(
    function_prompt: str,
    length_min: int = 50,
    length_max: int = 200,
    **kwargs,
) -> dict:
    """Generate a protein with ESM3."""
    from ct.generative.esm3_client import generate_protein

    return generate_protein(
        function_prompt=function_prompt,
        length_min=length_min,
        length_max=length_max,
    )


@registry.register(
    name="generative.generate_avoiding_target",
    description="Generate protein that binds target A while avoiding target B (generate-filter-rerank pipeline)",
    category="design",
    parameters={
        "positive_function": "Desired function (e.g., 'binds CRBN')",
        "negative_target": "Target to avoid (e.g., 'binds SALL4')",
    },
    estimated_cost=1.00,
)
def generative_generate_avoiding_target(
    positive_function: str,
    negative_target: str,
    **kwargs,
) -> dict:
    """Generate protein with positive and negative constraints."""
    from ct.generative.esm3_client import generate_avoiding_target

    return generate_avoiding_target(
        positive_function=positive_function,
        negative_target=negative_target,
    )


@registry.register(
    name="generative.optimize_binder",
    description="Optimize an existing binder sequence for better affinity and stability",
    category="design",
    parameters={
        "binder_sequence": "Starting binder sequence",
        "target_sequence": "Target protein sequence",
        "optimization_rounds": "Number of optimization rounds (default: 3)",
    },
    requires_gpu=True,
    gpu_profile="structure",
    estimated_cost=1.50,
    min_vram_gb=40,
)
def generative_optimize_binder(
    binder_sequence: str,
    target_sequence: str,
    optimization_rounds: int = 3,
    **kwargs,
) -> dict:
    """Optimize a binder sequence."""
    from ct.generative.boltzgen_optimizer import BoltzGenOptimizer

    optimizer = BoltzGenOptimizer()
    result = optimizer.optimize_binder(
        initial_sequence=binder_sequence,
        target_sequence=target_sequence,
        optimization_rounds=optimization_rounds,
    )

    return {
        "summary": (
            f"Optimized binder: {len(result.sequence)} residues, "
            f"predicted affinity {result.predicted_affinity_nm:.1f} nM"
        ),
        "optimized_sequence": result.sequence,
        "predicted_affinity_nm": result.predicted_affinity_nm,
        "confidence": result.confidence,
    }


@registry.register(
    name="generative.suggest_mutations",
    description="Suggest mutations to improve a protein sequence",
    category="design",
    parameters={
        "sequence": "Protein sequence to mutate",
        "positions": "Optional comma-separated positions to mutate (1-indexed)",
        "num_mutations": "Number of mutations to suggest",
        "preserve_function": "Optional function to preserve",
    },
    estimated_cost=0.20,
)
def generative_suggest_mutations(
    sequence: str,
    positions: str = "",
    num_mutations: int = 3,
    preserve_function: str = "",
    **kwargs,
) -> dict:
    """Suggest protein mutations."""
    from ct.generative.esm3_client import ESM3Client

    client = ESM3Client()

    position_list = [int(p.strip()) - 1 for p in positions.split(",") if p.strip()] if positions else None

    mutations = client.mutate(
        sequence=sequence,
        mutation_positions=position_list,
        num_mutations=num_mutations,
        preserve_function=preserve_function if preserve_function else None,
    )

    return {
        "summary": f"Suggested {len(mutations)} mutations",
        "mutations": mutations,
        "mutation_notations": [m["notation"] for m in mutations],
    }


# ============================================================
# VALIDATION TOOLS
# ============================================================

@registry.register(
    name="validation.validate_protein",
    description="Validate a protein sequence for stability, aggregation, immunogenicity, and solubility",
    category="safety",
    parameters={
        "sequence": "Protein sequence to validate",
    },
    estimated_cost=0.05,
)
def validation_validate_protein(sequence: str, **kwargs) -> dict:
    """Validate a protein sequence."""
    from ct.validation.protein_validator import validate_protein

    return validate_protein(sequence)


@registry.register(
    name="validation.batch_validate",
    description="Validate multiple protein sequences",
    category="safety",
    parameters={
        "sequences": "Comma-separated list of sequences to validate",
    },
    estimated_cost=0.10,
)
def validation_batch_validate(sequences: str, **kwargs) -> dict:
    """Validate multiple protein sequences."""
    from ct.validation.protein_validator import ProteinValidator

    seq_list = [s.strip() for s in sequences.split("|") if s.strip()]

    if not seq_list:
        return {
            "summary": "No sequences provided",
            "error": True,
        }

    validator = ProteinValidator()
    results = validator.batch_validate(seq_list)
    summary = validator.get_validation_summary(results)

    return {
        "summary": (
            f"Validated {summary['total']} sequences: "
            f"{summary['passed']} passed, {summary['failed']} failed"
        ),
        "total": summary["total"],
        "passed": summary["passed"],
        "failed": summary["failed"],
        "pass_rate": summary["pass_rate"],
        "average_stability": summary["avg_stability"],
        "high_aggregation_count": summary["high_aggregation_count"],
        "high_immunogenicity_count": summary["high_immunogenicity_count"],
        "results": [
            {
                "sequence": r.sequence[:50] + "..." if len(r.sequence) > 50 else r.sequence,
                "passed": r.passed,
                "stability": r.stability_score,
                "aggregation": r.aggregation_risk,
                "issues": r.issues,
            }
            for r in results
        ],
    }


@registry.register(
    name="validation.predict_aggregation",
    description="Predict aggregation propensity for a protein sequence",
    category="safety",
    parameters={
        "sequence": "Protein sequence",
    },
    estimated_cost=0.02,
)
def validation_predict_aggregation(sequence: str, **kwargs) -> dict:
    """Predict aggregation propensity."""
    from ct.validation.protein_validator import ProteinValidator

    validator = ProteinValidator()

    # Get aggregation prediction only
    clean_seq = validator._clean_sequence(sequence)
    score, risk = validator._predict_aggregation(clean_seq)

    return {
        "summary": f"Aggregation risk: {risk} (score: {score:.2f})",
        "aggregation_score": score,
        "aggregation_risk": risk,
        "recommendations": _get_aggregation_recommendations(score, risk),
    }


def _get_aggregation_recommendations(score: float, risk: str) -> list:
    """Get recommendations for reducing aggregation."""
    recommendations = []

    if risk == "high":
        recommendations.extend([
            "Consider reducing hydrophobic residue content (V, I, L, F, Y, W, M)",
            "Add charged residues (D, E, K, R) to improve solubility",
            "Break up long hydrophobic stretches with polar residues",
            "Consider adding solubility tags (MBP, GST, SUMO)",
        ])
    elif risk == "medium":
        recommendations.extend([
            "Monitor during expression - may need optimization",
            "Consider codon optimization for expression system",
            "Test different buffer conditions",
        ])

    return recommendations


@registry.register(
    name="validation.predict_immunogenicity",
    description="Predict immunogenicity risk for therapeutic protein",
    category="safety",
    parameters={
        "sequence": "Protein sequence",
    },
    estimated_cost=0.02,
)
def validation_predict_immunogenicity(sequence: str, **kwargs) -> dict:
    """Predict immunogenicity risk."""
    from ct.validation.protein_validator import ProteinValidator

    validator = ProteinValidator()
    clean_seq = validator._clean_sequence(sequence)
    score, risk = validator._predict_immunogenicity(clean_seq)

    return {
        "summary": f"Immunogenicity risk: {risk} (score: {score:.2f})",
        "immunogenicity_score": score,
        "immunogenicity_risk": risk,
        "recommendations": _get_immunogenicity_recommendations(risk),
    }


def _get_immunogenicity_recommendations(risk: str) -> list:
    """Get recommendations for reducing immunogenicity."""
    recommendations = []

    if risk == "high":
        recommendations.extend([
            "Consider humanization if from non-human source",
            "Remove predicted T-cell epitopes through mutagenesis",
            "Add PEGylation to reduce immune recognition",
            "Consider immune tolerance induction strategies",
        ])
    elif risk == "medium":
        recommendations.extend([
            "Consider epitope deimmunization",
            "Test in immunogenicity assays",
        ])

    return recommendations


# Register all tools
logger = __import__("logging").getLogger("ct.tools.phase2")
logger.info("Phase 2 tools registered: generative, validation")