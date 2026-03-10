"""
Design Pipeline for end-to-end binder design.

Orchestrates the generate-filter-rerank workflow:
1. Generate candidates with BoltzGen/ESM3
2. Filter by ADMET and off-target constraints
3. Score with Boltz-2 affinity
4. Validate structure and stability
5. Rerank and return top candidates
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("ct.generative.design_pipeline")


@dataclass
class DesignSpecification:
    """Specification for a design task."""
    target_sequence: str
    target_name: str = ""
    design_type: str = "binder"  # binder, glue, peptide, antibody
    positive_constraints: list[str] = field(default_factory=list)
    negative_constraints: list[str] = field(default_factory=list)
    pocket_residues: Optional[list[int]] = None
    length_range: tuple[int, int] = (50, 150)
    num_candidates: int = 10
    affinity_threshold_nm: float = 100.0
    stability_threshold: float = 0.5
    avoid_residues: list[str] = field(default_factory=list)
    covalent_site: Optional[dict] = None


@dataclass
class PipelineResult:
    """Result of the design pipeline."""
    specification: DesignSpecification
    candidates: list[dict]
    total_generated: int
    passed_filters: int
    design_time_seconds: float
    best_candidate: Optional[dict] = None
    summary: str = ""


class DesignPipeline:
    """
    End-to-end design pipeline for protein binders.

    Usage:
        pipeline = DesignPipeline()
        spec = DesignSpecification(
            target_sequence="MKT...",
            positive_constraints=["binds CRBN"],
            negative_constraints=["binds SALL4"],
            num_candidates=10,
        )
        result = pipeline.run(spec)
    """

    def __init__(
        self,
        use_boltzgen: bool = True,
        use_esm3: bool = True,
        use_admet_filter: bool = True,
        use_structure_validation: bool = True,
    ):
        """
        Initialize design pipeline.

        Args:
            use_boltzgen: Use BoltzGen for binder design
            use_esm3: Use ESM3 for sequence optimization
            use_admet_filter: Apply ADMET filtering
            use_structure_validation: Validate generated structures
        """
        self.use_boltzgen = use_boltzgen
        self.use_esm3 = use_esm3
        self.use_admet_filter = use_admet_filter
        self.use_structure_validation = use_structure_validation

        # Lazy-loaded components
        self._boltzgen = None
        self._esm3 = None
        self._admet = None
        self._boltz2 = None
        self._validator = None

    def run(self, spec: DesignSpecification) -> PipelineResult:
        """
        Run the complete design pipeline.

        Args:
            spec: Design specification

        Returns:
            PipelineResult with filtered candidates
        """
        start_time = time.time()
        logger.info(f"Starting design pipeline for {spec.target_name or 'unnamed target'}")

        # Phase 1: Generate candidates
        candidates = self._generate_candidates(spec)
        total_generated = len(candidates)
        logger.info(f"Generated {total_generated} initial candidates")

        # Phase 2: Filter by constraints
        candidates = self._filter_candidates(candidates, spec)
        logger.info(f"{len(candidates)} candidates passed initial filters")

        # Phase 3: Score with affinity prediction
        candidates = self._score_candidates(candidates, spec)
        logger.info(f"Scored {len(candidates)} candidates")

        # Phase 4: Validate structures
        if self.use_structure_validation:
            candidates = self._validate_candidates(candidates)
            logger.info(f"{len(candidates)} candidates passed validation")

        # Phase 5: Rerank
        candidates = self._rerank_candidates(candidates, spec)

        # Limit to requested number
        candidates = candidates[:spec.num_candidates]

        # Find best
        best = candidates[0] if candidates else None

        duration = time.time() - start_time

        return PipelineResult(
            specification=spec,
            candidates=candidates,
            total_generated=total_generated,
            passed_filters=len(candidates),
            design_time_seconds=duration,
            best_candidate=best,
            summary=self._generate_summary(candidates, duration),
        )

    def _generate_candidates(self, spec: DesignSpecification) -> list[dict]:
        """Generate initial candidate pool."""
        candidates = []

        # BoltzGen for structure-based design
        if self.use_boltzgen:
            boltzgen = self._get_boltzgen()
            result = boltzgen.design_binders(
                target_sequence=spec.target_sequence,
                target_name=spec.target_name,
                num_candidates=spec.num_candidates * 2,  # Generate extra for filtering
                pocket_residues=spec.pocket_residues,
                constraints={
                    "avoid_residues": spec.avoid_residues,
                    "covalent_site": spec.covalent_site,
                },
            )

            for candidate in result.candidates:
                candidates.append({
                    "sequence": candidate.sequence,
                    "source": "boltzgen",
                    "predicted_affinity_nm": candidate.predicted_affinity_nm,
                    "initial_confidence": candidate.confidence,
                    "pdb_content": candidate.pdb_content,
                })

        # ESM3 for function-constrained design
        if self.use_esm3 and spec.positive_constraints:
            esm3 = self._get_esm3()
            generations = esm3.generate_with_constraints(
                positive_constraints=spec.positive_constraints,
                negative_constraints=spec.negative_constraints,
                length_range=spec.length_range,
                num_candidates=spec.num_candidates,
            )

            for gen in generations:
                candidates.append({
                    "sequence": gen.sequence,
                    "source": "esm3",
                    "initial_confidence": gen.confidence,
                })

        return candidates

    def _filter_candidates(self, candidates: list[dict], spec: DesignSpecification) -> list[dict]:
        """Filter candidates by constraints."""
        filtered = []

        for candidate in candidates:
            # Check sequence validity
            if not candidate.get("sequence"):
                continue

            # Check length
            seq_len = len(candidate["sequence"])
            if not (spec.length_range[0] <= seq_len <= spec.length_range[1]):
                continue

            # ADMET filter
            if self.use_admet_filter:
                admet = self._get_admet()
                admet_result = admet.predict(candidate["sequence"])

                # Check for critical issues
                critical_issues = admet.get_critical_issues(admet_result)
                if any(issue["priority"] == 1 for issue in critical_issues):
                    continue

                candidate["admet_result"] = admet_result
                candidate["admet_flags"] = admet_result.flags

            filtered.append(candidate)

        return filtered

    def _score_candidates(self, candidates: list[dict], spec: DesignSpecification) -> list[dict]:
        """Score candidates with Boltz-2 affinity."""
        for candidate in candidates:
            # Skip if already scored
            if candidate.get("predicted_affinity_nm"):
                continue

            # Use Boltz-2 for affinity prediction
            boltz2 = self._get_boltz2()
            affinity_result = boltz2.predict_affinity(
                protein_sequence=spec.target_sequence,
                ligand_smiles="",  # For protein binders, no ligand
            )

            # Convert pKd to nM
            if affinity_result.affinity_pred_value:
                pKd = affinity_result.affinity_pred_value
                kd_nm = 10 ** (9 - pKd)  # Convert pKd to nM
                candidate["predicted_affinity_nm"] = kd_nm

            candidate["binding_probability"] = affinity_result.affinity_probability_binary

        return candidates

    def _validate_candidates(self, candidates: list[dict]) -> list[dict]:
        """Validate candidate structures."""
        validated = []

        for candidate in candidates:
            sequence = candidate.get("sequence")
            if not sequence:
                continue

            # Run structure validation
            validator = self._get_validator()
            validation = validator.validate(sequence)

            # Check thresholds
            if validation.get("stability_score", 0) < 0.3:
                continue
            if validation.get("aggregation_risk") == "high":
                continue

            candidate["validation"] = validation
            candidate["stability_score"] = validation.get("stability_score", 0.5)
            candidate["aggregation_risk"] = validation.get("aggregation_risk", "unknown")

            validated.append(candidate)

        return validated

    def _rerank_candidates(self, candidates: list[dict], spec: DesignSpecification) -> list[dict]:
        """Rerank candidates by combined score."""
        for candidate in candidates:
            score = 100.0

            # Affinity contribution (better affinity = higher score)
            affinity = candidate.get("predicted_affinity_nm")
            if affinity:
                if affinity < 10:  # Sub-10nM
                    score += 30
                elif affinity < 50:
                    score += 20
                elif affinity < 100:
                    score += 10
                elif affinity > spec.affinity_threshold_nm:
                    score -= 20

            # Stability contribution
            stability = candidate.get("stability_score", 0.5)
            score += stability * 20

            # Confidence contribution
            confidence = candidate.get("initial_confidence", 0.5)
            score += confidence * 10

            # Penalize for ADMET flags
            flags = candidate.get("admet_flags", [])
            score -= len(flags) * 5

            candidate["final_score"] = score

        # Sort by score (descending)
        candidates.sort(key=lambda c: c.get("final_score", 0), reverse=True)

        return candidates

    def _generate_summary(self, candidates: list[dict], duration: float) -> str:
        """Generate result summary."""
        if not candidates:
            return f"No candidates passed all filters (took {duration:.1f}s)"

        best = candidates[0]
        affinity = best.get("predicted_affinity_nm", "N/A")
        score = best.get("final_score", 0)

        return (
            f"Generated {len(candidates)} candidates in {duration:.1f}s. "
            f"Best candidate: {len(best.get('sequence', ''))} residues, "
            f"predicted affinity {affinity:.1f} nM, score {score:.0f}"
        )

    def _get_boltzgen(self):
        """Lazy-load BoltzGen."""
        if self._boltzgen is None:
            from ct.generative.boltzgen_optimizer import BoltzGenOptimizer
            self._boltzgen = BoltzGenOptimizer()
        return self._boltzgen

    def _get_esm3(self):
        """Lazy-load ESM3 client."""
        if self._esm3 is None:
            from ct.generative.esm3_client import ESM3Client
            self._esm3 = ESM3Client()
        return self._esm3

    def _get_admet(self):
        """Lazy-load ADMET predictor."""
        if self._admet is None:
            from ct.admet.predictor import ADMETPredictor
            self._admet = ADMETPredictor()
        return self._admet

    def _get_boltz2(self):
        """Lazy-load Boltz-2 optimizer."""
        if self._boltz2 is None:
            from ct.gpu_infrastructure.boltz2_optimizer import Boltz2Optimizer
            self._boltz2 = Boltz2Optimizer()
        return self._boltz2

    def _get_validator(self):
        """Lazy-load structure validator."""
        if self._validator is None:
            from ct.validation.protein_validator import ProteinValidator
            self._validator = ProteinValidator()
        return self._validator


# Convenience functions for tool registration
def design_binder(
    target_sequence: str,
    target_name: str = "",
    num_candidates: int = 10,
    **kwargs,
) -> dict:
    """
    Design protein binders against a target.

    Args:
        target_sequence: Target protein sequence
        target_name: Target identifier
        num_candidates: Number of candidates to return

    Returns:
        Dictionary with design results
    """
    pipeline = DesignPipeline()
    spec = DesignSpecification(
        target_sequence=target_sequence,
        target_name=target_name,
        num_candidates=num_candidates,
    )
    result = pipeline.run(spec)

    return {
        "summary": result.summary,
        "candidates": [
            {
                "sequence": c["sequence"],
                "predicted_affinity_nm": c.get("predicted_affinity_nm"),
                "final_score": c.get("final_score"),
                "source": c.get("source"),
            }
            for c in result.candidates
        ],
        "best_candidate": result.best_candidate,
        "design_time_seconds": result.design_time_seconds,
    }


def design_molecular_glue(
    e3_ligase: str,
    target_protein: str,
    avoid_offtargets: list[str] = None,
    **kwargs,
) -> dict:
    """
    Design molecular glue degraders.

    Args:
        e3_ligase: E3 ligase sequence (e.g., CRBN)
        target_protein: Target protein to degrade
        avoid_offtargets: List of off-target proteins to avoid

    Returns:
        Dictionary with design results
    """
    pipeline = DesignPipeline()
    spec = DesignSpecification(
        target_sequence=f"{e3_ligase}|{target_protein}",
        target_name="molecular_glue",
        design_type="glue",
        negative_constraints=[f"binds {t}" for t in (avoid_offtargets or [])],
        num_candidates=10,
    )
    result = pipeline.run(spec)

    return {
        "summary": result.summary,
        "candidates": result.candidates,
        "off_target_avoidance": avoid_offtargets,
    }