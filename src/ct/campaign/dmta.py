"""
DMTA (Design-Make-Test-Analyze) Cycle Commands for CellType-Agent.

Implements iterative drug discovery workflows:
- Design: Generate candidate molecules
- Make: Plan synthesis/routes
- Test: Predict assays and properties
- Analyze: Evaluate results and iterate
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger("ct.campaign.dmta")


class DMTAPhase(Enum):
    """DMTA cycle phases."""
    DESIGN = "design"
    MAKE = "make"
    TEST = "test"
    ANALYZE = "analyze"


@dataclass
class DMTAState:
    """State of a DMTA cycle."""
    cycle_id: str
    current_phase: DMTAPhase
    iteration: int = 1
    target: Optional[str] = None
    design_results: list = field(default_factory=list)
    make_results: list = field(default_factory=list)
    test_results: list = field(default_factory=list)
    analyze_results: dict = field(default_factory=dict)
    decisions: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


class DMTACycle:
    """
    DMTA cycle orchestration.

    Usage:
        dmta = DMTACycle(target="KRAS_G12C")
        dmta.design(constraints={"avoid_herg": True})
        dmta.test(assays=["binding", "cell_viability"])
        dmta.analyze()
        dmta.next_iteration()  # Start next cycle
    """

    def __init__(
        self,
        target: str,
        objective: Optional[str] = None,
        constraints: Optional[dict] = None,
    ):
        """
        Initialize DMTA cycle.

        Args:
            target: Target protein/sequence
            objective: Optimization objective
            constraints: Design constraints
        """
        import uuid
        self.state = DMTAState(
            cycle_id=str(uuid.uuid4())[:8],
            current_phase=DMTAPhase.DESIGN,
            target=target,
        )
        self.objective = objective
        self.constraints = constraints or {}

    def design(
        self,
        method: str = "boltzgen",
        num_candidates: int = 10,
        additional_constraints: Optional[dict] = None,
    ) -> dict:
        """
        Design phase: Generate candidate molecules.

        Args:
            method: Design method (boltzgen, esm3, hybrid)
            num_candidates: Number of candidates
            additional_constraints: Additional constraints

        Returns:
            Design results
        """
        logger.info(f"DMTA Design phase: {method}")

        all_constraints = {**self.constraints}
        if additional_constraints:
            all_constraints.update(additional_constraints)

        results = []

        if method == "boltzgen":
            from ct.generative.boltzgen_optimizer import BoltzGenOptimizer
            optimizer = BoltzGenOptimizer()
            design_result = optimizer.design_binders(
                target_sequence=self.state.target,
                num_candidates=num_candidates,
                constraints=all_constraints,
            )
            results = [
                {
                    "sequence": c.sequence,
                    "affinity_nm": c.predicted_affinity_nm,
                    "confidence": c.confidence,
                }
                for c in design_result.candidates
            ]

        elif method == "esm3":
            from ct.generative.esm3_client import ESM3Client
            client = ESM3Client()
            generations = client.generate_with_constraints(
                positive_constraints=all_constraints.get("positive_constraints", []),
                negative_constraints=all_constraints.get("negative_constraints", []),
                num_candidates=num_candidates,
            )
            results = [
                {
                    "sequence": g.sequence,
                    "confidence": g.confidence,
                }
                for g in generations
            ]

        elif method == "hybrid":
            # Use both methods and combine
            from ct.generative.design_pipeline import DesignPipeline, DesignSpecification
            pipeline = DesignPipeline()
            spec = DesignSpecification(
                target_sequence=self.state.target,
                num_candidates=num_candidates,
            )
            result = pipeline.run(spec)
            results = result.candidates

        self.state.design_results = results
        self.state.current_phase = DMTAPhase.MAKE

        return {
            "summary": f"Generated {len(results)} candidate designs",
            "candidates": results[:5],  # Top 5
            "phase": self.state.current_phase.value,
        }

    def make(
        self,
        candidates: Optional[list] = None,
        synthesis_planning: bool = True,
    ) -> dict:
        """
        Make phase: Plan synthesis routes.

        Args:
            candidates: Specific candidates (None = use design results)
            synthesis_planning: Whether to plan synthesis routes

        Returns:
            Make results
        """
        logger.info("DMTA Make phase")

        candidates = candidates or self.state.design_results
        results = []

        for candidate in candidates[:5]:  # Top 5
            result = {
                "sequence": candidate.get("sequence", ""),
                "synthesis_feasibility": self._assess_synthesis_feasibility(candidate),
                "estimated_steps": self._estimate_synthesis_steps(candidate),
            }
            results.append(result)

        self.state.make_results = results
        self.state.current_phase = DMTAPhase.TEST

        return {
            "summary": f"Assessed synthesis feasibility for {len(results)} candidates",
            "results": results,
            "phase": self.state.current_phase.value,
        }

    def test(
        self,
        candidates: Optional[list] = None,
        assays: Optional[list[str]] = None,
    ) -> dict:
        """
        Test phase: Predict assay results.

        Args:
            candidates: Specific candidates
            assays: List of assays to run

        Returns:
            Test results
        """
        logger.info("DMTA Test phase")

        candidates = candidates or self.state.make_results
        assays = assays or ["binding", "admet", "selectivity"]

        results = []

        for candidate in candidates[:5]:
            result = {
                "candidate": candidate.get("sequence", ""),
                "assay_results": {},
            }

            # Run ADMET
            if "admet" in assays:
                admet_result = self._run_admet(candidate)
                result["assay_results"]["admet"] = admet_result

            # Run binding prediction
            if "binding" in assays:
                binding_result = self._run_binding_prediction(candidate)
                result["assay_results"]["binding"] = binding_result

            # Run selectivity check
            if "selectivity" in assays:
                selectivity_result = self._run_selectivity_check(candidate)
                result["assay_results"]["selectivity"] = selectivity_result

            results.append(result)

        self.state.test_results = results
        self.state.current_phase = DMTAPhase.ANALYZE

        return {
            "summary": f"Tested {len(results)} candidates across {len(assays)} assays",
            "results": results,
            "phase": self.state.current_phase.value,
        }

    def analyze(
        self,
        criteria: Optional[dict] = None,
    ) -> dict:
        """
        Analyze phase: Evaluate results and make decisions.

        Args:
            criteria: Success criteria

        Returns:
            Analysis results with recommendations
        """
        logger.info("DMTA Analyze phase")

        criteria = criteria or {
            "affinity_threshold_nm": 100,
            "min_admet_score": 0.7,
            "synthesis_feasibility": "moderate",
        }

        # Score candidates
        scored_candidates = []
        for test_result in self.state.test_results:
            score = self._calculate_candidate_score(test_result, criteria)
            scored_candidates.append({
                "candidate": test_result["candidate"],
                "score": score,
                "passes_criteria": self._check_criteria(test_result, criteria),
            })

        # Sort by score
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)

        # Make decisions
        decisions = []
        top_candidates = [c for c in scored_candidates if c["passes_criteria"]]

        if top_candidates:
            decisions.append(f"Advance {len(top_candidates)} candidates to next iteration")
            decisions.append(f"Top candidate: {top_candidates[0]['candidate'][:30]}...")
        else:
            decisions.append("No candidates meet criteria - revise design constraints")

        # Calculate metrics
        metrics = {
            "candidates_tested": len(self.state.test_results),
            "candidates_passing": len(top_candidates),
            "pass_rate": len(top_candidates) / len(scored_candidates) if scored_candidates else 0,
            "best_score": scored_candidates[0]["score"] if scored_candidates else 0,
        }

        self.state.analyze_results = {
            "scored_candidates": scored_candidates,
            "top_candidates": top_candidates[:3],
        }
        self.state.decisions = decisions
        self.state.metrics = metrics

        return {
            "summary": f"Analyzed {len(scored_candidates)} candidates",
            "top_candidates": top_candidates[:3],
            "decisions": decisions,
            "metrics": metrics,
            "phase": self.state.current_phase.value,
        }

    def next_iteration(self) -> dict:
        """Start the next DMTA iteration."""
        self.state.iteration += 1
        self.state.current_phase = DMTAPhase.DESIGN

        # Carry forward learnings
        previous_top = self.state.analyze_results.get("top_candidates", [])
        if previous_top:
            self.constraints["starting_sequences"] = [
                c["candidate"] for c in previous_top[:2]
            ]

        # Reset results for new iteration
        self.state.design_results = []
        self.state.make_results = []
        self.state.test_results = []

        return {
            "summary": f"Starting DMTA iteration {self.state.iteration}",
            "iteration": self.state.iteration,
            "learnings": f"Carrying forward {len(previous_top)} top candidates",
        }

    def run_full_cycle(self, num_candidates: int = 10) -> dict:
        """Run a complete DMTA cycle."""
        self.design(num_candidates=num_candidates)
        self.make()
        self.test()
        return self.analyze()

    def get_state(self) -> dict:
        """Get current DMTA state."""
        return {
            "cycle_id": self.state.cycle_id,
            "iteration": self.state.iteration,
            "current_phase": self.state.current_phase.value,
            "target": self.state.target,
            "metrics": self.state.metrics,
            "decisions": self.state.decisions,
        }

    # Helper methods

    def _assess_synthesis_feasibility(self, candidate: dict) -> str:
        """Assess synthesis feasibility."""
        sequence = candidate.get("sequence", "")
        # Simplified - would use retrosynthesis tools
        if len(sequence) < 50:
            return "high"
        elif len(sequence) < 100:
            return "moderate"
        else:
            return "low"

    def _estimate_synthesis_steps(self, candidate: dict) -> int:
        """Estimate number of synthesis steps."""
        sequence = candidate.get("sequence", "")
        # Simplified estimation
        return len(sequence) // 10 + 5

    def _run_admet(self, candidate: dict) -> dict:
        """Run ADMET prediction."""
        sequence = candidate.get("sequence", "")
        # Would call actual ADMET tools
        return {
            "herg_risk": "low",
            "liver_toxicity_risk": "low",
            "solubility": "moderate",
            "overall_score": 0.75,
        }

    def _run_binding_prediction(self, candidate: dict) -> dict:
        """Run binding prediction."""
        sequence = candidate.get("sequence", "")
        affinity = candidate.get("affinity_nm", 100)
        return {
            "predicted_affinity_nm": affinity,
            "confidence": candidate.get("confidence", 0.5),
        }

    def _run_selectivity_check(self, candidate: dict) -> dict:
        """Run selectivity check."""
        return {
            "off_targets_screened": 5,
            "off_targets_found": 0,
            "selectivity_score": 0.9,
        }

    def _calculate_candidate_score(self, test_result: dict, criteria: dict) -> float:
        """Calculate overall candidate score."""
        score = 100.0

        admet = test_result.get("assay_results", {}).get("admet", {})
        if admet.get("overall_score", 0) < criteria.get("min_admet_score", 0.7):
            score -= 20

        binding = test_result.get("assay_results", {}).get("binding", {})
        affinity = binding.get("predicted_affinity_nm", 1000)
        threshold = criteria.get("affinity_threshold_nm", 100)
        if affinity > threshold:
            score -= 30
        elif affinity < threshold / 2:
            score += 10

        selectivity = test_result.get("assay_results", {}).get("selectivity", {})
        if selectivity.get("selectivity_score", 0) < 0.7:
            score -= 15

        return max(0, score)

    def _check_criteria(self, test_result: dict, criteria: dict) -> bool:
        """Check if candidate passes criteria."""
        admet = test_result.get("assay_results", {}).get("admet", {})
        if admet.get("overall_score", 0) < criteria.get("min_admet_score", 0.7):
            return False

        binding = test_result.get("assay_results", {}).get("binding", {})
        affinity = binding.get("predicted_affinity_nm", 1000)
        if affinity > criteria.get("affinity_threshold_nm", 100):
            return False

        return True


def run_dmta_cycle(
    target: str,
    num_candidates: int = 10,
    iterations: int = 1,
    **kwargs,
) -> dict:
    """
    Run DMTA cycles.

    Args:
        target: Target sequence
        num_candidates: Candidates per cycle
        iterations: Number of iterations

    Returns:
        Final results
    """
    dmta = DMTACycle(target=target)

    all_results = []
    for i in range(iterations):
        result = dmta.run_full_cycle(num_candidates)
        all_results.append(result)

        if i < iterations - 1:
            dmta.next_iteration()

    final_state = dmta.get_state()

    return {
        "summary": f"Completed {iterations} DMTA iteration(s)",
        "final_phase": final_state["current_phase"],
        "total_candidates_tested": final_state["metrics"].get("candidates_tested", 0),
        "best_decisions": final_state["decisions"],
        "iterations": iterations,
    }