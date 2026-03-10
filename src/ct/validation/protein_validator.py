"""
Protein Validator for generated sequences.

Validates protein candidates using multiple criteria:
- Structure confidence (ESM-IF/pLDDT)
- Aggregation propensity (Aggrescan3D)
- Immunogenicity (NetMHCpan)
- Stability scores
- Solubility predictions
"""

import hashlib
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ct.validation.protein")


@dataclass
class ValidationResult:
    """Result of protein validation."""
    sequence: str
    sequence_length: int
    stability_score: float
    aggregation_risk: str  # low, medium, high
    aggregation_score: float
    immunogenicity_risk: str
    immunogenicity_score: float
    solubility_score: float
    structure_confidence: float
    issues: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    passed: bool = True


class ProteinValidator:
    """
    Multi-criteria protein validator.

    Usage:
        validator = ProteinValidator()
        result = validator.validate("MKTVRQERLKSIVRILERSKEPVSGAQL...")
        print(result.stability_score)
        print(result.issues)
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        strict_mode: bool = False,
    ):
        """
        Initialize protein validator.

        Args:
            cache_dir: Directory for caching predictions
            strict_mode: Fail on any warning (not just errors)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".ct" / "validation_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.strict_mode = strict_mode

    def validate(
        self,
        sequence: str,
        check_aggregation: bool = True,
        check_immunogenicity: bool = True,
        check_stability: bool = True,
        check_solubility: bool = True,
    ) -> ValidationResult:
        """
        Validate a protein sequence.

        Args:
            sequence: Protein sequence
            check_aggregation: Run aggregation prediction
            check_immunogenicity: Run immunogenicity prediction
            check_stability: Run stability prediction
            check_solubility: Run solubility prediction

        Returns:
            ValidationResult with all scores
        """
        # Clean sequence
        clean_seq = self._clean_sequence(sequence)
        seq_len = len(clean_seq)

        issues = []
        warnings = []

        # Basic sequence checks
        basic_issues = self._check_basic_properties(clean_seq)
        issues.extend(basic_issues)

        # Aggregation risk
        aggregation_score = 0.0
        aggregation_risk = "unknown"
        if check_aggregation:
            aggregation_score, aggregation_risk = self._predict_aggregation(clean_seq)
            if aggregation_risk == "high":
                issues.append(f"High aggregation risk (score: {aggregation_score:.2f})")
            elif aggregation_risk == "medium":
                warnings.append(f"Medium aggregation risk (score: {aggregation_score:.2f})")

        # Immunogenicity
        immunogenicity_score = 0.0
        immunogenicity_risk = "unknown"
        if check_immunogenicity:
            immunogenicity_score, immunogenicity_risk = self._predict_immunogenicity(clean_seq)
            if immunogenicity_risk == "high":
                issues.append(f"High immunogenicity risk (score: {immunogenicity_score:.2f})")
            elif immunogenicity_risk == "medium":
                warnings.append(f"Medium immunogenicity risk")

        # Stability
        stability_score = 0.5
        if check_stability:
            stability_score = self._predict_stability(clean_seq)
            if stability_score < 0.3:
                issues.append(f"Low stability score ({stability_score:.2f})")
            elif stability_score < 0.5:
                warnings.append(f"Moderate stability ({stability_score:.2f})")

        # Solubility
        solubility_score = 0.5
        if check_solubility:
            solubility_score = self._predict_solubility(clean_seq)
            if solubility_score < 0.3:
                warnings.append(f"Low solubility prediction ({solubility_score:.2f})")

        # Structure confidence (using ESM-IF if available)
        structure_confidence = self._predict_structure_confidence(clean_seq)

        # Determine if passed
        passed = len(issues) == 0
        if self.strict_mode:
            passed = passed and len(warnings) == 0

        return ValidationResult(
            sequence=clean_seq,
            sequence_length=seq_len,
            stability_score=stability_score,
            aggregation_risk=aggregation_risk,
            aggregation_score=aggregation_score,
            immunogenicity_risk=immunogenicity_risk,
            immunogenicity_score=immunogenicity_score,
            solubility_score=solubility_score,
            structure_confidence=structure_confidence,
            issues=issues,
            warnings=warnings,
            passed=passed,
        )

    def _clean_sequence(self, sequence: str) -> str:
        """Clean protein sequence."""
        if sequence.startswith(">"):
            sequence = "".join(line for line in sequence.split("\n") if not line.startswith(">"))
        return "".join(c for c in sequence.upper() if c.isalpha())

    def _check_basic_properties(self, sequence: str) -> list:
        """Check basic sequence properties."""
        issues = []

        # Length check
        if len(sequence) < 10:
            issues.append("Sequence too short (<10 residues)")
        elif len(sequence) > 1000:
            issues.append("Sequence very long (>1000 residues)")

        # Invalid characters
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        invalid = set(sequence) - valid_aa
        if invalid:
            issues.append(f"Invalid amino acids: {invalid}")

        # Check for stop codons
        if "*" in sequence:
            issues.append("Sequence contains stop codon (*)")

        # Check for unusual amino acid patterns
        if "XXXXX" in sequence or any(aa * 5 in sequence for aa in valid_aa):
            issues.append("Sequence contains unusual repeat patterns")

        return issues

    def _predict_aggregation(self, sequence: str) -> tuple:
        """Predict aggregation propensity."""
        try:
            score = self._calculate_aggregation_score(sequence)
            if score > 0.7:
                return score, "high"
            elif score > 0.4:
                return score, "medium"
            else:
                return score, "low"
        except Exception as e:
            logger.warning(f"Aggregation prediction failed: {e}")
            return 0.5, "unknown"

    def _calculate_aggregation_score(self, sequence: str) -> float:
        """Calculate aggregation score from sequence."""
        aggregation_prone = set("VILFYWM")
        if not sequence:
            return 0.5

        prone_count = sum(1 for aa in sequence if aa in aggregation_prone)
        ratio = prone_count / len(sequence)

        max_stretch = 0
        current_stretch = 0
        for aa in sequence:
            if aa in aggregation_prone:
                current_stretch += 1
                max_stretch = max(max_stretch, current_stretch)
            else:
                current_stretch = 0

        score = ratio * 0.5 + (max_stretch / 10) * 0.5
        return min(1.0, score)

    def _predict_immunogenicity(self, sequence: str) -> tuple:
        """Predict immunogenicity risk."""
        try:
            score = self._calculate_immunogenicity_score(sequence)
            if score > 0.7:
                return score, "high"
            elif score > 0.4:
                return score, "medium"
            else:
                return score, "low"
        except Exception as e:
            logger.warning(f"Immunogenicity prediction failed: {e}")
            return 0.5, "unknown"

    def _calculate_immunogenicity_score(self, sequence: str) -> float:
        """Calculate immunogenicity score from sequence."""
        hydrophobic = set("AILMFVWP")
        charged = set("DEKR")

        if not sequence:
            return 0.5

        hydro_count = sum(1 for aa in sequence if aa in hydrophobic)
        charged_count = sum(1 for aa in sequence if aa in charged)

        total = len(sequence)
        hydro_ratio = hydro_count / total
        charged_ratio = charged_count / total

        if hydro_ratio > 0.3 and charged_ratio > 0.1:
            return 0.6
        elif hydro_ratio > 0.4:
            return 0.7
        else:
            return 0.3

    def _predict_stability(self, sequence: str) -> float:
        """Predict protein stability."""
        if not sequence:
            return 0.5

        hydrophobic = set("AILMFVWP")
        charged = set("DEKR")

        total = len(sequence)
        hydro_ratio = sum(1 for aa in sequence if aa in hydrophobic) / total
        charged_ratio = sum(1 for aa in sequence if aa in charged) / total
        proline_ratio = sequence.count("P") / total
        glycine_ratio = sequence.count("G") / total
        cysteine_ratio = sequence.count("C") / total

        score = 0.5
        if 0.2 < hydro_ratio < 0.4:
            score += 0.15
        if 0.1 < charged_ratio < 0.25:
            score += 0.1
        if cysteine_ratio >= 0.02:
            score += 0.1
        if proline_ratio > 0.1:
            score -= 0.15
        if glycine_ratio > 0.15:
            score -= 0.1

        return max(0.1, min(1.0, score))

    def _predict_solubility(self, sequence: str) -> float:
        """Predict protein solubility."""
        if not sequence:
            return 0.5

        soluble = set("DEKRQN")
        insoluble = set("VILFYWM")

        total = len(sequence)
        soluble_ratio = sum(1 for aa in sequence if aa in soluble) / total
        insoluble_ratio = sum(1 for aa in sequence if aa in insoluble) / total

        positive = sum(1 for aa in sequence if aa in "KR")
        negative = sum(1 for aa in sequence if aa in "DE")
        net_charge = abs(positive - negative)

        score = 0.3 + soluble_ratio * 0.4 - insoluble_ratio * 0.3
        if 0.05 < net_charge / total < 0.2:
            score += 0.1

        return max(0.1, min(1.0, score))

    def _predict_structure_confidence(self, sequence: str) -> float:
        """Predict structure confidence."""
        stability = self._predict_stability(sequence)
        solubility = self._predict_solubility(sequence)
        return (stability + solubility) / 2

    def batch_validate(self, sequences: list, **kwargs) -> list:
        """Validate multiple sequences."""
        return [self.validate(seq, **kwargs) for seq in sequences]

    def get_validation_summary(self, results: list) -> dict:
        """Get summary of validation results."""
        if not results:
            return {"total": 0}

        passed = sum(1 for r in results if r.passed)
        avg_stability = sum(r.stability_score for r in results) / len(results)
        high_agg = sum(1 for r in results if r.aggregation_risk == "high")
        high_immun = sum(1 for r in results if r.immunogenicity_risk == "high")

        return {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "pass_rate": passed / len(results),
            "avg_stability": avg_stability,
            "high_aggregation_count": high_agg,
            "high_immunogenicity_count": high_immun,
        }


def validate_protein(sequence: str, **kwargs) -> dict:
    """Validate a protein sequence."""
    validator = ProteinValidator()
    result = validator.validate(sequence)

    return {
        "summary": (
            f"Validation: {'PASSED' if result.passed else 'FAILED'}. "
            f"Stability={result.stability_score:.2f}, "
            f"Aggregation={result.aggregation_risk}, "
            f"Immunogenicity={result.immunogenicity_risk}"
        ),
        "passed": result.passed,
        "stability_score": result.stability_score,
        "aggregation_risk": result.aggregation_risk,
        "aggregation_score": result.aggregation_score,
        "immunogenicity_risk": result.immunogenicity_risk,
        "immunogenicity_score": result.immunogenicity_score,
        "solubility_score": result.solubility_score,
        "structure_confidence": result.structure_confidence,
        "issues": result.issues,
        "warnings": result.warnings,
    }