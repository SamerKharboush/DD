"""
Tests for CellType-Agent Phase 2 components.

Run with: pytest tests/test_phase2.py -v
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
from pathlib import Path
import numpy as np


# ============================================================
# BoltzGen Tests
# ============================================================

class TestBoltzGenOptimizer:
    """Tests for BoltzGen optimizer."""

    def test_clean_sequence(self):
        """Test sequence cleaning."""
        from ct.generative.boltzgen_optimizer import BoltzGenOptimizer

        optimizer = BoltzGenOptimizer()

        # Test with FASTA header
        seq = ">TP53\nMKTM...\nMTQEG"
        clean = optimizer._clean_sequence(seq)
        assert not clean.startswith(">")
        assert "\n" not in clean

    def test_generate_placeholder_candidates(self):
        """Test placeholder generation when BoltzGen not available."""
        from ct.generative.boltzgen_optimizer import BoltzGenOptimizer

        optimizer = BoltzGenOptimizer()
        candidates = optimizer._generate_placeholder_candidates(
            "MKTVRQERLKSIVRILERSKEPVSGAQL",
            "test_target",
            5,
        )

        assert len(candidates) == 5
        for c in candidates:
            assert len(c.sequence) >= 30
            assert c.predicted_affinity_nm is not None

    def test_e3_avoid_residues(self):
        """Test E3 ligase off-target avoidance."""
        from ct.generative.boltzgen_optimizer import BoltzGenOptimizer

        optimizer = BoltzGenOptimizer()

        # CRBN should avoid SALL4 and IKZF family
        avoid = optimizer._get_e3_avoid_residues("CRBN")
        assert "SALL4" in avoid
        assert "IKZF1" in avoid


# ============================================================
# ESM3 Tests
# ============================================================

class TestESM3Client:
    """Tests for ESM3 client."""

    def test_format_prompt(self):
        """Test prompt formatting."""
        from ct.generative.esm3_client import ESM3Client

        client = ESM3Client()

        prompt = {
            "sequence": "MKT___G",
            "function": "binds CRBN",
        }

        formatted = client._format_prompt(prompt)

        assert "<sequence>" in formatted
        assert "<function>" in formatted
        assert "binds CRBN" in formatted

    def test_estimate_stability(self):
        """Test stability estimation."""
        from ct.generative.esm3_client import ESM3Client

        client = ESM3Client()

        # Well-balanced sequence
        stability = client._estimate_stability("MKTVRQERLKSIVRILERSKEPVSGAQL")
        assert 0 < stability <= 1

        # Very hydrophobic sequence
        stability_hydro = client._estimate_stability("AAAAVVVVIIII")
        assert stability_hydro < stability

    def test_mutation_generation(self):
        """Test mutation generation."""
        from ct.generative.esm3_client import ESM3Client

        client = ESM3Client()

        sequence = "MKTVRQERLKSIVRILERSKEPVSGAQL"
        mutations = client.mutate(sequence, num_mutations=2)

        assert len(mutations) == 2
        for mut in mutations:
            assert "position" in mut
            assert "original" in mut
            assert "mutant" in mut
            assert "notation" in mut

    def test_score_against_constraint(self):
        """Test constraint scoring."""
        from ct.generative.esm3_client import ESM3Client

        client = ESM3Client()

        sequence = "MKTVRQERLKSIVRILERSKEPVSGAQL"
        score = client._score_against_constraint(sequence, "binds SALL4")

        assert 0 <= score <= 1


# ============================================================
# Design Pipeline Tests
# ============================================================

class TestDesignPipeline:
    """Tests for design pipeline."""

    def test_design_specification_defaults(self):
        """Test design specification creation."""
        from ct.generative.design_pipeline import DesignSpecification

        spec = DesignSpecification(
            target_sequence="MKTVRQERLKSIVRILERSKEPVSGAQL",
        )

        assert spec.num_candidates == 10
        assert spec.length_range == (50, 150)
        assert spec.affinity_threshold_nm == 100.0
        assert spec.design_type == "binder"

    def test_rerank_candidates(self):
        """Test candidate reranking."""
        from ct.generative.design_pipeline import DesignPipeline

        pipeline = DesignPipeline()

        candidates = [
            {"sequence": "A" * 50, "predicted_affinity_nm": 50, "stability_score": 0.7},
            {"sequence": "B" * 50, "predicted_affinity_nm": 10, "stability_score": 0.9},
            {"sequence": "C" * 50, "predicted_affinity_nm": 100, "stability_score": 0.5},
        ]

        from ct.generative.design_pipeline import DesignSpecification
        spec = DesignSpecification(target_sequence="TARGET")

        ranked = pipeline._rerank_candidates(candidates, spec)

        # Best candidate (low affinity, high stability) should be first
        assert ranked[0]["predicted_affinity_nm"] == 10


# ============================================================
# Validation Tests
# ============================================================

class TestProteinValidator:
    """Tests for protein validator."""

    def test_basic_sequence_checks(self):
        """Test basic sequence validation."""
        from ct.validation.protein_validator import ProteinValidator

        validator = ProteinValidator()

        issues = validator._check_basic_properties("MKTVRQERLKSIVRILERSKEPVSGAQL")
        assert len(issues) == 0

        # Too short
        issues = validator._check_basic_properties("AAA")
        assert any("short" in i.lower() for i in issues)

        # Invalid characters
        issues = validator._check_basic_properties("MKTXXXVQER")
        assert any("invalid" in i.lower() for i in issues)

    def test_aggregation_score(self):
        """Test aggregation scoring."""
        from ct.validation.protein_validator import ProteinValidator

        validator = ProteinValidator()

        # Hydrophobic sequence (high aggregation)
        score_hydro = validator._calculate_aggregation_score("VVVVIIIIAAAALLLL")
        assert score_hydro > 0.5

        # Mixed sequence (lower aggregation)
        score_mixed = validator._calculate_aggregation_score("MKTVRQERLKSIVRI")
        assert score_mixed < score_hydro

    def test_stability_prediction(self):
        """Test stability prediction."""
        from ct.validation.protein_validator import ProteinValidator

        validator = ProteinValidator()

        # Well-balanced protein
        stability = validator._predict_stability("MKTVRQERLKSIVRILERSKEPVSGAQL")
        assert 0 < stability <= 1

    def test_solubility_prediction(self):
        """Test solubility prediction."""
        from ct.validation.protein_validator import ProteinValidator

        validator = ProteinValidator()

        # Charged/soluble sequence
        solubility = validator._predict_solubility("DEDEKRKRQNQNQNQ")
        assert solubility > 0.5

    def test_full_validation(self):
        """Test complete validation."""
        from ct.validation.protein_validator import ProteinValidator

        validator = ProteinValidator()
        result = validator.validate("MKTVRQERLKSIVRILERSKEPVSGAQL")

        assert result.sequence_length == 27
        assert result.passed
        assert 0 <= result.stability_score <= 1
        assert result.aggregation_risk in ["low", "medium", "high", "unknown"]

    def test_batch_validation(self):
        """Test batch validation."""
        from ct.validation.protein_validator import ProteinValidator

        validator = ProteinValidator()
        sequences = [
            "MKTVRQERLKSIVRILERSKEPVSGAQL",
            "AAAAVVVVIIII",  # High aggregation
            "DEDEKRKRQN",  # High solubility
        ]

        results = validator.batch_validate(sequences)
        summary = validator.get_validation_summary(results)

        assert summary["total"] == 3
        assert "pass_rate" in summary


# ============================================================
# Structure I/O Tests
# ============================================================

class TestPDBHandler:
    """Tests for PDB handler."""

    def test_create_fasta(self):
        """Test FASTA generation from structure."""
        from ct.structure_io.pdb_handler import PDBHandler, PDBStructure

        handler = PDBHandler()
        structure = PDBStructure(
            pdb_id="TEST",
            sequence="MKTVRQERLKSIVRILERSKEPVSGAQL",
            chains=["A"],
            residues=[],
            coordinates=np.array([]),
        )

        fasta = handler.to_fasta(structure)

        assert fasta.startswith(">TEST")
        assert "MKTVRQERLKSIVRILERSKEPVSGAQL" in fasta


class TestFASTAHandler:
    """Tests for FASTA handler."""

    def test_detect_sequence_type(self):
        """Test sequence type detection."""
        from ct.structure_io.fasta_handler import FASTAHandler

        handler = FASTAHandler()

        # Protein
        assert handler._detect_sequence_type("MKTVRQERLKSIVRI") == "protein"

        # DNA
        assert handler._detect_sequence_type("ATGCATGC") == "DNA"

        # RNA
        assert handler._detect_sequence_type("AUGCAUGC") == "RNA"

    def test_translate_dna(self):
        """Test DNA translation."""
        from ct.structure_io.fasta_handler import FASTAHandler

        handler = FASTAHandler()

        # Start codon ATG = M
        protein = handler.translate_dna("ATG")
        assert protein == "M"

        # Known codon
        protein = handler.translate_dna("ATGTTT")
        assert protein == "MF"

    def test_reverse_complement(self):
        """Test reverse complement."""
        from ct.structure_io.fasta_handler import FASTAHandler

        handler = FASTAHandler()

        rc = handler.reverse_complement("ATGC")
        assert rc == "GCAT"


class TestH5ADHandler:
    """Tests for h5ad handler."""

    @patch("scanpy.read_h5ad")
    def test_summarize(self, mock_read):
        """Test h5ad summarization."""
        from ct.structure_io.h5ad_handler import H5ADHandler

        # Mock AnnData
        mock_adata = MagicMock()
        mock_adata.n_obs = 1000
        mock_adata.n_vars = 2000
        mock_adata.obs.columns.tolist.return_value = ["cell_type", "batch"]
        mock_adata.var.columns.tolist.return_value = ["gene_name"]
        mock_adata.obs.__getitem__ = lambda self, key: MagicMock(
            unique=lambda: ["T cell", "B cell"],
            value_counts=lambda: {"T cell": 600, "B cell": 400}
        ) if key == "cell_type" else MagicMock()
        mock_adata.raw = None
        mock_adata.uns = {}

        mock_read.return_value = mock_adata

        handler = H5ADHandler()
        summary = handler.summarize("test.h5ad")

        assert summary.n_cells == 1000
        assert summary.n_genes == 2000


# ============================================================
# Tool Registration Tests
# ============================================================

class TestPhase2Tools:
    """Tests for Phase 2 tool registration."""

    def test_generative_tools_callable(self):
        """Test that generative tools are callable."""
        from ct.tools.phase2_tools import (
            generative_design_binder,
            generative_generate_protein,
            generative_suggest_mutations,
        )

        assert callable(generative_design_binder)
        assert callable(generative_generate_protein)
        assert callable(generative_suggest_mutations)

    def test_validation_tools_callable(self):
        """Test that validation tools are callable."""
        from ct.tools.phase2_tools import (
            validation_validate_protein,
            validation_predict_aggregation,
            validation_predict_immunogenicity,
        )

        assert callable(validation_validate_protein)
        assert callable(validation_predict_aggregation)
        assert callable(validation_predict_immunogenicity)

    def test_structure_io_tools_callable(self):
        """Test that structure I/O tools are callable."""
        from ct.tools.structure_io_tools import (
            structure_parse_pdb,
            structure_analyze_h5ad,
            structure_parse_fasta,
        )

        assert callable(structure_parse_pdb)
        assert callable(structure_analyze_h5ad)
        assert callable(structure_parse_fasta)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])