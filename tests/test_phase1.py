"""
Tests for CellType-Agent Phase 1 components.

Run with: pytest tests/ -v
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
from pathlib import Path


# ============================================================
# Knowledge Graph Tests
# ============================================================

class TestDRKGLoader:
    """Tests for DRKG loader."""

    @patch("ct.knowledge_graph.drkg_loader.DRKGLoader.download")
    def test_load_dataframes(self, mock_download):
        """Test loading DRKG dataframes."""
        from ct.knowledge_graph.drkg_loader import DRKGLoader

        # Mock download
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_download.return_value = Path(tmpdir)

            # Create mock files
            drkg_dir = Path(tmpdir) / "drkg"
            drkg_dir.mkdir()

            # Write entity file
            entity_file = drkg_dir / "entity_dict.txt"
            entity_file.write_text("Gene::TP53\tTP53\nGene::KRAS\tKRAS\n")

            # Write relations file
            rel_file = drkg_dir / "drkg.tsv"
            rel_file.write_text("Gene::TP53\tinteracts\tGene::MDM2\n")

            loader = DRKGLoader(data_dir=Path(tmpdir))
            entities_df, relations_df = loader.load_dataframes()

            assert len(entities_df) == 2
            assert len(relations_df) == 1

    def test_clean_sequence(self):
        """Test protein sequence cleaning."""
        from ct.knowledge_graph.drkg_loader import DRKGLoader

        loader = DRKGLoader()

        # Test with FASTA header
        seq = ">TP53\nMKTM...\nMTQEG"
        clean = loader._clean_sequence(seq)
        assert not clean.startswith(">")
        assert "\n" not in clean


class TestNeo4jClient:
    """Tests for Neo4j client."""

    @patch("neo4j.GraphDatabase.driver")
    def test_run_query(self, mock_driver):
        """Test query execution."""
        from ct.knowledge_graph.neo4j_client import Neo4jClient, Neo4jConfig

        # Mock driver
        mock_session = MagicMock()
        mock_result = [Mock(data=lambda: {"name": "TP53", "type": "Gene"})]
        mock_session.run.return_value = mock_result

        mock_driver_instance = MagicMock()
        mock_driver_instance.session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_driver_instance.session.return_value.__exit__ = Mock(return_value=None)
        mock_driver.return_value = mock_driver_instance

        client = Neo4jClient()
        results = client.run_query("MATCH (g:Gene) RETURN g.name as name, g.type as type")

        assert len(results) == 1


class TestGraphRAGQueries:
    """Tests for GraphRAG query templates."""

    def test_template_count(self):
        """Test that we have enough templates."""
        from ct.knowledge_graph.graphrag_queries import GraphRAGQueries

        queries = GraphRAGQueries(neo4j_client=Mock())
        templates = queries.list_templates()

        # Should have at least 20 templates
        assert len(templates) >= 20

    def test_find_matching_template(self):
        """Test template matching."""
        from ct.knowledge_graph.graphrag_queries import GraphRAGQueries

        queries = GraphRAGQueries(neo4j_client=Mock())

        # Test various queries
        assert queries.find_matching_template("What drugs target KRAS?") == "drug_targets"
        assert queries.find_matching_template("Side effects of imatinib") == "drug_side_effects"
        assert queries.find_matching_template("Genes in MAPK pathway") == "pathway_genes"


# ============================================================
# ADMET Tests
# ============================================================

class TestADMETEndpoints:
    """Tests for ADMET endpoint definitions."""

    def test_endpoint_count(self):
        """Test that we have 41 endpoints."""
        from ct.admet.endpoints import ADMET_ENDPOINTS

        assert len(ADMET_ENDPOINTS) == 41

    def test_critical_endpoints(self):
        """Test critical endpoint list."""
        from ct.admet.endpoints import CRITICAL_ENDPOINTS

        assert "herg_inhibitor" in CRITICAL_ENDPOINTS
        assert "bbb_permeability" in CRITICAL_ENDPOINTS
        assert len(CRITICAL_ENDPOINTS) >= 8

    def test_endpoint_categories(self):
        """Test endpoint categories."""
        from ct.admet.endpoints import ADMET_ENDPOINTS, EndpointCategory

        # Check toxicity endpoints
        toxicity_endpoints = [
            e for e in ADMET_ENDPOINTS.values()
            if e.category == EndpointCategory.TOXICITY
        ]
        assert len(toxicity_endpoints) == 14


class TestADMETPredictor:
    """Tests for ADMET predictor."""

    def test_validate_smiles(self):
        """Test SMILES validation."""
        from ct.admet.predictor import ADMETPredictor

        predictor = ADMETPredictor()

        assert predictor._validate_smiles("CCO")  # Ethanol
        assert predictor._validate_smiles("c1ccccc1")  # Benzene
        assert not predictor._validate_smiles("")
        assert not predictor._validate_smiles("invalid\nsmiles")

    def test_generate_flags(self):
        """Test flag generation."""
        from ct.admet.predictor import ADMETPredictor

        predictor = ADMETPredictor()

        # Mock predictions with issues
        predictions = {
            "herg_inhibitor": 0.8,  # High risk
            "bbb_permeability": 0.9,  # Good
        }
        uncertainties = {"herg_inhibitor": 0.1, "bbb_permeability": 0.1}

        flags = predictor._generate_flags(predictions, uncertainties)

        assert any("HIGH RISK" in f for f in flags)
        assert any("hERG" in f for f in flags)

    def test_calculate_confidence(self):
        """Test confidence calculation."""
        from ct.admet.predictor import ADMETPredictor

        predictor = ADMETPredictor()

        # Low uncertainty = high confidence
        predictions = {"herg_inhibitor": 0.5}
        uncertainties = {"herg_inhibitor": 0.1}

        confidence = predictor._calculate_confidence(predictions, uncertainties)
        assert confidence > 0.8

        # High uncertainty = low confidence
        uncertainties = {"herg_inhibitor": 0.5}
        confidence = predictor._calculate_confidence(predictions, uncertainties)
        assert confidence < 0.6


# ============================================================
# Session Logging Tests
# ============================================================

class TestSessionLogger:
    """Tests for session logger."""

    def test_start_end_session(self):
        """Test session lifecycle."""
        from ct.session_logging.logger import SessionLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SessionLogger(log_dir=Path(tmpdir), auto_save=False)

            session_id = logger.start_session("What drugs target KRAS?")
            assert session_id is not None

            logger.log_tool_call("chembl.search", {"query": "KRAS"}, {"summary": "Found 50 drugs"})
            logger.log_reasoning(1, "Searching ChEMBL", "Found drugs", "Analyze results")

            ended_id = logger.end_session("Found 50 approved KRAS inhibitors")
            assert ended_id == session_id

    def test_add_feedback(self):
        """Test adding feedback."""
        from ct.session_logging.logger import SessionLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SessionLogger(log_dir=Path(tmpdir), auto_save=False)

            logger.start_session("Test query")
            logger.add_feedback(rating=4, feedback="Good analysis", outcome="validated")

            assert logger._current_trace.user_rating == 4
            assert logger._current_trace.outcome == "validated"

    def test_quality_score(self):
        """Test quality score calculation."""
        from ct.session_logging.logger import SessionLogger

        logger = SessionLogger()

        # High quality session
        session = {
            "user_rating": 5,
            "outcome": "validated",
            "tool_calls": ["tool1", "tool2", "tool3"],
            "conclusion": "Answer",
            "reasoning_steps": ["step1"],
        }

        score = logger._calculate_quality_score(session)
        assert score >= 0.8


class TestTraceStore:
    """Tests for trace store."""

    def test_save_and_retrieve(self):
        """Test saving and retrieving traces."""
        from ct.session_logging.trace_store import TraceStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = TraceStore(db_path=Path(tmpdir) / "traces.db")

            trace = {
                "session_id": "test-123",
                "query": "Test query",
                "tool_calls": [{"tool": "test"}],
                "conclusion": "Test conclusion",
                "outcome": "validated",
                "user_rating": 5,
                "tokens_used": 1000,
                "cost_usd": 0.05,
                "duration_seconds": 10.0,
                "model_name": "claude-opus-4-6",
            }

            store.save_trace(trace)

            retrieved = store.get_trace("test-123")
            assert retrieved is not None
            assert retrieved["query"] == "Test query"
            assert retrieved["user_rating"] == 5


class TestFeedbackCollector:
    """Tests for feedback collector."""

    def test_parse_feedback_command(self):
        """Test parsing feedback commands."""
        from ct.session_logging.feedback_collector import FeedbackCollector, FeedbackOutcome

        collector = FeedbackCollector()

        # Test rating only
        result = collector.parse_feedback_command("4")
        assert result["rating"] == 4

        # Test outcome only
        result = collector.parse_feedback_command("validated")
        assert result["outcome"] == FeedbackOutcome.VALIDATED

        # Test combined
        result = collector.parse_feedback_command("5 validated great analysis")
        assert result["rating"] == 5
        assert result["outcome"] == FeedbackOutcome.VALIDATED
        assert "great analysis" in result.get("feedback_text", "")


# ============================================================
# GPU Infrastructure Tests
# ============================================================

class TestGPUResourceManager:
    """Tests for GPU resource manager."""

    @patch("subprocess.run")
    def test_detect_gpus(self, mock_run):
        """Test GPU detection."""
        from ct.gpu_infrastructure.resource_manager import GPUResourceManager, GPUStatus

        # Mock nvidia-smi output
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "0, NVIDIA A100-SXM4-80GB, 81920, 1000, 80920, 10, 35\n"
        mock_run.return_value = mock_result

        manager = GPUResourceManager()
        gpus = manager.detect_gpus()

        assert len(gpus) == 1
        assert gpus[0].name == "NVIDIA A100-SXM4-80GB"
        assert gpus[0].vram_total_mb == 81920

    def test_estimate_vram_for_boltz2(self):
        """Test VRAM estimation."""
        from ct.gpu_infrastructure.resource_manager import GPUResourceManager

        manager = GPUResourceManager()

        # Small protein
        vram = manager.estimate_vram_for_boltz2(100, has_ligand=False)
        assert vram >= 8000  # Minimum

        # Large protein
        vram = manager.estimate_vram_for_boltz2(1000, has_ligand=True)
        assert vram > 10000


class TestBatchProcessor:
    """Tests for batch processor."""

    def test_estimate_duration(self):
        """Test duration estimation."""
        from ct.gpu_infrastructure.batch_processor import BatchProcessor

        processor = BatchProcessor()

        # Mock processor function
        def mock_processor(items, **kwargs):
            return items

        duration = processor.estimate_duration(1000, mock_processor, batch_size=32)

        assert duration > 0
        assert duration < 3600  # Less than an hour for 1000 items


# ============================================================
# Integration Tests
# ============================================================

class TestPhase1Integration:
    """Integration tests for Phase 1 components."""

    def test_tool_registration(self):
        """Test that all tools are registered."""
        from ct.tools.phase1_tools import (
            knowledge_search_entities,
            knowledge_get_drug_targets,
            admet_predict,
            boltz2_predict_affinity,
            session_feedback,
        )

        # All functions should be callable
        assert callable(knowledge_search_entities)
        assert callable(knowledge_get_drug_targets)
        assert callable(admet_predict)
        assert callable(boltz2_predict_affinity)
        assert callable(session_feedback)

    def test_end_to_end_admet_workflow(self):
        """Test end-to-end ADMET workflow."""
        from ct.admet.predictor import ADMETPredictor

        predictor = ADMETPredictor()

        # Predict for ethanol
        result = predictor.predict("CCO")

        assert result.smiles == "CCO"
        assert result.confidence >= 0
        assert isinstance(result.flags, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])