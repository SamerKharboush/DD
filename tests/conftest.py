"""
Pytest configuration for CellType-Agent tests.
"""

import os
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def mock_anthropic_key(monkeypatch):
    """Set mock API key for testing."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key-12345")


@pytest.fixture
def mock_neo4j_config(monkeypatch):
    """Set mock Neo4j config for testing."""
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "test-password")


@pytest.fixture
def sample_smiles():
    """Sample SMILES strings for testing."""
    return {
        "ethanol": "CCO",
        "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "caffeine": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
        "sotorasib": "CC(C)C[C@H](N)C(=O)Nc1ccc(F)cc1C(=O)Nc2nccc(N3CCN(C)CC3)n2",
    }


@pytest.fixture
def sample_queries():
    """Sample queries for testing."""
    return [
        "What drugs target KRAS?",
        "Design a KRAS G12C inhibitor",
        "Predict ADMET for aspirin",
        "What are the side effects of sotorasib?",
    ]


@pytest.fixture
def sample_session_data():
    """Sample session data for testing."""
    return {
        "session_id": "test-session-123",
        "query": "What drugs target KRAS?",
        "response": "Several drugs target KRAS, including sotorasib and adagrasib...",
        "tool_calls": [
            {
                "tool": "knowledge.query",
                "params": {"query": "KRAS inhibitors"},
                "result": {"results": [{"drug": "Sotorasib"}]},
            }
        ],
        "rating": 5,
    }


@pytest.fixture
def temp_session_dir(tmp_path):
    """Create a temporary directory for session logs."""
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    return session_dir


@pytest.fixture
def mock_tool_registry():
    """Mock tool registry for testing."""
    from ct.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry._initialized = True
    return registry