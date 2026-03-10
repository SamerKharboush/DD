"""
Tool Registry for CellType-Agent.

Central registry for all tools that agents can use.
"""

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger("ct.tools.registry")


class BaseTool:
    """Base class for all tools."""

    name: str = "base_tool"
    description: str = "Base tool class"
    parameters: dict = {}

    def run(self, **kwargs) -> Any:
        """Execute the tool."""
        raise NotImplementedError("Subclasses must implement run()")


class ToolRegistry:
    """
    Central registry for tools.

    Tools are registered by name and can be retrieved by agents.

    Usage:
        registry.register("admet.predict", ADMETPredictTool())
        tool = registry.get_tool("admet.predict")
        result = tool.run(smiles="CCO")
    """

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._initialized = False

    def register(self, name: str, tool: BaseTool) -> None:
        """Register a tool."""
        self._tools[name] = tool
        logger.debug(f"Registered tool: {name}")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        if not self._initialized:
            self._lazy_init()
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tools."""
        if not self._initialized:
            self._lazy_init()
        return list(self._tools.keys())

    def _lazy_init(self):
        """Lazy initialization of tools."""
        if self._initialized:
            return

        # Register core tools
        self._register_admet_tools()
        self._register_knowledge_tools()
        self._register_generative_tools()
        self._register_structure_tools()

        self._initialized = True
        logger.info(f"Initialized {len(self._tools)} tools")

    def _register_admet_tools(self):
        """Register ADMET prediction tools."""
        try:
            from ct.admet.predictor import ADMETPredictor

            class ADMETPredictTool(BaseTool):
                name = "admet.predict"
                description = "Predict ADMET properties for a compound"
                parameters = {"smiles": "SMILES string of compound"}

                def __init__(self):
                    self.predictor = ADMETPredictor()

                def run(self, smiles: str, **kwargs) -> dict:
                    return self.predictor.predict(smiles)

            self.register("admet.predict", ADMETPredictTool())

        except ImportError:
            logger.debug("ADMET tools not available")

        # Mock ADMET tool for when predictor not available
        class MockADMETTool(BaseTool):
            name = "admet.predict"
            description = "Mock ADMET prediction"
            parameters = {"smiles": "SMILES string"}

            def run(self, smiles: str, **kwargs) -> dict:
                return {
                    "smiles": smiles,
                    "logP": 2.5,
                    "solubility": "moderate",
                    "herg_risk": "low",
                    "overall_score": 0.75,
                }

        if "admet.predict" not in self._tools:
            self.register("admet.predict", MockADMETTool())

    def _register_knowledge_tools(self):
        """Register knowledge graph tools."""
        try:
            from ct.knowledge_graph import GraphRAG

            class KnowledgeQueryTool(BaseTool):
                name = "knowledge.query"
                description = "Query the biomedical knowledge graph"
                parameters = {"query": "Natural language query"}

                def __init__(self):
                    self.rag = GraphRAG()

                def run(self, query: str, **kwargs) -> dict:
                    results = self.rag.query(query)
                    return {"query": query, "results": results}

            self.register("knowledge.query", KnowledgeQueryTool())

        except ImportError:
            logger.debug("Knowledge graph tools not available")

        # Mock knowledge tool
        class MockKnowledgeTool(BaseTool):
            name = "knowledge.query"
            description = "Mock knowledge graph query"

            def run(self, query: str, **kwargs) -> dict:
                return {
                    "query": query,
                    "results": [
                        {"entity": "KRAS", "type": "Gene", "relevance": 0.95},
                        {"entity": "Sotorasib", "type": "Drug", "relevance": 0.92},
                    ],
                }

        if "knowledge.query" not in self._tools:
            self.register("knowledge.query", MockKnowledgeTool())

        # Additional knowledge tools
        class MockGeneDiseasesTool(BaseTool):
            name = "knowledge.get_gene_diseases"
            description = "Get diseases associated with a gene"

            def run(self, gene_name: str, **kwargs) -> dict:
                return {
                    "gene": gene_name,
                    "diseases": [
                        {"name": "Cancer", "association_score": 0.95},
                        {"name": "RASopathy", "association_score": 0.85},
                    ],
                }

        self.register("knowledge.get_gene_diseases", MockGeneDiseasesTool())

        class MockDrugTargetsTool(BaseTool):
            name = "knowledge.get_drug_targets"
            description = "Get targets for a drug"

            def run(self, drug_name: str, **kwargs) -> dict:
                return {
                    "drug": drug_name,
                    "targets": [
                        {"name": "KRAS G12C", "affinity_nm": 50},
                    ],
                }

        self.register("knowledge.get_drug_targets", MockDrugTargetsTool())

    def _register_generative_tools(self):
        """Register generative design tools."""
        class MockBoltzTool(BaseTool):
            name = "boltz2.predict_affinity"
            description = "Predict binding affinity"

            def run(self, protein_sequence: str = None, ligand_smiles: str = None, **kwargs) -> dict:
                return {
                    "affinity_nm": 50,
                    "confidence": 0.8,
                    "protein": protein_sequence[:50] if protein_sequence else "N/A",
                }

        self.register("boltz2.predict_affinity", MockBoltzTool())

        class MockDesignTool(BaseTool):
            name = "generative.design_binder"
            description = "Design a binder protein"

            def run(self, target_sequence: str = None, **kwargs) -> dict:
                return {
                    "candidates": [
                        {"sequence": "MKVLQEPTPDDVEPIVAAE", "score": 0.85},
                        {"sequence": "MKVLQEPTPDDVEPIVAAF", "score": 0.82},
                    ],
                }

        self.register("generative.design_binder", MockDesignTool())

    def _register_structure_tools(self):
        """Register structure I/O tools."""
        class MockH5ADTool(BaseTool):
            name = "structure.analyze_h5ad"
            description = "Analyze h5ad single-cell data"

            def run(self, file_path: str = None, **kwargs) -> dict:
                return {
                    "n_cells": 1000,
                    "n_genes": 20000,
                    "cell_types": ["T cell", "B cell", "NK cell"],
                }

        self.register("structure.analyze_h5ad", MockH5ADTool())

        class MockPDBTool(BaseTool):
            name = "structure.load_pdb"
            description = "Load PDB structure"

            def run(self, pdb_id: str = None, **kwargs) -> dict:
                return {
                    "pdb_id": pdb_id,
                    "n_chains": 2,
                    "n_residues": 350,
                    "resolution": 2.5,
                }

        self.register("structure.load_pdb", MockPDBTool())


# Global registry instance
registry = ToolRegistry()


def get_tool(name: str) -> Optional[BaseTool]:
    """Get a tool by name from the global registry."""
    return registry.get_tool(name)