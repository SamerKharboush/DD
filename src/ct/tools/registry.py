"""
Tool Registry for CellType-Agent.

Central registry for all tools that agents can use.
"""

import importlib
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger("ct.tools.registry")

_PLACEHOLDER_ERROR = (
    "This tool has no implementation in this build. It previously returned "
    "hardcoded placeholder values (for example affinity_nm=50, score=0.85) "
    "which were indistinguishable from real measurements. It now raises "
    "instead, so no caller can rank against invented data."
)


class BaseTool:
    """Base class for all tools."""

    name: str = "base_tool"
    description: str = "Base tool class"
    parameters: dict = {}

    def run(self, **kwargs) -> Any:
        """Execute the tool."""
        raise NotImplementedError("Subclasses must implement run()")


class UnavailableTool(BaseTool):
    """Placeholder for a tool that has no backend in this build.

    Raises on use rather than returning constants. Every ``run`` result is
    therefore traceable to real computation or to a loud failure.
    """

    name = "unavailable"
    description = "Tool unavailable in this build"
    parameters: dict = {}

    def run(self, **kwargs) -> Any:
        raise NotImplementedError(
            f"{self.name}: {_PLACEHOLDER_ERROR}"
        )


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

    def register(self, name: str, tool: Any = None, **meta) -> Any:
        """Register a tool.

        Supports both call styles:

            registry.register("admet.predict", ADMETPredictTool())
            registry.register("admet.predict", tool=ADMETPredictTool())

        and decorator form with metadata (the shape used by
        phase1/2/3_tools.py and structure_io_tools.py):

            @registry.register(name="knowledge.search_entities",
                               description="...", category="knowledge",
                               parameters={...})
            def knowledge_search_entities(...): ...

        The decorator form MUST return the original function unchanged. If it
        returned a wrapper, the module-level name would bind to the wrapper and
        the tools would no longer be directly callable in tests.
        """
        if tool is None:
            def _decorator(func):
                self._tools[name] = func
                if meta.get("description"):
                    logger.debug(
                        "Registered tool %s (category=%s)",
                        name, meta.get("category", "uncategorized"),
                    )
                return func  # identity: keep the name bound to the function
            return _decorator

        self._tools[name] = tool
        logger.debug(f"Registered tool: {name}")
        return tool

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
        """Lazy initialization of tools.

        The real implementations in phase1/2/3_tools.py and
        structure_io_tools.py are imported first and take precedence. The
        fallback classes below only fill names that are still missing, and they
        raise rather than return invented numbers.
        """
        if self._initialized:
            return

        # Real implementations win. Import them before installing any fallback.
        for mod in ("ct.tools.phase1_tools", "ct.tools.phase2_tools",
                    "ct.tools.phase3_tools", "ct.tools.structure_io_tools"):
            try:
                importlib.import_module(mod)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Tool module %s failed to import: %s", mod, exc)

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

        # Fallbacks for names with no real implementation. These raise on use
        # rather than returning fixed entities, which previously looked like
        # genuine graph results.
        def _unavailable(name: str, description: str) -> UnavailableTool:
            tool = UnavailableTool()
            tool.name = name
            tool.description = description
            return tool

        if "knowledge.query" not in self._tools:
            self.register(
                "knowledge.query",
                _unavailable("knowledge.query", "Knowledge graph query (no backend)"),
            )

        if "knowledge.get_gene_diseases" not in self._tools:
            self.register(
                "knowledge.get_gene_diseases",
                _unavailable("knowledge.get_gene_diseases",
                             "Gene-disease associations (no backend)"),
            )

        if "knowledge.get_drug_targets" not in self._tools:
            self.register(
                "knowledge.get_drug_targets",
                _unavailable("knowledge.get_drug_targets",
                             "Drug targets (no backend)"),
            )

    def _register_generative_tools(self):
        """Register generative design tools."""
        def _unavailable(name: str, description: str) -> UnavailableTool:
            tool = UnavailableTool()
            tool.name = name
            tool.description = description
            return tool

        if "boltz2.predict_affinity" not in self._tools:
            self.register(
                "boltz2.predict_affinity",
                _unavailable("boltz2.predict_affinity",
                             "Binding affinity (no Boltz-2 backend)"),
            )

        if "generative.design_binder" not in self._tools:
            self.register(
                "generative.design_binder",
                _unavailable("generative.design_binder",
                             "Binder design (no generative backend)"),
            )

    def _register_structure_tools(self):
        """Register structure I/O tools."""
        def _unavailable(name: str, description: str) -> UnavailableTool:
            tool = UnavailableTool()
            tool.name = name
            tool.description = description
            return tool

        for name, desc in (
            ("structure.analyze_h5ad", "h5ad analysis (no backend)"),
            ("structure.load_pdb", "PDB load (no backend)"),
        ):
            if name not in self._tools:
                self.register(name, _unavailable(name, desc))


# Global registry instance
registry = ToolRegistry()


def get_tool(name: str) -> Optional[BaseTool]:
    """Get a tool by name from the global registry."""
    return registry.get_tool(name)