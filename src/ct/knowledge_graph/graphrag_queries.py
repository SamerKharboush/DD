"""
GraphRAG Query Templates for CellType-Agent.

Provides predefined Cypher query templates for common biological queries.
These templates cover 80%+ of use cases and eliminate the need for
complex text-to-Cypher translation in most scenarios.

Phase 1: 20-30 template queries
Phase 2: Add LLM-based text-to-Cypher for uncovered queries
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger("ct.knowledge_graph.graphrag")


@dataclass
class QueryTemplate:
    """A predefined Cypher query template."""
    name: str
    description: str
    category: str
    query: str
    parameters: dict[str, str]  # param_name -> description
    example_usage: str


class GraphRAGQueries:
    """
    Collection of predefined GraphRAG queries for biological knowledge graph.

    Usage:
        queries = GraphRAGQueries()
        result = queries.execute("drug_targets", {"drug_name": "imatinib"})
    """

    def __init__(self, neo4j_client=None):
        """
        Initialize query templates.

        Args:
            neo4j_client: Optional Neo4j client instance
        """
        from ct.knowledge_graph.neo4j_client import get_neo4j_client
        self.client = neo4j_client or get_neo4j_client()
        self._templates = self._build_templates()

    def _build_templates(self) -> dict[str, QueryTemplate]:
        """Build all query templates."""
        templates = {}

        # ============================================================
        # DRUG QUERIES
        # ============================================================

        templates["drug_targets"] = QueryTemplate(
            name="drug_targets",
            description="Get all known protein targets for a drug",
            category="drug",
            query="""
            MATCH (dr:Drug)-[r]->(g:Gene)
            WHERE toLower(dr.name) CONTAINS toLower($drug_name)
               OR toLower(dr.id) CONTAINS toLower($drug_name)
            RETURN DISTINCT
                dr.name as drug,
                dr.id as drug_id,
                g.name as target,
                g.id as target_id,
                type(r) as relation
            ORDER BY g.name
            """,
            parameters={"drug_name": "Drug name or ID (e.g., 'imatinib', 'DB00619')"},
            example_usage="What proteins does imatinib bind to?"
        )

        templates["drug_diseases"] = QueryTemplate(
            name="drug_diseases",
            description="Get diseases treated by a drug",
            category="drug",
            query="""
            MATCH (dr:Drug)-[r]->(d:Disease)
            WHERE toLower(dr.name) CONTAINS toLower($drug_name)
            RETURN DISTINCT
                dr.name as drug,
                d.name as disease,
                d.id as disease_id,
                type(r) as relation
            ORDER BY d.name
            """,
            parameters={"drug_name": "Drug name or ID"},
            example_usage="What diseases does sotorasib treat?"
        )

        templates["drug_side_effects"] = QueryTemplate(
            name="drug_side_effects",
            description="Get side effects associated with a drug",
            category="drug",
            query="""
            MATCH (dr:Drug)-[r]->(s:SideEffect)
            WHERE toLower(dr.name) CONTAINS toLower($drug_name)
            RETURN DISTINCT
                dr.name as drug,
                s.name as side_effect,
                s.id as side_effect_id,
                count(r) as evidence_count
            ORDER BY evidence_count DESC
            LIMIT $limit
            """,
            parameters={
                "drug_name": "Drug name or ID",
                "limit": "Maximum number of results (default: 20)"
            },
            example_usage="What are the side effects of lenalidomide?"
        )

        templates["similar_drugs"] = QueryTemplate(
            name="similar_drugs",
            description="Find drugs that share targets with a given drug",
            category="drug",
            query="""
            MATCH (dr1:Drug)-[r1]->(g:Gene)<-[r2]-(dr2:Drug)
            WHERE toLower(dr1.name) CONTAINS toLower($drug_name)
              AND dr1 <> dr2
            WITH dr2, collect(DISTINCT g.name) as shared_targets, count(DISTINCT g) as overlap
            ORDER BY overlap DESC
            LIMIT $limit
            RETURN
                dr2.name as similar_drug,
                dr2.id as drug_id,
                overlap,
                shared_targets
            """,
            parameters={
                "drug_name": "Drug name",
                "limit": "Maximum results (default: 10)"
            },
            example_usage="Find drugs similar to imatinib"
        )

        # ============================================================
        # TARGET/GENE QUERIES
        # ============================================================

        templates["gene_diseases"] = QueryTemplate(
            name="gene_diseases",
            description="Get diseases associated with a gene",
            category="gene",
            query="""
            MATCH (g:Gene)-[r]-(d:Disease)
            WHERE toLower(g.name) = toLower($gene_name)
               OR toLower(g.id) CONTAINS toLower($gene_name)
            RETURN DISTINCT
                g.name as gene,
                d.name as disease,
                d.id as disease_id,
                type(r) as relation
            ORDER BY d.name
            """,
            parameters={"gene_name": "Gene symbol (e.g., 'KRAS', 'TP53')"},
            example_usage="What diseases are associated with BRCA1?"
        )

        templates["gene_pathways"] = QueryTemplate(
            name="gene_pathways",
            description="Get pathways that a gene participates in",
            category="gene",
            query="""
            MATCH (g:Gene)-[r]-(p:Pathway)
            WHERE toLower(g.name) = toLower($gene_name)
            RETURN DISTINCT
                g.name as gene,
                p.name as pathway,
                p.id as pathway_id,
                type(r) as relation
            ORDER BY p.name
            """,
            parameters={"gene_name": "Gene symbol"},
            example_usage="What pathways is EGFR involved in?"
        )

        templates["gene_interactions"] = QueryTemplate(
            name="gene_interactions",
            description="Get genes that interact with a given gene",
            category="gene",
            query="""
            MATCH (g1:Gene)-[r]-(g2:Gene)
            WHERE toLower(g1.name) = toLower($gene_name)
              AND g1 <> g2
            RETURN DISTINCT
                g2.name as interacting_gene,
                g2.id as gene_id,
                type(r) as relation,
                count(r) as evidence
            ORDER BY evidence DESC
            LIMIT $limit
            """,
            parameters={
                "gene_name": "Gene symbol",
                "limit": "Maximum results (default: 20)"
            },
            example_usage="What proteins interact with KRAS?"
        )

        templates["gene_drugs"] = QueryTemplate(
            name="gene_drugs",
            description="Get drugs that target a gene",
            category="gene",
            query="""
            MATCH (g:Gene)<-[r]-(dr:Drug)
            WHERE toLower(g.name) = toLower($gene_name)
            RETURN DISTINCT
                g.name as target,
                dr.name as drug,
                dr.id as drug_id,
                type(r) as relation
            ORDER BY dr.name
            """,
            parameters={"gene_name": "Gene symbol"},
            example_usage="What drugs target EGFR?"
        )

        # ============================================================
        # PATHWAY QUERIES
        # ============================================================

        templates["pathway_genes"] = QueryTemplate(
            name="pathway_genes",
            description="Get all genes in a pathway",
            category="pathway",
            query="""
            MATCH (p:Pathway)-[r]-(g:Gene)
            WHERE toLower(p.name) CONTAINS toLower($pathway_name)
               OR toLower(p.id) CONTAINS toLower($pathway_name)
            RETURN DISTINCT
                p.name as pathway,
                g.name as gene,
                g.id as gene_id,
                type(r) as relation
            ORDER BY g.name
            """,
            parameters={"pathway_name": "Pathway name (e.g., 'MAPK signaling')"},
            example_usage="What genes are in the PI3K pathway?"
        )

        templates["pathway_drugs"] = QueryTemplate(
            name="pathway_drugs",
            description="Get drugs that target genes in a pathway",
            category="pathway",
            query="""
            MATCH (p:Pathway)-[r1]-(g:Gene)<-[r2]-(dr:Drug)
            WHERE toLower(p.name) CONTAINS toLower($pathway_name)
            RETURN DISTINCT
                p.name as pathway,
                g.name as target,
                dr.name as drug,
                dr.id as drug_id,
                type(r2) as drug_target_relation
            ORDER BY dr.name
            """,
            parameters={"pathway_name": "Pathway name"},
            example_usage="What drugs target the MAPK pathway?"
        )

        # ============================================================
        # DISEASE QUERIES
        # ============================================================

        templates["disease_genes"] = QueryTemplate(
            name="disease_genes",
            description="Get genes associated with a disease",
            category="disease",
            query="""
            MATCH (d:Disease)-[r]-(g:Gene)
            WHERE toLower(d.name) CONTAINS toLower($disease_name)
               OR toLower(d.id) CONTAINS toLower($disease_name)
            RETURN DISTINCT
                d.name as disease,
                g.name as gene,
                g.id as gene_id,
                type(r) as relation
            ORDER BY g.name
            LIMIT $limit
            """,
            parameters={
                "disease_name": "Disease name (e.g., 'breast cancer', 'NSCLC')",
                "limit": "Maximum results (default: 50)"
            },
            example_usage="What genes are mutated in lung cancer?"
        )

        templates["disease_drugs"] = QueryTemplate(
            name="disease_drugs",
            description="Get drugs used to treat a disease",
            category="disease",
            query="""
            MATCH (d:Disease)<-[r]-(dr:Drug)
            WHERE toLower(d.name) CONTAINS toLower($disease_name)
            RETURN DISTINCT
                d.name as disease,
                dr.name as drug,
                dr.id as drug_id,
                type(r) as relation
            ORDER BY dr.name
            """,
            parameters={"disease_name": "Disease name"},
            example_usage="What drugs are approved for AML?"
        )

        # ============================================================
        # COMPLEX QUERIES (Multi-hop)
        # ============================================================

        templates["drug_target_pathway"] = QueryTemplate(
            name="drug_target_pathway",
            description="Get pathways affected by a drug through its targets",
            category="complex",
            query="""
            MATCH (dr:Drug)-[r1]->(g:Gene)-[r2]-(p:Pathway)
            WHERE toLower(dr.name) CONTAINS toLower($drug_name)
            RETURN DISTINCT
                dr.name as drug,
                g.name as target,
                p.name as pathway,
                p.id as pathway_id
            ORDER BY p.name
            """,
            parameters={"drug_name": "Drug name"},
            example_usage="What pathways does imatinib affect?"
        )

        templates["resistance_mechanisms"] = QueryTemplate(
            name="resistance_mechanisms",
            description="Find genes that may confer resistance to a drug",
            category="complex",
            query="""
            MATCH (dr:Drug)-[r1]->(g1:Gene)-[r2]-(g2:Gene)-[r3]-(d:Disease)
            WHERE toLower(dr.name) CONTAINS toLower($drug_name)
              AND g1 <> g2
            WITH g2, collect(DISTINCT d.name) as associated_diseases, count(DISTINCT d) as disease_count
            ORDER BY disease_count DESC
            LIMIT $limit
            RETURN
                g2.name as potential_resistance_gene,
                g2.id as gene_id,
                disease_count,
                associated_diseases
            """,
            parameters={
                "drug_name": "Drug name",
                "limit": "Maximum results (default: 10)"
            },
            example_usage="What genes might cause resistance to osimertinib?"
        )

        templates["combination_targets"] = QueryTemplate(
            name="combination_targets",
            description="Find genes that could be targeted in combination with a primary target",
            category="complex",
            query="""
            MATCH (g1:Gene)-[r1]-(p:Pathway)-[r2]-(g2:Gene)
            WHERE toLower(g1.name) = toLower($target_gene)
              AND g1 <> g2
              AND NOT (g2)-[:DRUG_TARGET]-(:Drug)
            WITH g2, collect(DISTINCT p.name) as shared_pathways, count(DISTINCT p) as pathway_overlap
            ORDER BY pathway_overlap DESC
            LIMIT $limit
            RETURN
                g2.name as combination_target,
                g2.id as gene_id,
                pathway_overlap,
                shared_pathways
            """,
            parameters={
                "target_gene": "Primary target gene symbol",
                "limit": "Maximum results (default: 10)"
            },
            example_usage="What genes could be combined with KRAS targeting?"
        )

        templates["off_target_path"] = QueryTemplate(
            name="off_target_path",
            description="Find potential off-target effects by pathway proximity",
            category="complex",
            query="""
            MATCH path = shortestPath(
                (g1:Gene)-[*1..3]-(g2:Gene)
            )
            WHERE toLower(g1.name) = toLower($target_gene)
              AND g1 <> g2
              AND g2:Gene
            RETURN DISTINCT
                g2.name as potential_off_target,
                g2.id as gene_id,
                [n in nodes(path) | n.name] as path,
                length(path) as distance
            ORDER BY distance
            LIMIT $limit
            """,
            parameters={
                "target_gene": "Target gene symbol",
                "limit": "Maximum results (default: 20)"
            },
            example_usage="What are potential off-targets for a KRAS inhibitor?"
        )

        # ============================================================
        # SEARCH QUERIES
        # ============================================================

        templates["search_entities"] = QueryTemplate(
            name="search_entities",
            description="Search for entities by name",
            category="search",
            query="""
            MATCH (n)
            WHERE toLower(n.name) CONTAINS toLower($search_term)
               OR toLower(n.id) CONTAINS toLower($search_term)
            RETURN DISTINCT
                n.id as id,
                n.name as name,
                labels(n)[0] as type
            ORDER BY size(n.name) ASC
            LIMIT $limit
            """,
            parameters={
                "search_term": "Search string",
                "limit": "Maximum results (default: 20)"
            },
            example_usage="Search for entities containing 'kinase'"
        )

        return templates

    def list_templates(self, category: Optional[str] = None) -> list[QueryTemplate]:
        """
        List available query templates.

        Args:
            category: Optional category filter

        Returns:
            List of query templates
        """
        templates = list(self._templates.values())
        if category:
            templates = [t for t in templates if t.category == category]
        return templates

    def get_template(self, name: str) -> Optional[QueryTemplate]:
        """Get a specific query template by name."""
        return self._templates.get(name)

    def execute(
        self,
        template_name: str,
        parameters: dict[str, Any],
    ) -> list[dict]:
        """
        Execute a query template with parameters.

        Args:
            template_name: Name of the query template
            parameters: Query parameters

        Returns:
            Query results
        """
        template = self._templates.get(template_name)
        if not template:
            raise ValueError(f"Unknown template: {template_name}")

        # Set default parameters
        params = dict(parameters)
        if "limit" in template.parameters and "limit" not in params:
            params["limit"] = 20

        # Execute query
        return self.client.run_query(template.query, params)

    def find_matching_template(self, query: str) -> Optional[str]:
        """
        Find a template that matches a natural language query.

        Simple keyword matching - Phase 2 will add LLM-based matching.

        Args:
            query: Natural language query

        Returns:
            Template name or None
        """
        query_lower = query.lower()

        # Drug-related queries
        if any(word in query_lower for word in ["drug", "compound", "inhibitor", "molecule"]):
            if any(word in query_lower for word in ["target", "bind", "binds"]):
                return "drug_targets"
            if any(word in query_lower for word in ["disease", "treat", "indication"]):
                return "drug_diseases"
            if any(word in query_lower for word in ["side effect", "adverse", "toxicity"]):
                return "drug_side_effects"
            if any(word in query_lower for word in ["similar", "like"]):
                return "similar_drugs"

        # Gene/target queries
        if any(word in query_lower for word in ["gene", "protein", "target"]):
            if any(word in query_lower for word in ["disease", "associated", "mutation"]):
                return "gene_diseases"
            if any(word in query_lower for word in ["pathway", "pathways"]):
                return "gene_pathways"
            if any(word in query_lower for word in ["interact", "interacts", "complex"]):
                return "gene_interactions"
            if any(word in query_lower for word in ["drug", "inhibitor", "compound"]):
                return "gene_drugs"

        # Pathway queries
        if any(word in query_lower for word in ["pathway", "signaling"]):
            if any(word in query_lower for word in ["gene", "protein"]):
                return "pathway_genes"
            if any(word in query_lower for word in ["drug", "target"]):
                return "pathway_drugs"

        # Disease queries
        if any(word in query_lower for word in ["disease", "cancer", "tumor", "indication"]):
            if any(word in query_lower for word in ["gene", "mutation", "altered"]):
                return "disease_genes"
            if any(word in query_lower for word in ["drug", "treatment", "therapy"]):
                return "disease_drugs"

        # Complex queries
        if any(word in query_lower for word in ["resistance", "resistant"]):
            return "resistance_mechanisms"
        if any(word in query_lower for word in ["combination", "combine"]):
            return "combination_targets"
        if any(word in query_lower for word in ["off-target", "off target"]):
            return "off_target_path"

        return None

    def execute_natural_language(
        self,
        query: str,
        entities: Optional[dict[str, str]] = None,
    ) -> tuple[Optional[list[dict]], Optional[str]]:
        """
        Execute a natural language query using template matching.

        Args:
            query: Natural language query
            entities: Optional pre-extracted entities (drug_name, gene_name, etc.)

        Returns:
            Tuple of (results, template_name) or (None, None) if no match
        """
        template_name = self.find_matching_template(query)
        if not template_name:
            return None, None

        template = self._templates[template_name]

        # Build parameters from entities or try to extract from query
        params = entities or self._extract_entities(query, template)

        if not params:
            return None, template_name

        results = self.execute(template_name, params)
        return results, template_name

    def _extract_entities(self, query: str, template: QueryTemplate) -> dict:
        """
        Simple entity extraction for query parameters.

        Phase 2 will use NER model or LLM.
        """
        params = {}
        query_words = query.split()

        for param_name, description in template.parameters.items():
            if param_name == "limit":
                params[param_name] = 20
                continue

            # Simple heuristic: look for quoted strings or capitalized words
            import re
            quoted = re.findall(r'"([^"]+)"', query)
            if quoted:
                params[param_name] = quoted[0]
                break

            # Look for capitalized terms
            caps = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', query)
            if caps:
                params[param_name] = caps[0]
                break

        return params

    def query(
        self,
        query: str,
        limit: int = 20,
    ) -> list[dict]:
        """
        Execute a natural language query.

        Convenience method for the API.

        Args:
            query: Natural language query
            limit: Maximum results

        Returns:
            List of results
        """
        results, template_name = self.execute_natural_language(query, {"limit": limit})
        if results is None:
            # Return empty results with message
            return {
                "results": [],
                "query": query,
                "message": "No matching query template found",
            }
        return results