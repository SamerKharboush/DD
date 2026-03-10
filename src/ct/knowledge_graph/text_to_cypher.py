"""
Text-to-Cypher Translation for CellType-Agent.

Phase 1: Template-based matching (in graphrag_queries.py)
Phase 2: LLM-based translation (this module)
Phase 3: Fine-tuned model

This module provides LLM-powered translation from natural language to Cypher
for queries not covered by templates.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger("ct.knowledge_graph.text_to_cypher")


@dataclass
class CypherTranslation:
    """Result of text-to-Cypher translation."""
    query: str
    parameters: dict[str, Any]
    confidence: float
    explanation: str
    raw_response: str


class TextToCypher:
    """
    Translate natural language queries to Cypher using LLM.

    Usage:
        translator = TextToCypher()
        result = translator.translate("What drugs target KRAS?")
        print(result.query)
    """

    SYSTEM_PROMPT = """You are a Cypher query expert for a biological knowledge graph.

The graph contains the following node types:
- Gene: proteins/genes with properties {id, name}
- Drug: small molecules/drugs with properties {id, name}
- Disease: diseases/conditions with properties {id, name}
- Pathway: biological pathways with properties {id, name}
- SideEffect: adverse events with properties {id, name}
- Anatomy: anatomical structures with properties {id, name}

Common relationship types:
- Drug->Gene (drug targets gene)
- Drug->Disease (drug treats disease)
- Drug->SideEffect (drug causes side effect)
- Gene->Disease (gene associated with disease)
- Gene->Pathway (gene participates in pathway)
- Gene->Gene (gene-gene interactions)

Rules:
1. Use MATCH for reading queries
2. Always use DISTINCT when returning multiple results
3. Use toLower() for case-insensitive matching
4. Return node properties as: n.id, n.name, labels(n)[0] as type
5. Use parameter placeholders: $param_name
6. Always include a LIMIT clause
7. Order by relevance (evidence count, etc.)

Example translations:
- "What drugs target EGFR?" -> MATCH (dr:Drug)-[r]->(g:Gene) WHERE toLower(g.name) = 'egfr' RETURN dr.name as drug
- "What diseases are associated with BRCA1?" -> MATCH (d:Disease)-[r]-(g:Gene) WHERE toLower(g.name) = 'brca1' RETURN d.name as disease

Return JSON with:
{
  "query": "CYPHER QUERY",
  "parameters": {"param_name": "value"},
  "explanation": "Brief explanation of the query",
  "confidence": 0.0-1.0
}"""

    USER_PROMPT_TEMPLATE = """Translate this natural language query to Cypher:

Query: {query}

Return valid JSON only, no markdown."""

    def __init__(self, llm_client=None, neo4j_client=None):
        """
        Initialize text-to-Cypher translator.

        Args:
            llm_client: Optional LLM client (uses default if not provided)
            neo4j_client: Optional Neo4j client for validation
        """
        self._llm_client = llm_client
        self._neo4j_client = neo4j_client

    @property
    def llm_client(self):
        """Lazy-load LLM client."""
        if self._llm_client is None:
            from ct.models.llm import get_llm_client
            self._llm_client = get_llm_client()
        return self._llm_client

    def translate(
        self,
        query: str,
        validate: bool = True,
        max_retries: int = 2,
    ) -> Optional[CypherTranslation]:
        """
        Translate natural language query to Cypher.

        Args:
            query: Natural language query
            validate: Whether to validate the generated Cypher
            max_retries: Maximum number of retries on validation failure

        Returns:
            CypherTranslation or None if translation failed
        """
        from ct.knowledge_graph.graphrag_queries import GraphRAGQueries

        # First, try template matching
        graphrag = GraphRAGQueries(self._neo4j_client)
        template_name = graphrag.find_matching_template(query)
        if template_name:
            logger.info(f"Using template: {template_name}")
            template = graphrag.get_template(template_name)
            params = graphrag._extract_entities(query, template)
            return CypherTranslation(
                query=template.query,
                parameters=params,
                confidence=0.95,
                explanation=f"Matched template: {template_name}",
                raw_response="",
            )

        # Fall back to LLM translation
        for attempt in range(max_retries + 1):
            try:
                result = self._llm_translate(query)

                if result and validate:
                    is_valid, error = self._validate_cypher(result.query)
                    if not is_valid:
                        logger.warning(f"Invalid Cypher (attempt {attempt + 1}): {error}")
                        if attempt < max_retries:
                            # Retry with error feedback
                            query_with_feedback = f"{query}\n\nPrevious attempt failed: {error}"
                            continue
                        return None

                return result

            except Exception as e:
                logger.error(f"Translation error (attempt {attempt + 1}): {e}")

        return None

    def _llm_translate(self, query: str) -> Optional[CypherTranslation]:
        """Use LLM to translate query."""
        try:
            response = self.llm_client.chat(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": self.USER_PROMPT_TEMPLATE.format(query=query)},
                ],
                temperature=0.1,
                max_tokens=500,
            )

            content = response.get("content", "")
            return self._parse_llm_response(content)

        except Exception as e:
            logger.error(f"LLM translation failed: {e}")
            return None

    def _parse_llm_response(self, content: str) -> Optional[CypherTranslation]:
        """Parse LLM response into CypherTranslation."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group())
                return CypherTranslation(
                    query=data.get("query", ""),
                    parameters=data.get("parameters", {}),
                    confidence=float(data.get("confidence", 0.5)),
                    explanation=data.get("explanation", ""),
                    raw_response=content,
                )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")

        return None

    def _validate_cypher(self, query: str) -> tuple[bool, str]:
        """
        Validate Cypher query syntax.

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Basic syntax checks
        query_upper = query.upper()

        # Must start with valid clause
        if not any(query_upper.strip().startswith(kw) for kw in ["MATCH", "WITH", "RETURN", "CALL"]):
            return False, "Query must start with MATCH, WITH, RETURN, or CALL"

        # Must have RETURN clause
        if "RETURN" not in query_upper:
            return False, "Query must have RETURN clause"

        # Check for common issues
        if "DELETE" in query_upper or "REMOVE" in query_upper or "SET" in query_upper:
            return False, "Write operations not allowed"

        # Check for balanced parentheses
        if query.count("(") != query.count(")"):
            return False, "Unbalanced parentheses"

        if query.count("[") != query.count("]"):
            return False, "Unbalanced brackets"

        # Try to parse with Neo4j (if available)
        if self._neo4j_client:
            try:
                # Use EXPLAIN to validate without executing
                self._neo4j_client.run_query(f"EXPLAIN {query}", {"limit": 1})
            except Exception as e:
                return False, f"Neo4j validation error: {str(e)[:200]}"

        return True, ""

    def translate_with_fallback(
        self,
        query: str,
        fallback_templates: Optional[list[str]] = None,
    ) -> tuple[Optional[list[dict]], str]:
        """
        Translate and execute with template fallback.

        Args:
            query: Natural language query
            fallback_templates: Templates to try if translation fails

        Returns:
            Tuple of (results, method_used)
        """
        from ct.knowledge_graph.graphrag_queries import GraphRAGQueries

        # Try translation
        translation = self.translate(query)
        if translation and translation.confidence >= 0.7:
            try:
                results = self._neo4j_client.run_query(
                    translation.query,
                    translation.parameters
                )
                return results, "llm_translation"
            except Exception as e:
                logger.error(f"Failed to execute translated query: {e}")

        # Fall back to templates
        graphrag = GraphRAGQueries(self._neo4j_client)

        # Try specific templates
        if fallback_templates:
            for template_name in fallback_templates:
                try:
                    template = graphrag.get_template(template_name)
                    if template:
                        params = graphrag._extract_entities(query, template)
                        results = graphrag.execute(template_name, params)
                        if results:
                            return results, f"template:{template_name}"
                except Exception:
                    continue

        # Try matching any template
        results, matched = graphrag.execute_natural_language(query)
        if results:
            return results, f"template:{matched}"

        return None, "failed"


def create_cypher_from_intent(
    intent: str,
    entities: dict[str, str],
    neo4j_client=None,
) -> Optional[str]:
    """
    Create Cypher query from structured intent and entities.

    This is a deterministic alternative to LLM translation
    for common query patterns.

    Args:
        intent: Query intent (e.g., 'drug_targets', 'gene_diseases')
        entities: Extracted entities (e.g., {'drug_name': 'imatinib'})
        neo4j_client: Optional Neo4j client

    Returns:
        Cypher query string or None
    """
    intent_queries = {
        "drug_targets": """
            MATCH (dr:Drug)-[r]->(g:Gene)
            WHERE toLower(dr.name) CONTAINS toLower($drug_name)
            RETURN DISTINCT dr.name as drug, g.name as target, type(r) as relation
            ORDER BY g.name
        """,
        "drug_diseases": """
            MATCH (dr:Drug)-[r]->(d:Disease)
            WHERE toLower(dr.name) CONTAINS toLower($drug_name)
            RETURN DISTINCT dr.name as drug, d.name as disease, type(r) as relation
            ORDER BY d.name
        """,
        "gene_diseases": """
            MATCH (g:Gene)-[r]-(d:Disease)
            WHERE toLower(g.name) = toLower($gene_name)
            RETURN DISTINCT g.name as gene, d.name as disease, type(r) as relation
            ORDER BY d.name
        """,
        "gene_pathways": """
            MATCH (g:Gene)-[r]-(p:Pathway)
            WHERE toLower(g.name) = toLower($gene_name)
            RETURN DISTINCT g.name as gene, p.name as pathway, type(r) as relation
            ORDER BY p.name
        """,
        "disease_genes": """
            MATCH (d:Disease)-[r]-(g:Gene)
            WHERE toLower(d.name) CONTAINS toLower($disease_name)
            RETURN DISTINCT d.name as disease, g.name as gene, type(r) as relation
            ORDER BY g.name
            LIMIT 50
        """,
        "pathway_genes": """
            MATCH (p:Pathway)-[r]-(g:Gene)
            WHERE toLower(p.name) CONTAINS toLower($pathway_name)
            RETURN DISTINCT p.name as pathway, g.name as gene, type(r) as relation
            ORDER BY g.name
        """,
    }

    return intent_queries.get(intent)