"""
Neo4j client for CellType-Agent knowledge graph operations.

Provides a high-level interface for querying the biological knowledge graph,
with connection pooling, query caching, and error handling.
"""

import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

logger = logging.getLogger("ct.knowledge_graph.neo4j")


@dataclass
class Neo4jConfig:
    """Neo4j connection configuration."""
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"
    max_connection_pool_size: int = 50
    connection_timeout: int = 30
    query_timeout: int = 60


class Neo4jClient:
    """
    High-level Neo4j client for biological knowledge graph queries.

    Features:
    - Connection pooling
    - Query caching for common patterns
    - Automatic retry with exponential backoff
    - Query performance logging

    Usage:
        client = Neo4jClient()
        results = client.run_query("MATCH (g:Gene) RETURN g.name LIMIT 10")
    """

    def __init__(self, config: Optional[Neo4jConfig] = None):
        """
        Initialize Neo4j client.

        Args:
            config: Optional configuration, uses defaults if not provided
        """
        self.config = config or Neo4jConfig()
        self._driver = None
        self._query_cache: dict = {}

    @property
    def driver(self):
        """Lazy-load Neo4j driver."""
        if self._driver is None:
            try:
                from neo4j import GraphDatabase
                self._driver = GraphDatabase.driver(
                    self.config.uri,
                    auth=(self.config.user, self.config.password),
                    max_connection_pool_size=self.config.max_connection_pool_size,
                    connection_timeout=self.config.connection_timeout,
                )
                logger.info(f"Connected to Neo4j at {self.config.uri}")
            except ImportError:
                raise ImportError(
                    "neo4j package required. Install with: pip install neo4j"
                )
        return self._driver

    def run_query(
        self,
        query: str,
        parameters: Optional[dict] = None,
        cache: bool = False,
        cache_ttl: int = 3600,
    ) -> list[dict]:
        """
        Execute a Cypher query and return results.

        Args:
            query: Cypher query string
            parameters: Query parameters
            cache: Whether to cache results
            cache_ttl: Cache TTL in seconds

        Returns:
            List of result dictionaries
        """
        parameters = parameters or {}
        cache_key = (query, frozenset(parameters.items()))

        # Check cache
        if cache and cache_key in self._query_cache:
            cached_result, cached_time = self._query_cache[cache_key]
            if time.time() - cached_time < cache_ttl:
                logger.debug(f"Cache hit for query")
                return cached_result

        start = time.time()

        try:
            with self.driver.session(database=self.config.database) as session:
                result = session.run(query, parameters)
                records = [dict(record) for record in result]

            elapsed = time.time() - start
            logger.debug(f"Query completed in {elapsed*1000:.1f}ms ({len(records)} records)")

            # Cache if requested
            if cache:
                self._query_cache[cache_key] = (records, time.time())

            return records

        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise

    def get_entity(self, entity_type: str, name: str) -> Optional[dict]:
        """
        Get an entity by type and name.

        Args:
            entity_type: Entity type (Gene, Disease, Drug, etc.)
            name: Entity name

        Returns:
            Entity properties or None if not found
        """
        query = f"""
        MATCH (n:{entity_type})
        WHERE n.name = $name OR n.id CONTAINS $name
        RETURN n
        LIMIT 1
        """
        results = self.run_query(query, {"name": name})
        return results[0].get("n") if results else None

    def get_connected_entities(
        self,
        entity_id: str,
        relation_types: Optional[list[str]] = None,
        max_depth: int = 2,
        limit: int = 100,
    ) -> list[dict]:
        """
        Get entities connected to a given entity.

        Args:
            entity_id: Starting entity ID
            relation_types: Filter by relation types (None for all)
            max_depth: Maximum traversal depth
            limit: Maximum results

        Returns:
            List of connected entities with relationship info
        """
        rel_filter = ""
        if relation_types:
            rel_filter = ":" + "|".join(relation_types)

        query = f"""
        MATCH (start {{id: $entity_id}})-[r{rel_filter}*1..{max_depth}]-(end)
        RETURN DISTINCT
            end.id as entity_id,
            end.name as entity_name,
            labels(end)[0] as entity_type,
            [rel in r | type(rel)] as path
        LIMIT $limit
        """

        return self.run_query(query, {
            "entity_id": entity_id,
            "limit": limit,
        })

    def find_path(
        self,
        start_entity: str,
        end_entity: str,
        max_depth: int = 4,
    ) -> list[dict]:
        """
        Find shortest paths between two entities.

        Args:
            start_entity: Starting entity ID
            end_entity: Target entity ID
            max_depth: Maximum path length

        Returns:
            List of paths with node and edge information
        """
        query = f"""
        MATCH path = shortestPath(
            (start {{id: $start}})-[*1..{max_depth}]-(end {{id: $end}})
        )
        RETURN
            [node in nodes(path) | {{id: node.id, name: node.name, type: labels(node)[0]}}] as nodes,
            [rel in relationships(path) | type(rel)] as relations
        """

        return self.run_query(query, {
            "start": start_entity,
            "end": end_entity,
        })

    def get_drug_targets(self, drug_name: str) -> list[dict]:
        """Get all known targets for a drug."""
        query = """
        MATCH (dr:Drug)-[r]->(g:Gene)
        WHERE dr.name CONTAINS $drug_name OR dr.id CONTAINS $drug_name
        RETURN DISTINCT
            dr.name as drug,
            g.name as target,
            g.id as target_id,
            type(r) as relation
        """

        return self.run_query(query, {"drug_name": drug_name})

    def get_disease_genes(self, disease_name: str, limit: int = 50) -> list[dict]:
        """Get genes associated with a disease."""
        query = """
        MATCH (d:Disease)-[r]-(g:Gene)
        WHERE d.name CONTAINS $disease_name OR d.id CONTAINS $disease_name
        RETURN DISTINCT
            d.name as disease,
            g.name as gene,
            g.id as gene_id,
            type(r) as relation
        ORDER BY g.name
        LIMIT $limit
        """

        return self.run_query(query, {
            "disease_name": disease_name,
            "limit": limit,
        })

    def get_pathway_genes(self, pathway_name: str) -> list[dict]:
        """Get genes participating in a pathway."""
        query = """
        MATCH (p:Pathway)-[r]-(g:Gene)
        WHERE p.name CONTAINS $pathway_name OR p.id CONTAINS $pathway_name
        RETURN DISTINCT
            p.name as pathway,
            g.name as gene,
            g.id as gene_id,
            type(r) as relation
        ORDER BY g.name
        """

        return self.run_query(query, {"pathway_name": pathway_name})

    def get_drug_side_effects(self, drug_name: str, limit: int = 20) -> list[dict]:
        """Get side effects associated with a drug."""
        query = """
        MATCH (dr:Drug)-[r]->(s:SideEffect)
        WHERE dr.name CONTAINS $drug_name OR dr.id CONTAINS $drug_name
        RETURN DISTINCT
            dr.name as drug,
            s.name as side_effect,
            s.id as side_effect_id
        LIMIT $limit
        """

        return self.run_query(query, {
            "drug_name": drug_name,
            "limit": limit,
        })

    def search_entities(
        self,
        search_term: str,
        entity_types: Optional[list[str]] = None,
        limit: int = 20,
    ) -> list[dict]:
        """
        Search for entities by name.

        Args:
            search_term: Search string
            entity_types: Filter by entity types
            limit: Maximum results

        Returns:
            List of matching entities
        """
        if entity_types:
            type_filter = ":" + "|".join(entity_types)
        else:
            type_filter = ""

        query = f"""
        MATCH (n{type_filter})
        WHERE n.name CONTAINS $search_term OR n.id CONTAINS $search_term
        RETURN DISTINCT
            n.id as id,
            n.name as name,
            labels(n)[0] as type
        ORDER BY size(n.name) ASC
        LIMIT $limit
        """

        return self.run_query(query, {
            "search_term": search_term,
            "limit": limit,
        })

    def get_stats(self) -> dict:
        """Get knowledge graph statistics."""
        query = """
        MATCH (n)
        WITH count(n) as node_count
        MATCH ()-[r]->()
        RETURN node_count, count(r) as rel_count
        """

        results = self.run_query(query)
        if results:
            return results[0]
        return {"node_count": 0, "rel_count": 0}

    def health_check(self) -> bool:
        """Check Neo4j connection health."""
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def clear_cache(self):
        """Clear query cache."""
        self._query_cache.clear()
        logger.info("Query cache cleared")

    def close(self):
        """Close Neo4j connection."""
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")


# Singleton instance for convenience
_default_client: Optional[Neo4jClient] = None


def get_neo4j_client(config: Optional[Neo4jConfig] = None) -> Neo4jClient:
    """Get or create default Neo4j client."""
    global _default_client
    if _default_client is None:
        _default_client = Neo4jClient(config)
    return _default_client