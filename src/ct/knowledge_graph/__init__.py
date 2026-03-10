"""
Knowledge Graph module for CellType-Agent Phase 1.

Implements DRKG integration with Neo4j for hallucination-free biological reasoning.
"""

from ct.knowledge_graph.drkg_loader import DRKGLoader
from ct.knowledge_graph.neo4j_client import Neo4jClient
from ct.knowledge_graph.graphrag_queries import GraphRAGQueries
from ct.knowledge_graph.text_to_cypher import TextToCypher

# Alias for backward compatibility
GraphRAG = GraphRAGQueries

__all__ = [
    "DRKGLoader",
    "Neo4jClient",
    "GraphRAGQueries",
    "GraphRAG",
    "TextToCypher",
]