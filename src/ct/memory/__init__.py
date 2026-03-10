"""
Vector Memory Module for CellType-Agent Phase 3.

Implements session memory with vector search for:
- Cross-session knowledge persistence
- Relevant context retrieval
- Agent memory sharing
"""

from ct.memory.vector_memory import VectorMemory, get_agent_memory

__all__ = [
    "VectorMemory",
    "get_agent_memory",
]