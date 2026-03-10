"""
Vector Memory for Agent Session Persistence.

Implements persistent memory using vector embeddings for:
- Cross-session context retrieval
- Agent knowledge sharing
- Relevant past session lookup
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("ct.memory.vector")


@dataclass
class MemoryEntry:
    """A memory entry in the vector store."""
    entry_id: str
    content: str
    embedding: Optional[list[float]] = None
    agent_role: Optional[str] = None
    session_id: Optional[str] = None
    query: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class SearchResult:
    """Result of a memory search."""
    entry: MemoryEntry
    score: float


class VectorMemory:
    """
    Vector-based memory for agent persistence.

    Features:
    - Semantic search over past sessions
    - Agent-specific memory isolation
    - Session context retrieval
    - Efficient similarity search

    Usage:
        memory = VectorMemory()
        memory.store("KRAS G12C inhibitors show promise", agent_role="chemist")
        results = memory.search("KRAS inhibitors")
    """

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        embedding_model: str = "text-embedding-3-small",
        use_qdrant: bool = False,
    ):
        """
        Initialize vector memory.

        Args:
            persist_dir: Directory for persistence
            embedding_model: Embedding model to use
            use_qdrant: Whether to use Qdrant (fallback to simple storage)
        """
        self.persist_dir = Path(persist_dir) if persist_dir else Path.home() / ".ct" / "memory"
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self.use_qdrant = use_qdrant

        self._entries: list[MemoryEntry] = []
        self._embeddings: np.ndarray = np.array([])
        self._embedding_client = None
        self._qdrant_client = None

        # Load existing entries
        self._load()

    @property
    def embedding_client(self):
        """Lazy-load embedding client."""
        if self._embedding_client is None:
            try:
                from openai import OpenAI
                self._embedding_client = OpenAI()
            except ImportError:
                logger.warning("OpenAI client not available for embeddings")
                self._embedding_client = False
        return self._embedding_client if self._embedding_client else None

    def store(
        self,
        content: str,
        agent_role: Optional[str] = None,
        session_id: Optional[str] = None,
        query: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Store a memory entry.

        Args:
            content: Content to store
            agent_role: Agent role that created this memory
            session_id: Associated session ID
            query: Original query that led to this memory
            metadata: Additional metadata

        Returns:
            Entry ID
        """
        # Generate entry ID
        entry_id = hashlib.md5(
            f"{content}{time.time()}{agent_role}".encode()
        ).hexdigest()[:16]

        # Get embedding
        embedding = self._get_embedding(content)

        # Create entry
        entry = MemoryEntry(
            entry_id=entry_id,
            content=content,
            embedding=embedding,
            agent_role=agent_role,
            session_id=session_id,
            query=query,
            metadata=metadata or {},
        )

        # Store
        self._entries.append(entry)

        # Update embeddings matrix
        if embedding:
            if len(self._embeddings) == 0:
                self._embeddings = np.array([embedding])
            else:
                self._embeddings = np.vstack([self._embeddings, embedding])

        # Persist
        self._save()

        logger.debug(f"Stored memory entry: {entry_id}")
        return entry_id

    def search(
        self,
        query: str,
        agent_role: Optional[str] = None,
        limit: int = 5,
        min_score: float = 0.5,
    ) -> list[SearchResult]:
        """
        Search for relevant memories.

        Args:
            query: Search query
            agent_role: Filter by agent role
            limit: Maximum results
            min_score: Minimum similarity score

        Returns:
            List of SearchResult
        """
        if not self._entries:
            return []

        # Get query embedding
        query_embedding = self._get_embedding(query)

        if query_embedding is None or len(self._embeddings) == 0:
            # Fallback to keyword search
            return self._keyword_search(query, agent_role, limit)

        # Calculate similarities
        query_vec = np.array(query_embedding)
        similarities = self._cosine_similarity(query_vec, self._embeddings)

        # Get top results
        results = []
        for idx, score in enumerate(similarities):
            if score < min_score:
                continue

            entry = self._entries[idx]

            # Filter by agent role
            if agent_role and entry.agent_role != agent_role:
                continue

            results.append(SearchResult(entry=entry, score=float(score)))

        # Sort by score
        results.sort(key=lambda r: r.score, reverse=True)

        return results[:limit]

    def _keyword_search(
        self,
        query: str,
        agent_role: Optional[str],
        limit: int,
    ) -> list[SearchResult]:
        """Fallback keyword search."""
        query_words = set(query.lower().split())
        results = []

        for entry in self._entries:
            if agent_role and entry.agent_role != agent_role:
                continue

            content_words = set(entry.content.lower().split())
            overlap = len(query_words & content_words)
            score = overlap / max(len(query_words), 1)

            if score > 0:
                results.append(SearchResult(entry=entry, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def _get_embedding(self, text: str) -> Optional[list[float]]:
        """Get embedding for text."""
        client = self.embedding_client
        if client is None:
            return None

        try:
            response = client.embeddings.create(
                model=self.embedding_model,
                input=text[:8000],  # Truncate if too long
            )
            return response.data[0].embedding
        except Exception as e:
            logger.warning(f"Failed to get embedding: {e}")
            return None

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity."""
        if len(vec2.shape) == 1:
            vec2 = vec2.reshape(1, -1)

        dot_product = np.dot(vec2, vec1)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2, axis=1)

        return dot_product / (norm1 * norm2 + 1e-8)

    def get_session_context(
        self,
        session_id: str,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """
        Get all memory entries for a session.

        Args:
            session_id: Session ID
            limit: Maximum entries

        Returns:
            List of memory entries
        """
        entries = [
            e for e in self._entries
            if e.session_id == session_id
        ]
        return entries[-limit:]

    def get_agent_memories(
        self,
        agent_role: str,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """
        Get recent memories for a specific agent.

        Args:
            agent_role: Agent role
            limit: Maximum entries

        Returns:
            List of memory entries
        """
        entries = [
            e for e in self._entries
            if e.agent_role == agent_role
        ]
        return entries[-limit:]

    def clear(self) -> None:
        """Clear all memories."""
        self._entries = []
        self._embeddings = np.array([])
        self._save()

    def _save(self) -> None:
        """Save memories to disk."""
        data_file = self.persist_dir / "memory.json"

        data = [
            {
                "entry_id": e.entry_id,
                "content": e.content,
                "agent_role": e.agent_role,
                "session_id": e.session_id,
                "query": e.query,
                "metadata": e.metadata,
                "created_at": e.created_at,
            }
            for e in self._entries
        ]

        with open(data_file, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        """Load memories from disk."""
        data_file = self.persist_dir / "memory.json"

        if not data_file.exists():
            return

        try:
            with open(data_file) as f:
                data = json.load(f)

            self._entries = [
                MemoryEntry(**item) for item in data
            ]

            # Rebuild embeddings
            embeddings = []
            for e in self._entries:
                if e.embedding:
                    embeddings.append(e.embedding)
                else:
                    # Generate missing embeddings
                    emb = self._get_embedding(e.content)
                    e.embedding = emb
                    if emb:
                        embeddings.append(emb)

            if embeddings:
                self._embeddings = np.array(embeddings)

            logger.info(f"Loaded {len(self._entries)} memory entries")

        except Exception as e:
            logger.error(f"Failed to load memories: {e}")

    def get_stats(self) -> dict:
        """Get memory statistics."""
        agent_counts = {}
        for entry in self._entries:
            if entry.agent_role:
                agent_counts[entry.agent_role] = agent_counts.get(entry.agent_role, 0) + 1

        return {
            "total_entries": len(self._entries),
            "by_agent": agent_counts,
            "persist_dir": str(self.persist_dir),
        }


# Singleton instance
_default_memory: Optional[VectorMemory] = None


def get_agent_memory() -> VectorMemory:
    """Get or create default memory instance."""
    global _default_memory
    if _default_memory is None:
        _default_memory = VectorMemory()
    return _default_memory


class AgentMemory:
    """
    Agent-specific memory interface.

    Provides a simpler interface for agents to store and retrieve memories.
    """

    def __init__(self, agent_role: str, memory: Optional[VectorMemory] = None):
        """
        Initialize agent memory.

        Args:
            agent_role: Agent role for filtering
            memory: Optional shared memory instance
        """
        self.agent_role = agent_role
        self.memory = memory or get_agent_memory()

    def remember(
        self,
        content: str,
        session_id: Optional[str] = None,
        query: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Store a memory."""
        return self.memory.store(
            content=content,
            agent_role=self.agent_role,
            session_id=session_id,
            query=query,
            metadata=metadata,
        )

    def recall(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:
        """Recall relevant memories."""
        results = self.memory.search(
            query=query,
            agent_role=self.agent_role,
            limit=limit,
        )
        return [
            {
                "content": r.entry.content,
                "score": r.score,
                "query": r.entry.query,
                "created_at": r.entry.created_at,
            }
            for r in results
        ]

    def get_recent(self, limit: int = 10) -> list[dict]:
        """Get recent memories for this agent."""
        entries = self.memory.get_agent_memories(self.agent_role, limit)
        return [
            {
                "content": e.content,
                "query": e.query,
                "created_at": e.created_at,
            }
            for e in entries
        ]