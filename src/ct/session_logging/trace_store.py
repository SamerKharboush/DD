"""
Trace Store for CellType-Agent session data.

Provides persistent storage and retrieval of session traces for:
- RLEF training data accumulation
- Session resumption
- Analytics and debugging
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("ct.session_logging.trace_store")


@dataclass
class TraceRecord:
    """A trace record for database storage."""
    session_id: str
    query: str
    tool_calls_json: str
    conclusion: str
    outcome: Optional[str]
    user_rating: Optional[int]
    user_feedback: Optional[str]
    tokens_used: int
    cost_usd: float
    duration_seconds: float
    model_name: str
    quality_score: float
    created_at: str
    updated_at: str


class TraceStore:
    """
    SQLite-based trace store for session data.

    Provides efficient querying and aggregation of session traces.

    Usage:
        store = TraceStore()
        store.save_trace(session_trace)
        traces = store.get_traces_for_training(min_quality=0.7)
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS traces (
        session_id TEXT PRIMARY KEY,
        query TEXT NOT NULL,
        tool_calls_json TEXT,
        conclusion TEXT,
        outcome TEXT,
        user_rating INTEGER,
        user_feedback TEXT,
        tokens_used INTEGER DEFAULT 0,
        cost_usd REAL DEFAULT 0.0,
        duration_seconds REAL DEFAULT 0.0,
        model_name TEXT,
        quality_score REAL DEFAULT 0.0,
        created_at TEXT,
        updated_at TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_outcome ON traces(outcome);
    CREATE INDEX IF NOT EXISTS idx_rating ON traces(user_rating);
    CREATE INDEX IF NOT EXISTS idx_quality ON traces(quality_score);
    CREATE INDEX IF NOT EXISTS idx_created ON traces(created_at);
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize trace store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path) if db_path else Path.home() / ".ct" / "traces.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA)
        logger.info(f"Initialized trace store at {self.db_path}")

    def save_trace(self, trace: dict) -> None:
        """
        Save a session trace to the store.

        Args:
            trace: Session trace dictionary
        """
        tool_calls_json = json.dumps(trace.get("tool_calls", []))
        quality_score = self._calculate_quality_score(trace)

        record = TraceRecord(
            session_id=trace["session_id"],
            query=trace["query"],
            tool_calls_json=tool_calls_json,
            conclusion=trace.get("conclusion", ""),
            outcome=trace.get("outcome"),
            user_rating=trace.get("user_rating"),
            user_feedback=trace.get("user_feedback"),
            tokens_used=trace.get("tokens_used", 0),
            cost_usd=trace.get("cost_usd", 0.0),
            duration_seconds=trace.get("duration_seconds", 0.0),
            model_name=trace.get("model_name", ""),
            quality_score=quality_score,
            created_at=trace.get("created_at", datetime.now().isoformat()),
            updated_at=trace.get("updated_at", datetime.now().isoformat()),
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO traces (
                    session_id, query, tool_calls_json, conclusion, outcome,
                    user_rating, user_feedback, tokens_used, cost_usd,
                    duration_seconds, model_name, quality_score, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.session_id, record.query, record.tool_calls_json,
                record.conclusion, record.outcome, record.user_rating,
                record.user_feedback, record.tokens_used, record.cost_usd,
                record.duration_seconds, record.model_name, record.quality_score,
                record.created_at, record.updated_at
            ))

        logger.debug(f"Saved trace {record.session_id}")

    def get_trace(self, session_id: str) -> Optional[dict]:
        """Get a trace by session ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM traces WHERE session_id = ?",
                (session_id,)
            )
            row = cursor.fetchone()

            if row:
                return self._row_to_dict(row)
        return None

    def get_traces_for_training(
        self,
        min_quality: float = 0.5,
        min_tool_calls: int = 2,
        require_outcome: bool = False,
        limit: int = 10000,
    ) -> list[dict]:
        """
        Get traces suitable for training.

        Args:
            min_quality: Minimum quality score
            min_tool_calls: Minimum number of tool calls
            require_outcome: Require an outcome label
            limit: Maximum number of traces

        Returns:
            List of trace dictionaries
        """
        query = """
            SELECT * FROM traces
            WHERE quality_score >= ?
            ORDER BY quality_score DESC, created_at DESC
            LIMIT ?
        """

        traces = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, (min_quality, limit))

            for row in cursor:
                trace = self._row_to_dict(row)

                # Apply additional filters
                if len(trace.get("tool_calls", [])) < min_tool_calls:
                    continue
                if require_outcome and not trace.get("outcome"):
                    continue

                traces.append(trace)

        return traces

    def get_traces_by_outcome(self, outcome: str, limit: int = 100) -> list[dict]:
        """Get traces with a specific outcome."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM traces WHERE outcome = ? ORDER BY created_at DESC LIMIT ?",
                (outcome, limit)
            )
            return [self._row_to_dict(row) for row in cursor]

    def get_traces_by_rating(self, min_rating: int = 4, limit: int = 100) -> list[dict]:
        """Get traces with high user ratings."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM traces WHERE user_rating >= ? ORDER BY user_rating DESC, created_at DESC LIMIT ?",
                (min_rating, limit)
            )
            return [self._row_to_dict(row) for row in cursor]

    def update_feedback(
        self,
        session_id: str,
        rating: Optional[int] = None,
        feedback: Optional[str] = None,
        outcome: Optional[str] = None,
    ) -> bool:
        """Update feedback for a trace."""
        updates = []
        params = []

        if rating is not None:
            updates.append("user_rating = ?")
            params.append(rating)
        if feedback is not None:
            updates.append("user_feedback = ?")
            params.append(feedback)
        if outcome is not None:
            updates.append("outcome = ?")
            params.append(outcome)

        if not updates:
            return False

        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(session_id)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                f"UPDATE traces SET {', '.join(updates)} WHERE session_id = ?",
                params
            )
            return cursor.rowcount > 0

    def get_stats(self) -> dict:
        """Get trace store statistics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Total counts
            total = conn.execute("SELECT COUNT(*) as count FROM traces").fetchone()["count"]

            # Average metrics
            avg_result = conn.execute("""
                SELECT
                    AVG(tokens_used) as avg_tokens,
                    AVG(cost_usd) as avg_cost,
                    AVG(duration_seconds) as avg_duration,
                    AVG(quality_score) as avg_quality
                FROM traces
            """).fetchone()

            # Outcome distribution
            outcomes = {}
            for row in conn.execute("SELECT outcome, COUNT(*) as count FROM traces GROUP BY outcome"):
                if row["outcome"]:
                    outcomes[row["outcome"]] = row["count"]

            # Rating distribution
            ratings = {}
            for row in conn.execute("SELECT user_rating, COUNT(*) as count FROM traces GROUP BY user_rating"):
                if row["user_rating"]:
                    ratings[row["user_rating"]] = row["count"]

            return {
                "total_traces": total,
                "avg_tokens": avg_result["avg_tokens"] or 0,
                "avg_cost": avg_result["avg_cost"] or 0,
                "avg_duration": avg_result["avg_duration"] or 0,
                "avg_quality": avg_result["avg_quality"] or 0,
                "outcome_distribution": outcomes,
                "rating_distribution": ratings,
                "training_ready": len(self.get_traces_for_training(min_quality=0.5, limit=100000)),
                "high_quality": len(self.get_traces_for_training(min_quality=0.8, limit=100000)),
            }

    def export_training_data(self, output_path: Path, format: str = "jsonl") -> int:
        """
        Export traces for training.

        Args:
            output_path: Output file path
            format: "jsonl" or "json"

        Returns:
            Number of traces exported
        """
        traces = self.get_traces_for_training(min_quality=0.5, limit=100000)

        with open(output_path, "w") as f:
            if format == "jsonl":
                for trace in traces:
                    f.write(json.dumps(trace) + "\n")
            else:
                json.dump(traces, f, indent=2)

        logger.info(f"Exported {len(traces)} traces to {output_path}")
        return len(traces)

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        """Convert a database row to dictionary."""
        return {
            "session_id": row["session_id"],
            "query": row["query"],
            "tool_calls": json.loads(row["tool_calls_json"]) if row["tool_calls_json"] else [],
            "conclusion": row["conclusion"],
            "outcome": row["outcome"],
            "user_rating": row["user_rating"],
            "user_feedback": row["user_feedback"],
            "tokens_used": row["tokens_used"],
            "cost_usd": row["cost_usd"],
            "duration_seconds": row["duration_seconds"],
            "model_name": row["model_name"],
            "quality_score": row["quality_score"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _calculate_quality_score(self, trace: dict) -> float:
        """Calculate quality score for a trace."""
        score = 0.0

        # Rating bonus
        rating = trace.get("user_rating")
        if rating:
            score += 0.3
            if rating >= 4:
                score += 0.2

        # Outcome bonus
        if trace.get("outcome"):
            score += 0.2
            if trace["outcome"] == "validated":
                score += 0.1

        # Tool call complexity
        tool_calls = trace.get("tool_calls", [])
        if len(tool_calls) >= 3:
            score += 0.1
        if len(tool_calls) >= 5:
            score += 0.05

        # Has conclusion
        if trace.get("conclusion"):
            score += 0.1

        return min(1.0, score)

    def vacuum(self):
        """Vacuum the database to reclaim space."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("VACUUM")
        logger.info("Database vacuumed")