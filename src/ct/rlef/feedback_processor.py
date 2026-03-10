"""
Feedback Processor for RLEF.

Processes and manages user feedback:
- Feedback collection
- Quality filtering
- Preference pair generation
- Feedback analytics
"""

import json
import logging
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ct.rlef.feedback")


@dataclass
class FeedbackEntry:
    """A single feedback entry."""
    session_id: str
    query: str
    response: str
    rating: int  # 1-5
    tool_calls: list[dict] = field(default_factory=list)
    conclusion: str = ""
    comments: str = ""
    timestamp: float = field(default_factory=time.time)
    user_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "query": self.query,
            "response": self.response,
            "rating": self.rating,
            "tool_calls": self.tool_calls,
            "conclusion": self.conclusion,
            "comments": self.comments,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FeedbackEntry":
        return cls(
            session_id=data.get("session_id", ""),
            query=data.get("query", ""),
            response=data.get("response", ""),
            rating=data.get("rating", 0),
            tool_calls=data.get("tool_calls", []),
            conclusion=data.get("conclusion", ""),
            comments=data.get("comments", ""),
            timestamp=data.get("timestamp", time.time()),
            user_id=data.get("user_id"),
            metadata=data.get("metadata", {}),
        )


class FeedbackProcessor:
    """
    Processes and manages user feedback.

    Features:
    - Collects feedback from multiple sources
    - Filters low-quality feedback
    - Generates preference pairs
    - Provides analytics

    Usage:
        processor = FeedbackProcessor()
        processor.add_feedback(entry)
        pairs = processor.generate_preference_pairs()
        analytics = processor.get_analytics()
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        min_rating: int = 2,
        quality_threshold: float = 0.5,
    ):
        """
        Initialize feedback processor.

        Args:
            data_dir: Directory for storing feedback data
            min_rating: Minimum rating to consider valid
            quality_threshold: Minimum quality score for training
        """
        self.data_dir = Path(data_dir) if data_dir else Path.home() / ".ct" / "feedback"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.min_rating = min_rating
        self.quality_threshold = quality_threshold

        self.feedback_entries: list[FeedbackEntry] = []
        self._load_stored_feedback()

    def _load_stored_feedback(self):
        """Load previously stored feedback."""
        feedback_file = self.data_dir / "feedback.jsonl"
        if feedback_file.exists():
            with open(feedback_file) as f:
                for line in f:
                    try:
                        entry = FeedbackEntry.from_dict(json.loads(line))
                        self.feedback_entries.append(entry)
                    except json.JSONDecodeError:
                        continue

            logger.info(f"Loaded {len(self.feedback_entries)} feedback entries")

    def add_feedback(
        self,
        session_id: str,
        query: str,
        response: str,
        rating: int,
        tool_calls: Optional[list[dict]] = None,
        conclusion: str = "",
        comments: str = "",
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Add a feedback entry.

        Args:
            session_id: Session identifier
            query: User query
            response: Agent response
            rating: User rating (1-5)
            tool_calls: Tool calls made during session
            conclusion: Final conclusion
            comments: User comments
            user_id: Optional user identifier

        Returns:
            True if feedback was added successfully
        """
        if rating < self.min_rating:
            logger.debug(f"Skipping low-rated feedback: {rating}")
            return False

        entry = FeedbackEntry(
            session_id=session_id,
            query=query,
            response=response,
            rating=rating,
            tool_calls=tool_calls or [],
            conclusion=conclusion,
            comments=comments,
            user_id=user_id,
        )

        self.feedback_entries.append(entry)
        self._save_feedback(entry)

        logger.info(f"Added feedback for session {session_id}: rating={rating}")
        return True

    def _save_feedback(self, entry: FeedbackEntry):
        """Save feedback to disk."""
        feedback_file = self.data_dir / "feedback.jsonl"
        with open(feedback_file, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

    def generate_preference_pairs(
        self,
        strategy: str = "rating",
        min_gap: int = 2,
    ) -> list[tuple[dict, dict]]:
        """
        Generate preference pairs for training.

        Args:
            strategy: Pairing strategy ("rating", "similar", "temporal")
            min_gap: Minimum rating gap for pairing

        Returns:
            List of (preferred, dispreferred) tuples
        """
        pairs = []

        if strategy == "rating":
            pairs = self._generate_rating_pairs(min_gap)
        elif strategy == "similar":
            pairs = self._generate_similar_pairs()
        elif strategy == "temporal":
            pairs = self._generate_temporal_pairs()

        logger.info(f"Generated {len(pairs)} preference pairs using {strategy} strategy")
        return pairs

    def _generate_rating_pairs(self, min_gap: int) -> list[tuple[dict, dict]]:
        """Generate pairs based on rating difference."""
        # Group by similar queries
        query_groups = defaultdict(list)
        for entry in self.feedback_entries:
            query_key = entry.query.lower().strip()[:50]
            query_groups[query_key].append(entry)

        pairs = []
        for query_key, entries in query_groups.items():
            if len(entries) < 2:
                continue

            # Sort by rating
            sorted_entries = sorted(entries, key=lambda e: e.rating, reverse=True)

            # Create pairs with sufficient rating gap
            for i, high in enumerate(sorted_entries):
                for low in sorted_entries[i+1:]:
                    if high.rating - low.rating >= min_gap:
                        pairs.append((
                            high.to_dict(),
                            low.to_dict(),
                        ))

        return pairs

    def _generate_similar_pairs(self) -> list[tuple[dict, dict]]:
        """Generate pairs from similar queries using embedding similarity."""
        # This would use embeddings to find semantically similar queries
        # For now, fall back to rating pairs
        return self._generate_rating_pairs(min_gap=1)

    def _generate_temporal_pairs(self) -> list[tuple[dict, dict]]:
        """Generate pairs comparing recent vs older responses."""
        if len(self.feedback_entries) < 2:
            return []

        # Sort by timestamp
        sorted_entries = sorted(self.feedback_entries, key=lambda e: e.timestamp)

        # Compare older low-rated vs newer high-rated (improvement)
        mid_point = len(sorted_entries) // 2
        older = sorted_entries[:mid_point]
        newer = sorted_entries[mid_point:]

        pairs = []
        for old_entry in older:
            if old_entry.rating < 3:
                # Find matching newer entry with higher rating
                for new_entry in newer:
                    if (new_entry.rating >= 4 and
                        self._queries_similar(old_entry.query, new_entry.query)):
                        pairs.append((
                            new_entry.to_dict(),
                            old_entry.to_dict(),
                        ))
                        break

        return pairs

    def _queries_similar(self, query1: str, query2: str, threshold: float = 0.7) -> bool:
        """Check if two queries are semantically similar."""
        # Simple word overlap for now
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())

        if not words1 or not words2:
            return False

        overlap = len(words1 & words2)
        union = len(words1 | words2)

        return overlap / union >= threshold

    def filter_by_quality(self, min_quality: float = 0.5) -> list[FeedbackEntry]:
        """
        Filter feedback by quality score.

        Args:
            min_quality: Minimum quality score (0-1)

        Returns:
            Filtered feedback entries
        """
        filtered = []
        for entry in self.feedback_entries:
            quality = self._compute_quality_score(entry)
            if quality >= min_quality:
                filtered.append(entry)

        return filtered

    def _compute_quality_score(self, entry: FeedbackEntry) -> float:
        """Compute quality score for feedback entry."""
        score = 0.0

        # Rating contribution (0-0.4)
        score += (entry.rating / 5.0) * 0.4

        # Response length contribution (0-0.2)
        if len(entry.response) > 50:
            score += 0.2

        # Tool calls contribution (0-0.2)
        if entry.tool_calls:
            score += min(0.2, len(entry.tool_calls) * 0.05)

        # Comments contribution (0-0.2)
        if entry.comments:
            score += min(0.2, len(entry.comments) / 100)

        return min(1.0, score)

    def get_analytics(self) -> dict:
        """Get feedback analytics."""
        if not self.feedback_entries:
            return {
                "total_feedback": 0,
                "average_rating": 0,
                "rating_distribution": {},
            }

        ratings = [e.rating for e in self.feedback_entries]
        rating_dist = defaultdict(int)
        for r in ratings:
            rating_dist[r] += 1

        # Compute trends
        daily_ratings = defaultdict(list)
        for entry in self.feedback_entries:
            date = datetime.fromtimestamp(entry.timestamp).strftime("%Y-%m-%d")
            daily_ratings[date].append(entry.rating)

        trends = {
            date: statistics.mean(ratings)
            for date, ratings in sorted(daily_ratings.items())
        }

        return {
            "total_feedback": len(self.feedback_entries),
            "average_rating": statistics.mean(ratings),
            "median_rating": statistics.median(ratings),
            "std_rating": statistics.stdev(ratings) if len(ratings) > 1 else 0,
            "rating_distribution": dict(rating_dist),
            "daily_trends": trends,
            "quality_filtered_count": len(self.filter_by_quality(self.quality_threshold)),
        }

    def export_for_training(
        self,
        output_file: Path,
        format: str = "jsonl",
    ) -> int:
        """
        Export feedback for model training.

        Args:
            output_file: Output file path
            format: Export format ("jsonl", "json", "csv")

        Returns:
            Number of entries exported
        """
        # Filter by quality
        entries = self.filter_by_quality(self.quality_threshold)

        if format == "jsonl":
            with open(output_file, "w") as f:
                for entry in entries:
                    f.write(json.dumps(entry.to_dict()) + "\n")

        elif format == "json":
            data = [e.to_dict() for e in entries]
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)

        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Exported {len(entries)} entries to {output_file}")
        return len(entries)

    def clear_feedback(self):
        """Clear all stored feedback."""
        self.feedback_entries = []
        feedback_file = self.data_dir / "feedback.jsonl"
        if feedback_file.exists():
            feedback_file.unlink()
        logger.info("Cleared all feedback")


class FeedbackAggregator:
    """
    Aggregates feedback from multiple sources.
    """

    def __init__(self):
        self.sources: dict[str, FeedbackProcessor] = {}

    def register_source(self, name: str, processor: FeedbackProcessor):
        """Register a feedback source."""
        self.sources[name] = processor

    def aggregate(self) -> list[FeedbackEntry]:
        """Aggregate feedback from all sources."""
        all_entries = []
        seen_sessions = set()

        for name, processor in self.sources.items():
            for entry in processor.feedback_entries:
                if entry.session_id not in seen_sessions:
                    all_entries.append(entry)
                    seen_sessions.add(entry.session_id)

        return all_entries

    def get_unified_analytics(self) -> dict:
        """Get unified analytics across all sources."""
        analytics = {"sources": {}}

        for name, processor in self.sources.items():
            analytics["sources"][name] = processor.get_analytics()

        # Compute totals
        total_feedback = sum(
            a["total_feedback"]
            for a in analytics["sources"].values()
        )

        all_ratings = []
        for processor in self.sources.values():
            all_ratings.extend(e.rating for e in processor.feedback_entries)

        analytics["total_feedback"] = total_feedback
        if all_ratings:
            analytics["overall_average_rating"] = statistics.mean(all_ratings)

        return analytics