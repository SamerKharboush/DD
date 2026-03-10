"""
Feedback Collector for CellType-Agent RLEF.

Collects and manages user feedback on agent predictions for
Reinforcement Learning from Experimental Feedback (RLEF).
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from ct.session_logging.trace_store import TraceStore

logger = logging.getLogger("ct.session_logging.feedback")


class FeedbackOutcome(Enum):
    """Possible feedback outcomes."""
    VALIDATED = "validated"
    PARTIALLY_VALIDATED = "partially_validated"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"


@dataclass
class FeedbackEntry:
    """A single feedback entry."""
    session_id: str
    outcome: FeedbackOutcome
    rating: Optional[int]  # 1-5 stars
    feedback_text: Optional[str]
    corrected_conclusion: Optional[str]
    timestamp: str
    user_id: Optional[str] = None


class FeedbackCollector:
    """
    Collects and manages user feedback for RLEF.

    Provides multiple feedback collection methods:
    1. Simple rating (1-5 stars)
    2. Outcome labels (validated/refuted)
    3. Detailed text feedback
    4. Corrected conclusions

    Usage:
        collector = FeedbackCollector()
        collector.add_rating("session-123", 4)
        collector.add_outcome("session-123", FeedbackOutcome.VALIDATED)
        collector.add_feedback("session-123", "Great analysis, but missed SALL4 risk")
    """

    OUTCOME_PROMPTS = {
        FeedbackOutcome.VALIDATED: (
            "Prediction was confirmed by experimental data or literature.",
            "Use this when the agent's conclusion matches reality."
        ),
        FeedbackOutcome.PARTIALLY_VALIDATED: (
            "Prediction was partially correct.",
            "Use this when some aspects were right but others were wrong."
        ),
        FeedbackOutcome.REFUTED: (
            "Prediction was contradicted by evidence.",
            "Use this when the agent's conclusion was definitively wrong."
        ),
        FeedbackOutcome.INCONCLUSIVE: (
            "Unable to determine if prediction was correct.",
            "Use this when evidence is insufficient."
        ),
    }

    def __init__(self, trace_store: Optional[TraceStore] = None):
        """
        Initialize feedback collector.

        Args:
            trace_store: Optional trace store for persistence
        """
        self.trace_store = trace_store or TraceStore()

    def add_rating(
        self,
        session_id: str,
        rating: int,
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Add a star rating for a session.

        Args:
            session_id: Session ID
            rating: Rating from 1-5
            user_id: Optional user identifier

        Returns:
            True if successful
        """
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be between 1 and 5")

        return self.trace_store.update_feedback(
            session_id,
            rating=rating,
        )

    def add_outcome(
        self,
        session_id: str,
        outcome: FeedbackOutcome,
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Add an outcome label for a session.

        Args:
            session_id: Session ID
            outcome: Feedback outcome
            user_id: Optional user identifier

        Returns:
            True if successful
        """
        return self.trace_store.update_feedback(
            session_id,
            outcome=outcome.value,
        )

    def add_feedback(
        self,
        session_id: str,
        feedback_text: str,
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Add detailed text feedback.

        Args:
            session_id: Session ID
            feedback_text: Detailed feedback
            user_id: Optional user identifier

        Returns:
            True if successful
        """
        return self.trace_store.update_feedback(
            session_id,
            feedback=feedback_text,
        )

    def add_corrected_conclusion(
        self,
        session_id: str,
        corrected_conclusion: str,
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Add a corrected conclusion.

        Args:
            session_id: Session ID
            corrected_conclusion: What the correct conclusion should have been
            user_id: Optional user identifier

        Returns:
            True if successful
        """
        # Store in feedback field with marker
        feedback = f"[CORRECTED CONCLUSION]\n{corrected_conclusion}"
        return self.trace_store.update_feedback(
            session_id,
            feedback=feedback,
        )

    def get_feedback_prompt(self) -> str:
        """
        Get a formatted feedback prompt for the user.

        Returns:
            Formatted prompt string
        """
        prompt = """
## Rate this result

Please provide feedback on the agent's response:

**Rating (1-5 stars):**
- ⭐⭐⭐⭐⭐ (5): Perfect - exactly what I needed
- ⭐⭐⭐⭐ (4): Great - mostly correct with minor issues
- ⭐⭐⭐ (3): Okay - partially helpful
- ⭐⭐ (2): Poor - significant issues
- ⭐ (1): Wrong - not helpful at all

**Outcome:**
- `[validated]` Prediction was confirmed by evidence
- `[partially_validated]` Some aspects were correct, others wrong
- `[refuted]` Prediction was definitively wrong
- `[inconclusive]` Unable to determine correctness

**Additional feedback (optional):**
What worked well? What could be improved?
"""
        return prompt

    def get_quick_feedback_options(self) -> list[dict]:
        """
        Get quick feedback options for UI.

        Returns:
            List of feedback options with labels and values
        """
        return [
            {"label": "✅ Correct", "value": "validated", "rating": 5},
            {"label": "🔶 Partially correct", "value": "partially_validated", "rating": 3},
            {"label": "❌ Incorrect", "value": "refuted", "rating": 1},
            {"label": "❓ Not sure", "value": "inconclusive", "rating": None},
        ]

    def parse_feedback_command(self, command: str) -> Optional[dict]:
        """
        Parse a feedback command string.

        Supports formats like:
        - "5" -> rating
        - "validated" -> outcome
        - "3 partially_validated good analysis" -> combined
        - "rating:4 outcome:validated" -> explicit

        Args:
            command: User command string

        Returns:
            Parsed feedback dictionary or None
        """
        command = command.strip().lower()
        result = {}

        # Try to parse explicit format
        if ":" in command:
            parts = command.split()
            for part in parts:
                if ":" in part:
                    key, value = part.split(":", 1)
                    if key == "rating":
                        try:
                            result["rating"] = int(value)
                        except ValueError:
                            pass
                    elif key == "outcome":
                        try:
                            result["outcome"] = FeedbackOutcome(value)
                        except ValueError:
                            pass
                    elif key == "feedback":
                        result["feedback_text"] = value
            return result if result else None

        # Try simple formats
        parts = command.split(maxsplit=2)

        for part in parts:
            # Try rating
            try:
                rating = int(part)
                if 1 <= rating <= 5:
                    result["rating"] = rating
                    continue
            except ValueError:
                pass

            # Try outcome
            try:
                outcome = FeedbackOutcome(part)
                result["outcome"] = outcome
                continue
            except ValueError:
                pass

            # Treat as feedback text
            if "feedback_text" not in result:
                result["feedback_text"] = part

        return result if result else None

    def get_training_pairs(self) -> list[tuple[dict, str]]:
        """
        Get validated/refuted pairs for DPO training.

        Returns:
            List of (validated_trace, refuted_trace) pairs
        """
        validated = self.trace_store.get_traces_by_outcome(
            FeedbackOutcome.VALIDATED.value, limit=1000
        )
        refuted = self.trace_store.get_traces_by_outcome(
            FeedbackOutcome.REFUTED.value, limit=1000
        )

        # For DPO, we need pairs where the query is similar
        # This is a simplified implementation
        pairs = []

        for v_trace in validated:
            # Find similar refuted traces (by query similarity)
            # For now, just pair randomly
            if refuted:
                r_trace = refuted.pop()
                pairs.append((v_trace, r_trace))

        return pairs

    def get_stats(self) -> dict:
        """Get feedback collection statistics."""
        store_stats = self.trace_store.get_stats()

        return {
            "total_traces": store_stats["total_traces"],
            "with_rating": sum(store_stats["rating_distribution"].values()),
            "with_outcome": sum(store_stats["outcome_distribution"].values()),
            "rating_distribution": store_stats["rating_distribution"],
            "outcome_distribution": store_stats["outcome_distribution"],
            "validation_rate": self._calculate_validation_rate(store_stats),
            "feedback_rate": self._calculate_feedback_rate(store_stats),
        }

    def _calculate_validation_rate(self, stats: dict) -> float:
        """Calculate rate of validated outcomes."""
        total = stats["total_traces"]
        if total == 0:
            return 0.0

        validated = stats["outcome_distribution"].get("validated", 0)
        partially = stats["outcome_distribution"].get("partially_validated", 0)

        return (validated + partially * 0.5) / total

    def _calculate_feedback_rate(self, stats: dict) -> float:
        """Calculate rate of feedback collection."""
        total = stats["total_traces"]
        if total == 0:
            return 0.0

        with_rating = sum(stats["rating_distribution"].values())
        with_outcome = sum(stats["outcome_distribution"].values())

        # Count unique traces with any feedback
        # This is approximate since we don't track uniqueness
        return min(1.0, (with_rating + with_outcome) / total)