"""
Session Logger for CellType-Agent.

Logs complete session data including:
- User queries
- Tool calls and results
- Agent reasoning traces
- Final conclusions
- Timing and cost metrics

This data is used for:
1. RLEF training (feedback-based learning)
2. LoRA fine-tuning data collection
3. Analytics and debugging
"""

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("ct.session_logging")


@dataclass
class ToolCall:
    """Record of a single tool call."""
    tool_name: str
    parameters: dict[str, Any]
    result: dict[str, Any]
    result_summary: str
    timestamp: str
    duration_ms: float
    success: bool
    error: Optional[str] = None


@dataclass
class ReasoningStep:
    """Record of agent reasoning at each step."""
    step_number: int
    thinking: str
    decision: str
    next_action: str
    timestamp: str


@dataclass
class SessionTrace:
    """Complete trace of a session."""
    session_id: str
    query: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning_steps: list[ReasoningStep] = field(default_factory=list)
    conclusion: str = ""
    outcome: Optional[str] = None  # "validated", "partially_validated", "refuted", None
    user_feedback: Optional[str] = None
    user_rating: Optional[int] = None  # 1-5 stars
    tokens_used: int = 0
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    model_name: str = ""
    created_at: str = ""
    updated_at: str = ""
    metadata: dict = field(default_factory=dict)


class SessionLogger:
    """
    Logger for CellType-Agent sessions.

    Usage:
        logger = SessionLogger()
        logger.start_session("What drugs target EGFR?")
        logger.log_tool_call("chembl.search", {"query": "EGFR"}, result)
        logger.log_reasoning(1, "Searching ChEMBL...", "Found 500 compounds")
        logger.end_session("Found 50 approved EGFR inhibitors")
        logger.save()
    """

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        auto_save: bool = True,
        min_quality_score: float = 0.5,
    ):
        """
        Initialize session logger.

        Args:
            log_dir: Directory to store session logs
            auto_save: Automatically save after each session
            min_quality_score: Minimum score for session to be included in training data
        """
        self.log_dir = Path(log_dir) if log_dir else Path.home() / ".ct" / "sessions"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.auto_save = auto_save
        self.min_quality_score = min_quality_score

        self._current_trace: Optional[SessionTrace] = None
        self._start_time: Optional[float] = None
        self._tokens_used: int = 0
        self._cost: float = 0.0

    @property
    def current_session_id(self) -> Optional[str]:
        """Get the current session ID."""
        if self._current_trace:
            return self._current_trace.session_id
        return None

    def start_session(
        self,
        query: str,
        model_name: str = "",
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Start a new session.

        Args:
            query: User's query
            model_name: LLM model being used
            metadata: Optional metadata

        Returns:
            Session ID
        """
        self._current_trace = SessionTrace(
            session_id=str(uuid.uuid4()),
            query=query,
            model_name=model_name,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            metadata=metadata or {},
        )
        self._start_time = time.time()
        self._tokens_used = 0
        self._cost = 0.0

        logger.info(f"Started session {self._current_trace.session_id}")
        return self._current_trace.session_id

    def log_tool_call(
        self,
        tool_name: str,
        parameters: dict[str, Any],
        result: dict[str, Any],
        duration_ms: float = 0.0,
    ) -> None:
        """
        Log a tool call.

        Args:
            tool_name: Name of the tool called
            parameters: Parameters passed to the tool
            result: Tool result
            duration_ms: Duration of the call in milliseconds
        """
        if self._current_trace is None:
            logger.warning("No active session to log tool call")
            return

        # Extract summary from result
        result_summary = result.get("summary", str(result)[:500]) if result else ""

        # Determine success
        success = True
        error = None
        if result and isinstance(result, dict):
            if result.get("error") or result.get("is_error"):
                success = False
                error = result.get("error", result.get("summary", "Unknown error"))

        tool_call = ToolCall(
            tool_name=tool_name,
            parameters=parameters,
            result=result if isinstance(result, dict) else {"value": str(result)},
            result_summary=result_summary,
            timestamp=datetime.now().isoformat(),
            duration_ms=duration_ms,
            success=success,
            error=error,
        )

        self._current_trace.tool_calls.append(tool_call)
        self._current_trace.updated_at = datetime.now().isoformat()

        logger.debug(f"Logged tool call: {tool_name}")

    def log_reasoning(
        self,
        step_number: int,
        thinking: str,
        decision: str,
        next_action: str,
    ) -> None:
        """
        Log an agent reasoning step.

        Args:
            step_number: Step number in the reasoning chain
            thinking: Agent's thought process
            decision: Decision made
            next_action: What action will be taken next
        """
        if self._current_trace is None:
            logger.warning("No active session to log reasoning")
            return

        reasoning = ReasoningStep(
            step_number=step_number,
            thinking=thinking,
            decision=decision,
            next_action=next_action,
            timestamp=datetime.now().isoformat(),
        )

        self._current_trace.reasoning_steps.append(reasoning)
        self._current_trace.updated_at = datetime.now().isoformat()

    def log_tokens(self, tokens: int, cost: float = 0.0) -> None:
        """Log token usage."""
        self._tokens_used += tokens
        self._cost += cost

    def end_session(
        self,
        conclusion: str,
        outcome: Optional[str] = None,
    ) -> Optional[str]:
        """
        End the current session.

        Args:
            conclusion: Final conclusion/answer
            outcome: "validated", "partially_validated", "refuted", or None

        Returns:
            Session ID or None if no active session
        """
        if self._current_trace is None:
            logger.warning("No active session to end")
            return None

        self._current_trace.conclusion = conclusion
        self._current_trace.outcome = outcome
        self._current_trace.duration_seconds = time.time() - self._start_time
        self._current_trace.tokens_used = self._tokens_used
        self._current_trace.cost_usd = self._cost
        self._current_trace.updated_at = datetime.now().isoformat()

        session_id = self._current_trace.session_id

        if self.auto_save:
            self.save()

        logger.info(
            f"Ended session {session_id} "
            f"({len(self._current_trace.tool_calls)} tools, "
            f"{self._current_trace.duration_seconds:.1f}s)"
        )

        return session_id

    def add_feedback(
        self,
        session_id: Optional[str] = None,
        rating: Optional[int] = None,
        feedback: Optional[str] = None,
        feedback_text: Optional[str] = None,  # Alias for feedback
        outcome: Optional[str] = None,
    ) -> None:
        """
        Add user feedback to a session.

        Args:
            session_id: Session ID (uses current if None)
            rating: 1-5 star rating
            feedback: Text feedback
            feedback_text: Alias for feedback
            outcome: Outcome label
        """
        # Handle feedback_text alias
        if feedback is None and feedback_text is not None:
            feedback = feedback_text
        if session_id and self._current_trace and self._current_trace.session_id != session_id:
            # Load existing session
            trace = self.load_session(session_id)
            if trace:
                if rating is not None:
                    trace.user_rating = rating
                if feedback:
                    trace.user_feedback = feedback
                if outcome:
                    trace.outcome = outcome
                self._save_trace(trace)
                return

        # Add to current session
        if self._current_trace:
            if rating is not None:
                self._current_trace.user_rating = rating
            if feedback:
                self._current_trace.user_feedback = feedback
            if outcome:
                self._current_trace.outcome = outcome
            self._current_trace.updated_at = datetime.now().isoformat()

            if self.auto_save:
                self.save()

    def save(self) -> Optional[Path]:
        """Save the current session to disk."""
        if self._current_trace is None:
            return None

        return self._save_trace(self._current_trace)

    def _save_trace(self, trace: SessionTrace) -> Path:
        """Save a trace to disk."""
        # Create date-based subdirectory
        date_dir = self.log_dir / datetime.now().strftime("%Y-%m")
        date_dir.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        file_path = date_dir / f"{trace.session_id}.json"
        with open(file_path, "w") as f:
            json.dump(asdict(trace), f, indent=2, default=str)

        logger.debug(f"Saved session to {file_path}")
        return file_path

    def load_session(self, session_id: str) -> Optional[SessionTrace]:
        """Load a session from disk."""
        # Search for session file
        for date_dir in self.log_dir.iterdir():
            if date_dir.is_dir():
                session_file = date_dir / f"{session_id}.json"
                if session_file.exists():
                    with open(session_file) as f:
                        data = json.load(f)

                    return SessionTrace(
                        session_id=data["session_id"],
                        query=data["query"],
                        tool_calls=[ToolCall(**tc) for tc in data.get("tool_calls", [])],
                        reasoning_steps=[ReasoningStep(**rs) for rs in data.get("reasoning_steps", [])],
                        conclusion=data.get("conclusion", ""),
                        outcome=data.get("outcome"),
                        user_feedback=data.get("user_feedback"),
                        user_rating=data.get("user_rating"),
                        tokens_used=data.get("tokens_used", 0),
                        cost_usd=data.get("cost_usd", 0.0),
                        duration_seconds=data.get("duration_seconds", 0.0),
                        model_name=data.get("model_name", ""),
                        created_at=data.get("created_at", ""),
                        updated_at=data.get("updated_at", ""),
                        metadata=data.get("metadata", {}),
                    )

        return None

    def list_sessions(
        self,
        limit: int = 100,
        min_tool_calls: int = 0,
        with_feedback_only: bool = False,
    ) -> list[dict]:
        """
        List recent sessions.

        Args:
            limit: Maximum number of sessions to return
            min_tool_calls: Minimum number of tool calls
            with_feedback_only: Only include sessions with user feedback

        Returns:
            List of session summaries
        """
        sessions = []

        for date_dir in sorted(self.log_dir.iterdir(), reverse=True):
            if not date_dir.is_dir():
                continue

            for session_file in sorted(date_dir.glob("*.json"), reverse=True):
                try:
                    with open(session_file) as f:
                        data = json.load(f)

                    # Apply filters
                    if min_tool_calls > 0 and len(data.get("tool_calls", [])) < min_tool_calls:
                        continue
                    if with_feedback_only and not data.get("user_rating"):
                        continue

                    sessions.append({
                        "session_id": data["session_id"],
                        "query": data["query"][:100] + "..." if len(data["query"]) > 100 else data["query"],
                        "tool_call_count": len(data.get("tool_calls", [])),
                        "duration_seconds": data.get("duration_seconds", 0),
                        "user_rating": data.get("user_rating"),
                        "outcome": data.get("outcome"),
                        "created_at": data.get("created_at"),
                    })

                    if len(sessions) >= limit:
                        return sessions

                except Exception as e:
                    logger.warning(f"Failed to load session {session_file}: {e}")

        return sessions

    def get_training_data(
        self,
        min_quality_score: Optional[float] = None,
        limit: int = 10000,
    ) -> list[dict]:
        """
        Get sessions suitable for LoRA fine-tuning.

        Args:
            min_quality_score: Minimum quality score (uses instance default if None)
            limit: Maximum sessions to return

        Returns:
            List of training examples
        """
        min_score = min_quality_score or self.min_quality_score
        training_data = []

        for date_dir in self.log_dir.iterdir():
            if not date_dir.is_dir():
                continue

            for session_file in date_dir.glob("*.json"):
                try:
                    with open(session_file) as f:
                        data = json.load(f)

                    # Calculate quality score
                    quality_score = self._calculate_quality_score(data)
                    if quality_score < min_score:
                        continue

                    # Format for training
                    training_example = {
                        "session_id": data["session_id"],
                        "query": data["query"],
                        "tool_calls": [
                            {
                                "tool": tc["tool_name"],
                                "parameters": tc["parameters"],
                                "result_summary": tc["result_summary"],
                            }
                            for tc in data.get("tool_calls", [])
                        ],
                        "conclusion": data.get("conclusion", ""),
                        "outcome": data.get("outcome"),
                        "user_rating": data.get("user_rating"),
                        "quality_score": quality_score,
                    }

                    training_data.append(training_example)

                    if len(training_data) >= limit:
                        return training_data

                except Exception as e:
                    logger.warning(f"Failed to process session {session_file}: {e}")

        return training_data

    def _calculate_quality_score(self, session_data: dict) -> float:
        """
        Calculate quality score for a session.

        Higher score = better for training.

        Factors:
        - Has user rating
        - Rating >= 4
        - Has outcome label
        - Has conclusion
        - Multiple tool calls
        - Reasoning steps present
        """
        score = 0.0

        # User rating bonus
        rating = session_data.get("user_rating")
        if rating:
            score += 0.3
            if rating >= 4:
                score += 0.2

        # Outcome bonus
        if session_data.get("outcome"):
            score += 0.2
            if session_data["outcome"] == "validated":
                score += 0.1

        # Tool call complexity
        tool_calls = session_data.get("tool_calls", [])
        if len(tool_calls) >= 3:
            score += 0.1
        if len(tool_calls) >= 5:
            score += 0.05

        # Has conclusion
        if session_data.get("conclusion"):
            score += 0.1

        # Reasoning present
        if session_data.get("reasoning_steps"):
            score += 0.05

        return min(1.0, score)

    def get_stats(self) -> dict:
        """Get session logging statistics."""
        total_sessions = 0
        total_tool_calls = 0
        with_feedback = 0
        with_rating = 0
        outcomes = {}

        for date_dir in self.log_dir.iterdir():
            if not date_dir.is_dir():
                continue

            for session_file in date_dir.glob("*.json"):
                try:
                    with open(session_file) as f:
                        data = json.load(f)

                    total_sessions += 1
                    total_tool_calls += len(data.get("tool_calls", []))

                    if data.get("user_feedback"):
                        with_feedback += 1
                    if data.get("user_rating"):
                        with_rating += 1

                    outcome = data.get("outcome")
                    if outcome:
                        outcomes[outcome] = outcomes.get(outcome, 0) + 1

                except Exception:
                    pass

        return {
            "total_sessions": total_sessions,
            "total_tool_calls": total_tool_calls,
            "avg_tool_calls": total_tool_calls / total_sessions if total_sessions else 0,
            "with_feedback": with_feedback,
            "with_rating": with_rating,
            "feedback_rate": with_feedback / total_sessions if total_sessions else 0,
            "outcomes": outcomes,
            "training_ready": sum(1 for _ in self.get_training_data(limit=100000)),
        }

    def log_session(
        self,
        query: str,
        response: str,
        tool_calls: Optional[list[dict]] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Log a complete session in one call.

        Convenience method for simple logging.

        Args:
            query: User query
            response: Agent response
            tool_calls: List of tool call records
            metadata: Additional metadata

        Returns:
            Session ID
        """
        session_id = self.start_session(
            query=query,
            model_name=metadata.get("model", "") if metadata else "",
            metadata=metadata,
        )

        # Log tool calls
        if tool_calls:
            for tc in tool_calls:
                self.log_tool_call(
                    tool_name=tc.get("tool", "unknown"),
                    parameters=tc.get("params", {}),
                    result=tc.get("result", {}),
                    duration_ms=tc.get("duration_ms", 0),
                )

        # End session
        self.end_session(conclusion=response)

        return session_id