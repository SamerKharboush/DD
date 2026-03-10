"""
Error Handling and Validation for CellType-Agent.

Provides:
- Custom exceptions
- Input validation
- Rate limiting
- Retry logic
"""

import functools
import logging
import time
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger("ct.utils.errors")

T = TypeVar("T")


# ============================================
# Custom Exceptions
# ============================================

class CelltypeAgentError(Exception):
    """Base exception for CellType-Agent."""
    pass


class ValidationError(CelltypeAgentError):
    """Input validation error."""
    def __init__(self, message: str, field: Optional[str] = None):
        self.field = field
        super().__init__(message)


class ModelNotFoundError(CelltypeAgentError):
    """Model not found error."""
    pass


class ToolExecutionError(CelltypeAgentError):
    """Tool execution error."""
    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' failed: {message}")


class KnowledgeGraphError(CelltypeAgentError):
    """Knowledge graph error."""
    pass


class LLMError(CelltypeAgentError):
    """LLM API error."""
    def __init__(self, provider: str, message: str):
        self.provider = provider
        super().__init__(f"{provider} API error: {message}")


class RateLimitError(LLMError):
    """Rate limit exceeded error."""
    def __init__(self, provider: str, retry_after: Optional[int] = None):
        self.retry_after = retry_after
        super().__init__(provider, f"Rate limit exceeded. Retry after {retry_after}s")


class GPUNotAvailableError(CelltypeAgentError):
    """GPU not available error."""
    pass


# ============================================
# Input Validation
# ============================================

def validate_smiles(smiles: str) -> bool:
    """
    Validate SMILES string.

    Args:
        smiles: SMILES string to validate

    Returns:
        True if valid

    Raises:
        ValidationError: If SMILES is invalid
    """
    if not smiles:
        raise ValidationError("SMILES cannot be empty", "smiles")

    # Basic validation
    if len(smiles) > 10000:
        raise ValidationError("SMILES too long (max 10000 chars)", "smiles")

    # Check for invalid characters
    invalid_chars = set(smiles) - set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789[]()=#-+@/\\%.*:~")
    if invalid_chars and invalid_chars != {' '}:
        logger.warning(f"Unusual characters in SMILES: {invalid_chars}")

    # Try to parse with RDKit if available
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValidationError(f"Invalid SMILES: {smiles[:50]}...", "smiles")
    except ImportError:
        pass

    return True


def validate_protein_sequence(sequence: str) -> bool:
    """
    Validate protein sequence.

    Args:
        sequence: Amino acid sequence

    Returns:
        True if valid

    Raises:
        ValidationError: If sequence is invalid
    """
    if not sequence:
        raise ValidationError("Sequence cannot be empty", "sequence")

    # Check length
    if len(sequence) > 10000:
        raise ValidationError("Sequence too long (max 10000 residues)", "sequence")

    # Check for valid amino acids
    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")  # 20 standard amino acids
    sequence_upper = sequence.upper().replace(" ", "").replace("\n", "")

    invalid = set(sequence_upper) - valid_aas
    if invalid:
        raise ValidationError(f"Invalid amino acids: {invalid}", "sequence")

    return True


def validate_query(query: str) -> bool:
    """
    Validate user query.

    Args:
        query: User query

    Returns:
        True if valid

    Raises:
        ValidationError: If query is invalid
    """
    if not query or not query.strip():
        raise ValidationError("Query cannot be empty", "query")

    if len(query) > 100000:
        raise ValidationError("Query too long (max 100000 chars)", "query")

    return True


def validate_rating(rating: int) -> bool:
    """
    Validate rating (1-5).

    Args:
        rating: Rating value

    Returns:
        True if valid

    Raises:
        ValidationError: If rating is invalid
    """
    if not isinstance(rating, int):
        raise ValidationError("Rating must be an integer", "rating")

    if rating < 1 or rating > 5:
        raise ValidationError("Rating must be between 1 and 5", "rating")

    return True


# ============================================
# Retry Logic
# ============================================

def retry_on_error(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    Decorator for retrying on errors.

    Args:
        max_retries: Maximum number of retries
        delay: Initial delay between retries
        backoff: Backoff multiplier
        exceptions: Exceptions to catch

    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff

            raise last_exception

        return wrapper
    return decorator


def retry_with_fallback(
    primary_func: Callable[..., T],
    fallback_func: Callable[..., T],
    max_retries: int = 2,
    exceptions: tuple = (Exception,),
) -> T:
    """
    Execute primary function with fallback.

    Args:
        primary_func: Primary function to try
        fallback_func: Fallback function if primary fails
        max_retries: Retries for primary
        exceptions: Exceptions to catch

    Returns:
        Result from primary or fallback
    """
    try:
        return retry_on_error(max_retries=max_retries, exceptions=exceptions)(primary_func)()
    except Exception as e:
        logger.warning(f"Primary function failed: {e}. Using fallback.")
        return fallback_func()


# ============================================
# Timeout Handling
# ============================================

def with_timeout(timeout_seconds: float):
    """
    Decorator to add timeout to function.

    Note: This is a simplified version. For production, use
    multiprocessing or async with proper timeout handling.

    Args:
        timeout_seconds: Timeout in seconds

    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds}s")

            # Set signal handler
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds))

            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

            return result

        return wrapper
    return decorator


# ============================================
# Rate Limiting
# ============================================

class RateLimiter:
    """
    Simple rate limiter.

    Usage:
        limiter = RateLimiter(max_calls=10, period=60)
        if limiter.allow():
            make_api_call()
    """

    def __init__(self, max_calls: int, period: float):
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum calls allowed in period
            period: Time period in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self.calls: list[float] = []

    def allow(self) -> bool:
        """Check if a call is allowed."""
        now = time.time()

        # Remove old calls
        self.calls = [t for t in self.calls if now - t < self.period]

        if len(self.calls) >= self.max_calls:
            return False

        self.calls.append(now)
        return True

    def wait_time(self) -> float:
        """Get time to wait until next call is allowed."""
        now = time.time()
        self.calls = [t for t in self.calls if now - t < self.period]

        if len(self.calls) < self.max_calls:
            return 0.0

        # Wait until oldest call expires
        oldest = min(self.calls)
        return max(0, self.period - (now - oldest))


# ============================================
# Safe Execution
# ============================================

def safe_execute(
    func: Callable[..., T],
    default: Any = None,
    log_errors: bool = True,
) -> Any:
    """
    Safely execute a function, returning default on error.

    Args:
        func: Function to execute
        default: Default return value on error
        log_errors: Whether to log errors

    Returns:
        Function result or default
    """
    try:
        return func()
    except Exception as e:
        if log_errors:
            logger.error(f"Safe execute caught error: {e}")
        return default


# ============================================
# Error Response Formatting
# ============================================

def format_error_response(error: Exception) -> dict:
    """
    Format error for API response.

    Args:
        error: Exception to format

    Returns:
        Error dictionary
    """
    if isinstance(error, ValidationError):
        return {
            "error": "validation_error",
            "message": str(error),
            "field": error.field,
        }
    elif isinstance(error, RateLimitError):
        return {
            "error": "rate_limit",
            "message": str(error),
            "retry_after": error.retry_after,
        }
    elif isinstance(error, CelltypeAgentError):
        return {
            "error": "application_error",
            "message": str(error),
        }
    else:
        return {
            "error": "internal_error",
            "message": "An unexpected error occurred",
        }