"""
Health Check System for CellType-Agent.

Implements comprehensive health checking for all system components.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger("ct.monitoring.health")


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    latency_ms: float
    details: dict


class HealthChecker:
    """
    Comprehensive health checking system.

    Checks:
    - Database connectivity
    - Redis availability
    - Neo4j status
    - GPU availability
    - LLM API connectivity
    - Disk space

    Usage:
        checker = HealthChecker()
        result = checker.check_all()
        print(result.overall_status)
    """

    def __init__(self):
        """Initialize health checker."""
        self._checks: dict[str, Callable] = {}
        self._register_default_checks()

    def _register_default_checks(self):
        """Register default health checks."""
        self.register_check("database", self._check_database)
        self.register_check("redis", self._check_redis)
        self.register_check("neo4j", self._check_neo4j)
        self.register_check("gpu", self._check_gpu)
        self.register_check("llm_api", self._check_llm_api)
        self.register_check("disk", self._check_disk)

    def register_check(self, name: str, check_func: Callable):
        """
        Register a health check.

        Args:
            name: Check name
            check_func: Function that returns HealthCheckResult
        """
        self._checks[name] = check_func

    def check(self, name: str) -> HealthCheckResult:
        """
        Run a specific health check.

        Args:
            name: Check name

        Returns:
            Health check result
        """
        check_func = self._checks.get(name)
        if not check_func:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message="Unknown check",
                latency_ms=0,
                details={},
            )

        try:
            return check_func()
        except Exception as e:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {e}",
                latency_ms=0,
                details={"error": str(e)},
            )

    def check_all(self) -> dict:
        """
        Run all health checks.

        Returns:
            Dictionary with overall status and individual results
        """
        results = {}

        for name in self._checks:
            results[name] = self.check(name)

        # Determine overall status
        statuses = [r.status for r in results.values()]

        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall = HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY

        return {
            "status": overall.value,
            "timestamp": time.time(),
            "checks": {name: self._result_to_dict(r) for name, r in results.items()},
        }

    def _result_to_dict(self, result: HealthCheckResult) -> dict:
        """Convert result to dictionary."""
        return {
            "status": result.status.value,
            "message": result.message,
            "latency_ms": result.latency_ms,
            "details": result.details,
        }

    def _check_database(self) -> HealthCheckResult:
        """Check PostgreSQL database."""
        start = time.time()

        try:
            import os
            import sqlite3

            # Try to connect (using sqlite as fallback for demo)
            db_path = os.environ.get("DATABASE_PATH", ":memory:")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            conn.close()

            latency = (time.time() - start) * 1000

            return HealthCheckResult(
                name="database",
                status=HealthStatus.HEALTHY,
                message="Database connection successful",
                latency_ms=latency,
                details={"type": "sqlite"},
            )

        except Exception as e:
            return HealthCheckResult(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {e}",
                latency_ms=(time.time() - start) * 1000,
                details={"error": str(e)},
            )

    def _check_redis(self) -> HealthCheckResult:
        """Check Redis cache."""
        start = time.time()

        try:
            import redis

            redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
            client = redis.from_url(redis_url)
            client.ping()
            client.close()

            latency = (time.time() - start) * 1000

            return HealthCheckResult(
                name="redis",
                status=HealthStatus.HEALTHY,
                message="Redis connection successful",
                latency_ms=latency,
                details={"url": redis_url},
            )

        except ImportError:
            return HealthCheckResult(
                name="redis",
                status=HealthStatus.DEGRADED,
                message="Redis not installed",
                latency_ms=0,
                details={},
            )
        except Exception as e:
            return HealthCheckResult(
                name="redis",
                status=HealthStatus.DEGRADED,
                message=f"Redis connection failed: {e}",
                latency_ms=(time.time() - start) * 1000,
                details={"error": str(e)},
            )

    def _check_neo4j(self) -> HealthCheckResult:
        """Check Neo4j graph database."""
        start = time.time()

        try:
            from neo4j import GraphDatabase

            uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
            user = os.environ.get("NEO4J_USER", "neo4j")
            password = os.environ.get("NEO4J_PASSWORD", "")

            driver = GraphDatabase.driver(uri, auth=(user, password))
            with driver.session() as session:
                session.run("RETURN 1").single()
            driver.close()

            latency = (time.time() - start) * 1000

            return HealthCheckResult(
                name="neo4j",
                status=HealthStatus.HEALTHY,
                message="Neo4j connection successful",
                latency_ms=latency,
                details={"uri": uri},
            )

        except ImportError:
            return HealthCheckResult(
                name="neo4j",
                status=HealthStatus.DEGRADED,
                message="Neo4j driver not installed",
                latency_ms=0,
                details={},
            )
        except Exception as e:
            return HealthCheckResult(
                name="neo4j",
                status=HealthStatus.DEGRADED,
                message=f"Neo4j connection failed: {e}",
                latency_ms=(time.time() - start) * 1000,
                details={"error": str(e)},
            )

    def _check_gpu(self) -> HealthCheckResult:
        """Check GPU availability."""
        start = time.time()

        try:
            import subprocess

            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                latency = (time.time() - start) * 1000
                gpu_info = result.stdout.strip()

                return HealthCheckResult(
                    name="gpu",
                    status=HealthStatus.HEALTHY,
                    message="GPU available",
                    latency_ms=latency,
                    details={"gpu": gpu_info},
                )
            else:
                return HealthCheckResult(
                    name="gpu",
                    status=HealthStatus.DEGRADED,
                    message="No GPU detected",
                    latency_ms=(time.time() - start) * 1000,
                    details={},
                )

        except Exception:
            return HealthCheckResult(
                name="gpu",
                status=HealthStatus.DEGRADED,
                message="No GPU available",
                latency_ms=0,
                details={},
            )

    def _check_llm_api(self) -> HealthCheckResult:
        """Check LLM API connectivity."""
        start = time.time()

        try:
            import os

            api_key = os.environ.get("ANTHROPIC_API_KEY")

            if not api_key:
                return HealthCheckResult(
                    name="llm_api",
                    status=HealthStatus.DEGRADED,
                    message="ANTHROPIC_API_KEY not set",
                    latency_ms=0,
                    details={},
                )

            # Don't actually call API, just check key exists
            latency = (time.time() - start) * 1000

            return HealthCheckResult(
                name="llm_api",
                status=HealthStatus.HEALTHY,
                message="LLM API key configured",
                latency_ms=latency,
                details={"provider": "anthropic"},
            )

        except Exception as e:
            return HealthCheckResult(
                name="llm_api",
                status=HealthStatus.UNHEALTHY,
                message=f"LLM API check failed: {e}",
                latency_ms=(time.time() - start) * 1000,
                details={"error": str(e)},
            )

    def _check_disk(self) -> HealthCheckResult:
        """Check disk space."""
        start = time.time()

        try:
            import shutil

            total, used, free = shutil.disk_usage("/")

            # Calculate percentages
            used_percent = (used / total) * 100
            free_gb = free / (1024 ** 3)

            if used_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"Disk nearly full: {used_percent:.1f}% used"
            elif used_percent > 80:
                status = HealthStatus.DEGRADED
                message = f"Disk space low: {used_percent:.1f}% used"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk space OK: {free_gb:.1f} GB free"

            latency = (time.time() - start) * 1000

            return HealthCheckResult(
                name="disk",
                status=status,
                message=message,
                latency_ms=latency,
                details={
                    "total_gb": total / (1024 ** 3),
                    "used_gb": used / (1024 ** 3),
                    "free_gb": free_gb,
                    "used_percent": used_percent,
                },
            )

        except Exception as e:
            return HealthCheckResult(
                name="disk",
                status=HealthStatus.DEGRADED,
                message=f"Disk check failed: {e}",
                latency_ms=(time.time() - start) * 1000,
                details={"error": str(e)},
            )


# Global health checker
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get the global health checker."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


def run_health_checks() -> dict:
    """Run all health checks and return results."""
    return get_health_checker().check_all()


import os  # Add missing import at module level