"""
Performance Testing Utilities for CellType-Agent.

Provides load testing and benchmarking tools.
"""

import asyncio
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("ct.testing.performance")


@dataclass
class PerformanceResult:
    """Result of a performance test."""
    name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time_s: float
    min_latency_ms: float
    max_latency_ms: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    requests_per_second: float
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_time_s": self.total_time_s,
            "latency_ms": {
                "min": self.min_latency_ms,
                "max": self.max_latency_ms,
                "avg": self.avg_latency_ms,
                "p50": self.p50_latency_ms,
                "p95": self.p95_latency_ms,
                "p99": self.p99_latency_ms,
            },
            "requests_per_second": self.requests_per_second,
            "error_rate": self.failed_requests / self.total_requests if self.total_requests > 0 else 0,
        }


class LoadTester:
    """
    Load testing framework for API endpoints.

    Features:
    - Concurrent requests
    - Ramp-up support
    - Statistics collection
    - Error tracking

    Usage:
        tester = LoadTester()
        result = await tester.run("http://localhost:8000/api/v1/query", method="POST", json={"query": "test"})
        print(result.requests_per_second)
    """

    def __init__(self, max_concurrent: int = 10):
        """
        Initialize load tester.

        Args:
            max_concurrent: Maximum concurrent requests
        """
        self.max_concurrent = max_concurrent

    async def run(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[dict] = None,
        json_data: Optional[dict] = None,
        num_requests: int = 100,
        ramp_up_s: float = 0.0,
    ) -> PerformanceResult:
        """
        Run load test.

        Args:
            url: Target URL
            method: HTTP method
            headers: Request headers
            json_data: JSON body
            num_requests: Total requests to make
            ramp_up_s: Ramp-up time in seconds

        Returns:
            Performance result
        """
        import aiohttp

        latencies: list[float] = []
        errors: list[str] = []
        successful = 0
        failed = 0

        start_time = time.time()
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def make_request(session: aiohttp.ClientSession, request_num: int) -> Optional[float]:
            nonlocal successful, failed

            # Ramp up delay
            if ramp_up_s > 0:
                delay = (request_num / num_requests) * ramp_up_s
                await asyncio.sleep(delay)

            async with semaphore:
                request_start = time.time()

                try:
                    async with session.request(
                        method,
                        url,
                        headers=headers,
                        json=json_data,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as response:
                        latency = (time.time() - request_start) * 1000

                        if response.status < 400:
                            successful += 1
                            return latency
                        else:
                            failed += 1
                            errors.append(f"HTTP {response.status}")
                            return None

                except Exception as e:
                    failed += 1
                    errors.append(str(e))
                    return None

        async with aiohttp.ClientSession() as session:
            tasks = [
                make_request(session, i)
                for i in range(num_requests)
            ]

            results = await asyncio.gather(*tasks)

        total_time = time.time() - start_time
        latencies = [r for r in results if r is not None]

        return self._calculate_result(
            name=f"load_test_{url}",
            total_requests=num_requests,
            successful=successful,
            failed=failed,
            latencies=latencies,
            total_time=total_time,
            errors=errors[:10],  # Limit errors
        )

    def _calculate_result(
        self,
        name: str,
        total_requests: int,
        successful: int,
        failed: int,
        latencies: list[float],
        total_time: float,
        errors: list[str],
    ) -> PerformanceResult:
        """Calculate performance result from raw data."""
        if latencies:
            sorted_latencies = sorted(latencies)
            count = len(sorted_latencies)

            return PerformanceResult(
                name=name,
                total_requests=total_requests,
                successful_requests=successful,
                failed_requests=failed,
                total_time_s=total_time,
                min_latency_ms=sorted_latencies[0],
                max_latency_ms=sorted_latencies[-1],
                avg_latency_ms=statistics.mean(sorted_latencies),
                p50_latency_ms=sorted_latencies[int(count * 0.5)],
                p95_latency_ms=sorted_latencies[int(count * 0.95)],
                p99_latency_ms=sorted_latencies[int(count * 0.99)],
                requests_per_second=successful / total_time if total_time > 0 else 0,
                errors=errors,
            )
        else:
            return PerformanceResult(
                name=name,
                total_requests=total_requests,
                successful_requests=successful,
                failed_requests=failed,
                total_time_s=total_time,
                min_latency_ms=0,
                max_latency_ms=0,
                avg_latency_ms=0,
                p50_latency_ms=0,
                p95_latency_ms=0,
                p99_latency_ms=0,
                requests_per_second=0,
                errors=errors,
            )


class BenchmarkSuite:
    """
    Suite of performance benchmarks.

    Usage:
        suite = BenchmarkSuite()
        suite.add_benchmark("query", lambda: run_query("test"))
        results = suite.run_all()
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize benchmark suite.

        Args:
            output_dir: Directory to save results
        """
        self.output_dir = output_dir or Path.home() / ".ct" / "benchmarks"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._benchmarks: dict[str, Callable] = {}

    def add_benchmark(self, name: str, func: Callable, iterations: int = 100):
        """
        Add a benchmark.

        Args:
            name: Benchmark name
            func: Function to benchmark
            iterations: Number of iterations
        """
        self._benchmarks[name] = (func, iterations)

    def run_benchmark(self, name: str) -> PerformanceResult:
        """Run a single benchmark."""
        func, iterations = self._benchmarks[name]

        latencies = []
        successful = 0
        failed = 0
        errors = []

        start_time = time.time()

        for _ in range(iterations):
            try:
                iter_start = time.time()
                func()
                latency = (time.time() - iter_start) * 1000
                latencies.append(latency)
                successful += 1
            except Exception as e:
                failed += 1
                errors.append(str(e))

        total_time = time.time() - start_time

        return self._calculate_result(
            name=name,
            total_requests=iterations,
            successful=successful,
            failed=failed,
            latencies=latencies,
            total_time=total_time,
            errors=errors[:10],
        )

    def run_all(self) -> dict[str, PerformanceResult]:
        """Run all benchmarks."""
        results = {}

        for name in self._benchmarks:
            logger.info(f"Running benchmark: {name}")
            results[name] = self.run_benchmark(name)

        return results

    def save_results(self, results: dict[str, PerformanceResult], filename: str):
        """Save results to file."""
        output_path = self.output_dir / filename

        data = {name: r.to_dict() for name, r in results.items()}

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved benchmark results to {output_path}")

    def _calculate_result(self, *args, **kwargs) -> PerformanceResult:
        """Calculate performance result."""
        # Same logic as LoadTester
        name = kwargs.get("name") or args[0]
        total_requests = kwargs.get("total_requests") or args[1]
        successful = kwargs.get("successful") or args[2]
        failed = kwargs.get("failed") or args[3]
        latencies = kwargs.get("latencies") or args[4]
        total_time = kwargs.get("total_time") or args[5]
        errors = kwargs.get("errors") or args[6]

        if latencies:
            import statistics
            sorted_latencies = sorted(latencies)
            count = len(sorted_latencies)

            return PerformanceResult(
                name=name,
                total_requests=total_requests,
                successful_requests=successful,
                failed_requests=failed,
                total_time_s=total_time,
                min_latency_ms=sorted_latencies[0],
                max_latency_ms=sorted_latencies[-1],
                avg_latency_ms=statistics.mean(sorted_latencies),
                p50_latency_ms=sorted_latencies[int(count * 0.5)],
                p95_latency_ms=sorted_latencies[int(count * 0.95)],
                p99_latency_ms=sorted_latencies[int(count * 0.99)],
                requests_per_second=successful / total_time if total_time > 0 else 0,
                errors=errors,
            )
        else:
            return PerformanceResult(
                name=name,
                total_requests=total_requests,
                successful_requests=successful,
                failed_requests=failed,
                total_time_s=total_time,
                min_latency_ms=0,
                max_latency_ms=0,
                avg_latency_ms=0,
                p50_latency_ms=0,
                p95_latency_ms=0,
                p99_latency_ms=0,
                requests_per_second=0,
                errors=errors,
            )


def measure_time(func: Callable) -> Callable:
    """Decorator to measure function execution time."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        latency = (time.time() - start) * 1000
        logger.debug(f"{func.__name__} took {latency:.2f}ms")
        return result
    return wrapper


async def measure_async_time(func: Callable) -> Callable:
    """Decorator to measure async function execution time."""
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        latency = (time.time() - start) * 1000
        logger.debug(f"{func.__name__} took {latency:.2f}ms")
        return result
    return wrapper