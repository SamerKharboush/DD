"""
Batch Processor for GPU-accelerated inference.

Provides efficient batch processing for:
- Virtual screening (multiple ligands against one protein)
- Parallel structure predictions
- GPU memory optimization
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ct.gpu_infrastructure.resource_manager import GPUResourceManager

logger = logging.getLogger("ct.gpu_infrastructure")


@dataclass
class BatchJob:
    """A batch processing job."""
    job_id: str
    items: list[Any]
    processor: Callable
    batch_size: int = 1
    status: str = "pending"
    results: list[Any] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    progress_pct: float = 0.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    gpu_index: Optional[int] = None


@dataclass
class BatchResult:
    """Result of batch processing."""
    job_id: str
    total_items: int
    successful: int
    failed: int
    results: list[Any]
    errors: list[dict]
    duration_seconds: float
    throughput_per_second: float


class BatchProcessor:
    """
    Efficient batch processor for GPU inference.

    Features:
    - Automatic batching with memory optimization
    - Progress tracking
    - Error handling with retry
    - Multi-GPU support

    Usage:
        processor = BatchProcessor()
        job = processor.submit(
            job_id="virtual-screen",
            items=ligand_smiles_list,
            processor=predict_affinity,
            batch_size=32,
        )
        result = processor.wait_for_completion(job)
    """

    def __init__(
        self,
        gpu_manager: Optional[GPUResourceManager] = None,
        max_workers: int = 4,
        default_batch_size: int = 16,
    ):
        """
        Initialize batch processor.

        Args:
            gpu_manager: GPU resource manager
            max_workers: Maximum concurrent workers
            default_batch_size: Default batch size
        """
        self.gpu_manager = gpu_manager or GPUResourceManager()
        self.max_workers = max_workers
        self.default_batch_size = default_batch_size

        self._jobs: dict[str, BatchJob] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def submit(
        self,
        job_id: str,
        items: list[Any],
        processor: Callable,
        batch_size: Optional[int] = None,
        gpu_index: Optional[int] = None,
        priority: int = 0,
    ) -> BatchJob:
        """
        Submit a batch processing job.

        Args:
            job_id: Unique job identifier
            items: List of items to process
            processor: Processing function
            batch_size: Batch size (auto-determined if None)
            gpu_index: Specific GPU to use
            priority: Job priority (higher = more important)

        Returns:
            BatchJob instance
        """
        # Determine batch size
        if batch_size is None:
            batch_size = self._determine_batch_size(items, processor)

        job = BatchJob(
            job_id=job_id,
            items=items,
            processor=processor,
            batch_size=batch_size,
            gpu_index=gpu_index,
        )

        self._jobs[job_id] = job
        logger.info(
            f"Submitted batch job {job_id}: {len(items)} items, "
            f"batch_size={batch_size}"
        )

        return job

    def start(self, job_id: str) -> bool:
        """
        Start processing a job.

        Args:
            job_id: Job identifier

        Returns:
            True if job started
        """
        job = self._jobs.get(job_id)
        if not job or job.status != "pending":
            return False

        # Reserve GPU if needed
        if job.gpu_index is None:
            # Estimate VRAM needed
            vram_needed = getattr(job.processor, "min_vram_mb", 10000)
            job.gpu_index = self.gpu_manager.reserve_gpu(
                vram_mb=vram_needed,
                job_id=job_id,
            )
            if job.gpu_index is None:
                job.status = "failed"
                job.errors.append({"error": "No GPU available"})
                return False

        job.status = "running"
        job.start_time = time.time()

        # Submit to executor
        self._executor.submit(self._process_job, job)

        return True

    def _process_job(self, job: BatchJob) -> None:
        """Process a batch job."""
        try:
            total = len(job.items)
            processed = 0

            # Process in batches
            for i in range(0, total, job.batch_size):
                batch = job.items[i:i + job.batch_size]

                try:
                    # Call processor with batch
                    batch_results = job.processor(
                        batch,
                        gpu_index=job.gpu_index,
                    )

                    # Ensure batch_results is a list
                    if not isinstance(batch_results, list):
                        batch_results = [batch_results]

                    job.results.extend(batch_results)
                    processed += len(batch)

                except Exception as e:
                    logger.error(f"Batch {i//job.batch_size} failed: {e}")
                    job.errors.append({
                        "batch_start": i,
                        "batch_size": len(batch),
                        "error": str(e),
                    })
                    # Add None results for failed batch
                    job.results.extend([None] * len(batch))
                    processed += len(batch)

                # Update progress
                job.progress_pct = processed / total * 100

            job.status = "completed"
            job.end_time = time.time()

        except Exception as e:
            logger.error(f"Job {job.job_id} failed: {e}")
            job.status = "failed"
            job.errors.append({"error": str(e)})
            job.end_time = time.time()

        finally:
            # Release GPU
            if job.gpu_index is not None:
                self.gpu_manager.release_reservation(job.gpu_index)

    def wait_for_completion(
        self,
        job: BatchJob,
        timeout_s: Optional[float] = None,
    ) -> BatchResult:
        """
        Wait for job completion.

        Args:
            job: Batch job
            timeout_s: Timeout in seconds

        Returns:
            BatchResult
        """
        start = time.time()

        while job.status in ("pending", "running"):
            if timeout_s and (time.time() - start) > timeout_s:
                job.status = "timeout"
                break
            time.sleep(0.1)

        duration = (job.end_time or time.time()) - (job.start_time or start)
        total = len(job.items)
        successful = sum(1 for r in job.results if r is not None)

        return BatchResult(
            job_id=job.job_id,
            total_items=total,
            successful=successful,
            failed=total - successful,
            results=job.results,
            errors=job.errors,
            duration_seconds=duration,
            throughput_per_second=successful / duration if duration > 0 else 0,
        )

    def get_job(self, job_id: str) -> Optional[BatchJob]:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def get_job_status(self, job_id: str) -> dict:
        """Get job status."""
        job = self._jobs.get(job_id)
        if not job:
            return {"status": "not_found"}

        return {
            "job_id": job.job_id,
            "status": job.status,
            "total_items": len(job.items),
            "processed_items": len(job.results),
            "progress_pct": job.progress_pct,
            "errors_count": len(job.errors),
        }

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        job = self._jobs.get(job_id)
        if not job or job.status not in ("pending", "running"):
            return False

        job.status = "cancelled"

        if job.gpu_index is not None:
            self.gpu_manager.release_reservation(job.gpu_index)

        return True

    def _determine_batch_size(self, items: list, processor: Callable) -> int:
        """Determine optimal batch size."""
        # Get recommended batch size from processor if available
        if hasattr(processor, "recommended_batch_size"):
            return processor.recommended_batch_size

        # Estimate based on GPU memory
        if self.gpu_manager._gpus:
            gpu = list(self.gpu_manager._gpus.values())[0]
            item_size_mb = getattr(processor, "item_memory_mb", 500)
            batch_size = max(1, int(gpu.vram_free_mb * 0.7 / item_size_mb))
            return min(batch_size, self.default_batch_size)

        return self.default_batch_size

    def estimate_duration(
        self,
        num_items: int,
        processor: Callable,
        batch_size: Optional[int] = None,
    ) -> float:
        """
        Estimate processing duration.

        Args:
            num_items: Number of items to process
            processor: Processing function
            batch_size: Batch size

        Returns:
            Estimated duration in seconds
        """
        batch_size = batch_size or self.default_batch_size

        # Get time per item from processor if available
        time_per_item = getattr(processor, "time_per_item", 0.5)

        # Batch processing is faster per item
        batch_speedup = min(batch_size, 8)  # Up to 8x speedup

        total_time = (num_items / batch_speedup) * time_per_item

        return total_time

    def optimize_for_throughput(
        self,
        items: list[Any],
        processor: Callable,
    ) -> dict:
        """
        Analyze and recommend optimal processing parameters.

        Args:
            items: Items to process
            processor: Processing function

        Returns:
            Optimization recommendations
        """
        num_items = len(items)

        # Get GPU info
        gpu_summary = self.gpu_manager.get_summary()
        available_gpus = gpu_summary["available_count"]

        # Calculate recommendations
        batch_size = self._determine_batch_size(items, processor)
        estimated_duration = self.estimate_duration(num_items, processor, batch_size)

        recommendations = {
            "num_items": num_items,
            "recommended_batch_size": batch_size,
            "estimated_duration_s": estimated_duration,
            "available_gpus": available_gpus,
            "recommended_workers": min(available_gpus, self.max_workers),
        }

        # If multiple GPUs available, recommend splitting
        if available_gpus > 1 and num_items > batch_size * 2:
            items_per_gpu = num_items // available_gpus
            recommendations["multi_gpu_split"] = {
                "gpus": available_gpus,
                "items_per_gpu": items_per_gpu,
                "estimated_speedup": available_gpus * 0.8,  # 80% efficiency
                "parallel_duration_s": estimated_duration / (available_gpus * 0.8),
            }

        return recommendations


async def process_batch_async(
    items: list[Any],
    processor: Callable,
    batch_size: int = 16,
) -> list[Any]:
    """
    Async batch processing utility.

    Args:
        items: Items to process
        processor: Async or sync processing function
        batch_size: Batch size

    Returns:
        List of results
    """
    results = []

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]

        if asyncio.iscoroutinefunction(processor):
            batch_results = await processor(batch)
        else:
            batch_results = await asyncio.to_thread(processor, batch)

        if isinstance(batch_results, list):
            results.extend(batch_results)
        else:
            results.append(batch_results)

    return results