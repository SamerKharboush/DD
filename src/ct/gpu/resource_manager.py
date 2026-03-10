"""
GPU Resource Manager for CellType-Agent.

Manages GPU allocation, memory, and batch processing.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("ct.gpu.manager")


@dataclass
class GPUInfo:
    """GPU information."""
    id: int
    name: str
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int
    utilization_percent: float


class GPUResourceManager:
    """
    GPU resource management.

    Features:
    - GPU availability checking
    - Memory monitoring
    - Resource allocation
    - Batch scheduling

    Usage:
        manager = GPUResourceManager()
        if manager.is_available():
            info = manager.get_gpu_info()
    """

    def __init__(self):
        self._gpu_available = None
        self._gpu_info: Optional[GPUInfo] = None

    def is_available(self) -> bool:
        """Check if GPU is available."""
        if self._gpu_available is not None:
            return self._gpu_available

        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            self._gpu_available = result.returncode == 0
        except Exception:
            self._gpu_available = False

        return self._gpu_available

    def get_gpu_info(self) -> Optional[GPUInfo]:
        """Get GPU information."""
        if not self.is_available():
            return None

        try:
            import subprocess
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                return GPUInfo(
                    id=int(parts[0]),
                    name=parts[1],
                    memory_total_mb=int(parts[2]),
                    memory_used_mb=int(parts[3]),
                    memory_free_mb=int(parts[4]),
                    utilization_percent=float(parts[5]),
                )
        except Exception as e:
            logger.error(f"Failed to get GPU info: {e}")

        return None

    def get_memory_usage(self) -> dict:
        """Get memory usage statistics."""
        info = self.get_gpu_info()
        if info:
            return {
                "total_mb": info.memory_total_mb,
                "used_mb": info.memory_used_mb,
                "free_mb": info.memory_free_mb,
                "utilization_percent": info.utilization_percent,
            }
        return {"available": False}

    def allocate(self, memory_mb: int) -> bool:
        """Attempt to allocate GPU memory."""
        if not self.is_available():
            return False

        info = self.get_gpu_info()
        if info and info.memory_free_mb >= memory_mb:
            return True

        return False


_gpu_manager: Optional[GPUResourceManager] = None


def get_gpu_manager() -> GPUResourceManager:
    """Get the global GPU manager."""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUResourceManager()
    return _gpu_manager