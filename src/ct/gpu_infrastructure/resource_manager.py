"""
GPU Resource Manager for CellType-Agent.

Manages GPU resources including:
- Detection and monitoring
- Memory allocation
- Job scheduling
- Multi-GPU coordination
"""

import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger("ct.gpu_infrastructure")


class GPUStatus(Enum):
    """GPU availability status."""
    AVAILABLE = "available"
    BUSY = "busy"
    ERROR = "error"
    NOT_FOUND = "not_found"


@dataclass
class GPUInfo:
    """Information about a GPU device."""
    index: int
    name: str
    vram_total_mb: int
    vram_used_mb: int
    vram_free_mb: int
    utilization_pct: float
    temperature_c: float
    status: GPUStatus = GPUStatus.AVAILABLE

    @property
    def vram_used_pct(self) -> float:
        return (self.vram_used_mb / self.vram_total_mb * 100) if self.vram_total_mb > 0 else 0


@dataclass
class GPUReservation:
    """A GPU reservation."""
    gpu_index: int
    reserved_mb: int
    job_id: str
    created_at: float
    expires_at: Optional[float] = None


class GPUResourceManager:
    """
    Manages GPU resources for CellType-Agent.

    Features:
    - GPU detection and monitoring via nvidia-smi
    - Memory reservation and allocation
    - Job queue management
    - Multi-GPU coordination

    Usage:
        manager = GPUResourceManager()
        if manager.has_available_gpu(vram_needed_mb=20000):
            gpu = manager.reserve_gpu(vram_mb=20000, job_id="boltz2-predict")
            # ... use GPU ...
            manager.release_reservation(gpu)
    """

    def __init__(
        self,
        min_vram_gb: int = 20,
        reservation_timeout_s: int = 3600,
    ):
        """
        Initialize GPU resource manager.

        Args:
            min_vram_gb: Minimum VRAM to consider a GPU usable
            reservation_timeout_s: Auto-expire reservations after this time
        """
        self.min_vram_gb = min_vram_gb
        self.reservation_timeout_s = reservation_timeout_s

        self._gpus: dict[int, GPUInfo] = {}
        self._reservations: dict[int, GPUReservation] = {}
        self._last_update: float = 0

        # Initial detection
        self.detect_gpus()

    def detect_gpus(self) -> list[GPUInfo]:
        """
        Detect available GPUs using nvidia-smi.

        Returns:
            List of detected GPU info
        """
        self._gpus = {}

        try:
            # Query GPU info
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                logger.warning(f"nvidia-smi failed: {result.stderr}")
                return []

            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 7:
                    index = int(parts[0])
                    name = parts[1]
                    vram_total = int(parts[2])
                    vram_used = int(parts[3])
                    vram_free = int(parts[4])
                    utilization = float(parts[5])
                    temperature = float(parts[6])

                    status = GPUStatus.AVAILABLE
                    if vram_free < self.min_vram_gb * 1024:
                        status = GPUStatus.BUSY

                    self._gpus[index] = GPUInfo(
                        index=index,
                        name=name,
                        vram_total_mb=vram_total,
                        vram_used_mb=vram_used,
                        vram_free_mb=vram_free,
                        utilization_pct=utilization,
                        temperature_c=temperature,
                        status=status,
                    )

            self._last_update = time.time()
            logger.info(f"Detected {len(self._gpus)} GPUs")

        except FileNotFoundError:
            logger.warning("nvidia-smi not found - no GPU available")
        except Exception as e:
            logger.error(f"GPU detection error: {e}")

        return list(self._gpus.values())

    def refresh_status(self) -> None:
        """Refresh GPU status from nvidia-smi."""
        # Only refresh if more than 5 seconds since last update
        if time.time() - self._last_update < 5:
            return

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,memory.free,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                return

            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    index = int(parts[0])
                    if index in self._gpus:
                        gpu = self._gpus[index]
                        gpu.vram_used_mb = int(parts[1])
                        gpu.vram_free_mb = int(parts[2])
                        gpu.utilization_pct = float(parts[3])
                        gpu.temperature_c = float(parts[4])

                        # Update status based on reservations
                        if index in self._reservations:
                            gpu.status = GPUStatus.BUSY
                        elif gpu.vram_free_mb < self.min_vram_gb * 1024:
                            gpu.status = GPUStatus.BUSY
                        else:
                            gpu.status = GPUStatus.AVAILABLE

            self._last_update = time.time()

        except Exception as e:
            logger.error(f"GPU refresh error: {e}")

    def get_gpu(self, index: int) -> Optional[GPUInfo]:
        """Get info for a specific GPU."""
        self.refresh_status()
        return self._gpus.get(index)

    def get_available_gpus(self, min_vram_mb: int = 0) -> list[GPUInfo]:
        """
        Get list of available GPUs.

        Args:
            min_vram_mb: Minimum free VRAM required

        Returns:
            List of available GPUs
        """
        self.refresh_status()

        available = []
        for gpu in self._gpus.values():
            if gpu.status == GPUStatus.AVAILABLE and gpu.vram_free_mb >= min_vram_mb:
                available.append(gpu)

        # Sort by most free VRAM
        available.sort(key=lambda g: g.vram_free_mb, reverse=True)
        return available

    def has_available_gpu(self, vram_needed_mb: int = 0) -> bool:
        """Check if any GPU has enough VRAM."""
        return len(self.get_available_gpus(vram_needed_mb)) > 0

    def reserve_gpu(
        self,
        vram_mb: int,
        job_id: str,
        gpu_index: Optional[int] = None,
        timeout_s: Optional[int] = None,
    ) -> Optional[int]:
        """
        Reserve a GPU for a job.

        Args:
            vram_mb: VRAM to reserve
            job_id: Unique job identifier
            gpu_index: Specific GPU index (None for auto-select)
            timeout_s: Reservation timeout

        Returns:
            GPU index if reserved, None if unavailable
        """
        self.refresh_status()

        # Clean expired reservations
        self._clean_expired_reservations()

        # Auto-select best GPU
        if gpu_index is None:
            available = self.get_available_gpus(vram_mb)
            if not available:
                return None
            gpu_index = available[0].index

        # Check if specific GPU is available
        if gpu_index in self._reservations:
            return None

        gpu = self._gpus.get(gpu_index)
        if not gpu or gpu.vram_free_mb < vram_mb:
            return None

        # Create reservation
        timeout = timeout_s or self.reservation_timeout_s
        reservation = GPUReservation(
            gpu_index=gpu_index,
            reserved_mb=vram_mb,
            job_id=job_id,
            created_at=time.time(),
            expires_at=time.time() + timeout if timeout else None,
        )

        self._reservations[gpu_index] = reservation
        gpu.status = GPUStatus.BUSY

        logger.info(f"Reserved GPU {gpu_index} ({vram_mb}MB) for job {job_id}")
        return gpu_index

    def release_reservation(self, gpu_index: int) -> bool:
        """
        Release a GPU reservation.

        Args:
            gpu_index: GPU index to release

        Returns:
            True if reservation was released
        """
        if gpu_index not in self._reservations:
            return False

        reservation = self._reservations.pop(gpu_index)
        logger.info(f"Released GPU {gpu_index} from job {reservation.job_id}")

        if gpu_index in self._gpus:
            self._gpus[gpu_index].status = GPUStatus.AVAILABLE

        return True

    def release_job(self, job_id: str) -> int:
        """
        Release all reservations for a job.

        Args:
            job_id: Job identifier

        Returns:
            Number of GPUs released
        """
        released = 0
        for gpu_index, reservation in list(self._reservations.items()):
            if reservation.job_id == job_id:
                self.release_reservation(gpu_index)
                released += 1
        return released

    def _clean_expired_reservations(self) -> None:
        """Clean up expired reservations."""
        now = time.time()
        for gpu_index, reservation in list(self._reservations.items()):
            if reservation.expires_at and reservation.expires_at < now:
                logger.warning(f"Reservation expired on GPU {gpu_index}")
                self.release_reservation(gpu_index)

    def get_reservation(self, gpu_index: int) -> Optional[GPUReservation]:
        """Get reservation for a GPU."""
        return self._reservations.get(gpu_index)

    def get_summary(self) -> dict:
        """Get GPU resource summary."""
        self.refresh_status()

        total_vram = sum(g.vram_total_mb for g in self._gpus.values())
        free_vram = sum(g.vram_free_mb for g in self._gpus.values())
        reserved_vram = sum(r.reserved_mb for r in self._reservations.values())

        return {
            "gpu_count": len(self._gpus),
            "available_count": len(self.get_available_gpus()),
            "total_vram_gb": total_vram / 1024,
            "free_vram_gb": free_vram / 1024,
            "reserved_vram_gb": reserved_vram / 1024,
            "gpus": [
                {
                    "index": g.index,
                    "name": g.name,
                    "vram_free_gb": g.vram_free_mb / 1024,
                    "status": g.status.value,
                }
                for g in self._gpus.values()
            ],
        }

    def estimate_vram_for_boltz2(
        self,
        protein_length: int,
        has_ligand: bool = False,
    ) -> int:
        """
        Estimate VRAM needed for Boltz-2 prediction.

        Args:
            protein_length: Number of amino acids
            has_ligand: Whether a ligand is present

        Returns:
            Estimated VRAM in MB
        """
        # Base VRAM for model
        base_vram_mb = 4000

        # VRAM scales roughly with sequence length
        # ~100MB per 100 residues
        protein_vram = protein_length * 1.0

        # Ligand adds ~500MB
        ligand_vram = 500 if has_ligand else 0

        # Add buffer
        total_mb = int((base_vram_mb + protein_vram + ligand_vram) * 1.2)

        return max(8000, total_mb)  # Minimum 8GB

    def estimate_batch_size(
        self,
        protein_length: int,
        vram_available_mb: int,
    ) -> int:
        """
        Estimate maximum batch size for virtual screening.

        Args:
            protein_length: Protein sequence length
            vram_available_mb: Available VRAM in MB

        Returns:
            Recommended batch size
        """
        # Single prediction VRAM
        single_vram = self.estimate_vram_for_boltz2(protein_length)

        # Estimate batch size
        batch_size = max(1, int(vram_available_mb / single_vram * 0.8))

        # Cap at reasonable limits
        return min(batch_size, 64)

    def check_requirements(self, min_vram_gb: int = 20) -> dict:
        """
        Check if system meets GPU requirements.

        Args:
            min_vram_gb: Minimum required VRAM

        Returns:
            Dictionary with check results
        """
        self.detect_gpus()

        issues = []
        warnings = []

        if not self._gpus:
            issues.append("No GPU detected")
        else:
            for gpu in self._gpus.values():
                if gpu.vram_total_mb < min_vram_gb * 1024:
                    issues.append(
                        f"GPU {gpu.index} ({gpu.name}) has insufficient VRAM: "
                        f"{gpu.vram_total_mb / 1024:.1f}GB < {min_vram_gb}GB required"
                    )

        return {
            "compatible": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "gpus": [
                {
                    "index": g.index,
                    "name": g.name,
                    "vram_gb": g.vram_total_mb / 1024,
                }
                for g in self._gpus.values()
            ],
        }