"""
GPU Service Module for CellType-Agent.

Provides GPU-accelerated services:
- Boltz-2 structure prediction
- DiffDock docking
- ESMFold folding
"""

from ct.gpu.boltz2_service import Boltz2Service
from ct.gpu.diffdock_service import DiffDockService
from ct.gpu.resource_manager import GPUResourceManager, get_gpu_manager

__all__ = [
    "Boltz2Service",
    "DiffDockService",
    "GPUResourceManager",
    "get_gpu_manager",
]