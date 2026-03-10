"""
Campaign Module for CellType-Agent Phase 3.

Implements long-running research campaigns:
- DMTA cycle orchestration
- Multi-session research programs
- Progress tracking
"""

from ct.campaign.dmta import DMTACycle, DMTAPhase, run_dmta_cycle

__all__ = [
    "DMTACycle",
    "DMTAPhase",
    "run_dmta_cycle",
]