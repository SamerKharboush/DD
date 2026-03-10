"""
Protein Validation Module for CellType-Agent Phase 2.

Validates generated proteins using:
- ESM-IF for structure confidence
- Aggrescan3D for aggregation propensity
- NetMHCpan for immunogenicity
- Stability predictors
"""

from ct.validation.protein_validator import ProteinValidator

__all__ = ["ProteinValidator"]