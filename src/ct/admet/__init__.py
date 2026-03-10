"""
ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) prediction module.

Implements GNN-based ADMET prediction with 41 endpoints for drug discovery.
Integrates ADMET-AI and ChemProp models for comprehensive property prediction.
"""

from ct.admet.predictor import ADMETPredictor
from ct.admet.endpoints import ADMET_ENDPOINTS, CRITICAL_ENDPOINTS

__all__ = [
    "ADMETPredictor",
    "ADMET_ENDPOINTS",
    "CRITICAL_ENDPOINTS",
]