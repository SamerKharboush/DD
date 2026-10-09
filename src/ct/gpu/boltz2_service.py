"""
Boltz-2 Service for Structure Prediction.

Provides interface to Boltz-2 model for:
- Protein structure prediction
- Protein-ligand binding affinity prediction
- Complex structure modeling
"""

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ct.gpu.boltz2")

# Placeholder output is only served when this env var is explicitly set, so a
# demo/test run is opt-in and a normal deployment gets an honest 503 instead of
# invented numbers.
ALLOW_PLACEHOLDER_ENV = "CT_GPU_SERVICES"


def placeholder_allowed() -> bool:
    """True only when the operator has explicitly accepted placeholder output."""
    return os.environ.get(ALLOW_PLACEHOLDER_ENV, "").strip().lower() in {
        "1", "true", "yes", "on",
    }


@dataclass
class StructurePrediction:
    """Result of structure prediction.

    ``is_real_result`` is False for placeholder output. Any consumer that ranks
    or aggregates predictions must check it: a placeholder confidence is a
    constant, not a measurement.
    """

    sequence: str
    pdb_content: str
    confidence: float
    plddt: float
    prediction_time_s: float
    is_real_result: bool = field(default=False)


@dataclass
class AffinityPrediction:
    """Result of affinity prediction. See StructurePrediction.is_real_result."""

    protein_sequence: str
    ligand_smiles: str
    affinity_nm: float
    confidence: float
    delta_g: float
    prediction_time_s: float
    is_real_result: bool = field(default=False)


class Boltz2Service:
    """Boltz-2 structure prediction service.

    .. warning::
       **PLACEHOLDER IMPLEMENTATION — NOT A REAL PREDICTION.**

    The methods below return fixed placeholder values, not model output. They
    deliberately report ``is_real_result = False`` so downstream code can refuse
    to rank against them; earlier revisions returned plausible constants
    (``confidence=0.7``, ``affinity_nm=50.0``, ``plddt=70.0``) with no such
    marker, which made synthetic data indistinguishable from measured data and
    let optimizers rank against noise.

    Serving placeholder numbers over HTTP is worse than serving nothing, because
    the caller cannot tell them apart from a real result. ``is_available()``
    reports whether a *real* backend is present and returns ``False`` by
    default, so the API layer's 503 guard actually fires instead of silently
    answering with constants.

    To make this real, resolve ``_backend``, run inference, and set
    ``is_real_result = True`` only when a genuine prediction was produced.
    """

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        model_size: str = "small",
        gpu_id: int = 0,
    ):
        self.model_dir = Path(model_dir) if model_dir else Path.home() / ".ct" / "models" / "boltz2"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_size = model_size
        self.gpu_id = gpu_id
        self._model = None
        self._loaded = False

    def load_model(self) -> bool:
        """Load the Boltz-2 backend.

        Placeholder build: there is no backend to load, so this returns False
        unless placeholder output was explicitly allowed. It no longer claims
        success by flipping ``_loaded`` unconditionally, which previously made
        the API's 503 guard unreachable.
        """
        if self._loaded:
            return True
        if not placeholder_allowed():
            logger.debug(
                "Boltz-2 backend not present; placeholder output disabled "
                "(set %s=1 to override).", ALLOW_PLACEHOLDER_ENV,
            )
            return False
        self._loaded = True
        return True

    def predict_structure(self, sequence: str) -> StructurePrediction:
        start_time = time.time()
        return StructurePrediction(
            sequence=sequence,
            pdb_content="MOCK_PDB",
            confidence=0.7,
            plddt=70.0,
            prediction_time_s=time.time() - start_time,
            is_real_result=False,
        )

    def predict_affinity(self, protein_sequence: str, ligand_smiles: str) -> AffinityPrediction:
        start_time = time.time()
        return AffinityPrediction(
            protein_sequence=protein_sequence[:50],
            ligand_smiles=ligand_smiles,
            affinity_nm=50.0,
            confidence=0.6,
            delta_g=-5.0,
            prediction_time_s=time.time() - start_time,
            is_real_result=False,
        )

    def is_available(self) -> bool:
        """True only when a real backend exists.

        Returning an unconditional True here is what allowed
        ``POST /api/v1/gpu/predict-structure`` to answer with a constant
        confidence while presenting it as a model result. Placeholder mode is
        honoured only when explicitly enabled.
        """
        return placeholder_allowed()

    def get_status(self) -> dict:
        return {
            "loaded": self._loaded,
            "model_size": self.model_size,
            "backend_available": self.is_available(),
            "placeholder_output": not self.is_available(),
            "is_real_inference": False,
        }