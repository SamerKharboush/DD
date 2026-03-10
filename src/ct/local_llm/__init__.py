"""
Local LLM Module for CellType-Agent Phase 5.

Implements local model deployment:
- vLLM server integration
- LoRA fine-tuning
- Model quantization
- Hybrid routing (local + cloud)
"""

from ct.local_llm.local_client import LocalLLMClient
from ct.local_llm.lora_trainer import LoRATrainer
from ct.local_llm.hybrid_router import HybridRouter

__all__ = [
    "LocalLLMClient",
    "LoRATrainer",
    "HybridRouter",
]