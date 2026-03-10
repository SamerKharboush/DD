"""
Local LLM Client for offline inference.

Supports:
- vLLM server for fast inference
- llama.cpp for CPU inference
- Quantized models (GPTQ, AWQ)
"""

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger("ct.local_llm.client")


@dataclass
class LocalLLMConfig:
    """Configuration for local LLM."""
    model_name: str = "llama-3-70b-instruct"
    model_path: Optional[str] = None
    host: str = "localhost"
    port: int = 8000
    quantization: Optional[str] = None  # "gptq", "awq", "fp8"
    max_tokens: int = 4096
    temperature: float = 0.7
    gpu_memory_utilization: float = 0.9


class LocalLLMClient:
    """
    Client for local LLM inference.

    Usage:
        client = LocalLLMClient()
        response = client.chat("What drugs target KRAS?")
    """

    def __init__(
        self,
        config: Optional[LocalLLMConfig] = None,
        auto_start: bool = False,
    ):
        """
        Initialize local LLM client.

        Args:
            config: LLM configuration
            auto_start: Whether to auto-start the server
        """
        self.config = config or LocalLLMConfig()
        self._server_process = None
        self._is_running = False

        if auto_start:
            self.start_server()

    def start_server(self) -> bool:
        """Start the vLLM server."""
        if self._is_running:
            return True

        logger.info(f"Starting vLLM server for {self.config.model_name}...")

        # Build vLLM command
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.config.model_path or self.config.model_name,
            "--host", self.config.host,
            "--port", str(self.config.port),
            "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
            "--max-model-len", str(self.config.max_tokens),
        ]

        if self.config.quantization:
            cmd.extend(["--quantization", self.config.quantization])

        try:
            self._server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait for server to start
            for _ in range(60):  # 60 second timeout
                try:
                    response = requests.get(
                        f"http://{self.config.host}:{self.config.port}/health"
                    )
                    if response.status_code == 200:
                        self._is_running = True
                        logger.info("vLLM server started successfully")
                        return True
                except requests.exceptions.ConnectionError:
                    pass
                time.sleep(1)

            logger.error("vLLM server failed to start within 60 seconds")
            return False

        except Exception as e:
            logger.error(f"Failed to start vLLM server: {e}")
            return False

    def stop_server(self) -> None:
        """Stop the vLLM server."""
        if self._server_process:
            self._server_process.terminate()
            self._server_process = None
            self._is_running = False
            logger.info("vLLM server stopped")

    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send a chat message to the local LLM.

        Args:
            message: User message
            system_prompt: Optional system prompt
            temperature: Override temperature
            max_tokens: Override max tokens

        Returns:
            Model response
        """
        url = f"http://{self.config.host}:{self.config.port}/v1/chat/completions"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to local LLM server")
            return self._fallback_inference(message, system_prompt)

        except Exception as e:
            logger.error(f"Local LLM error: {e}")
            return f"Error: {e}"

    def _fallback_inference(
        self,
        message: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Fallback to llama.cpp or return error."""
        # Try llama.cpp if available
        try:
            result = subprocess.run(
                ["llama-cli", "-m", self.config.model_path or "", "-p", message],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                return result.stdout
        except Exception:
            pass

        return "Error: Local LLM server not running. Start with client.start_server()"

    def embed(self, text: str) -> list[float]:
        """Get embeddings from local model."""
        url = f"http://{self.config.host}:{self.config.port}/v1/embeddings"

        payload = {
            "model": self.config.model_name,
            "input": text,
        }

        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return []

    def is_available(self) -> bool:
        """Check if local LLM is available."""
        try:
            response = requests.get(
                f"http://{self.config.host}:{self.config.port}/health",
                timeout=5,
            )
            return response.status_code == 200
        except Exception:
            return False

    def get_model_info(self) -> dict:
        """Get model information."""
        try:
            response = requests.get(
                f"http://{self.config.host}:{self.config.port}/v1/models",
                timeout=10,
            )
            response.raise_for_status()
            return response.json()
        except Exception:
            return {"error": "Server not available"}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._server_process:
            self.stop_server()


def load_quantized_model(
    model_name: str,
    quantization: str = "awq",
) -> str:
    """
    Load a quantized model.

    Args:
        model_name: Model name or path
        quantization: Quantization type

    Returns:
        Path to quantized model
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading {model_name} with {quantization} quantization...")

    if quantization == "awq":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        # Apply AWQ quantization
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

        quantize_config = BaseQuantizeConfig(
            bits=4,
            group_size=128,
            desc_act=False,
        )

        model = AutoGPTQForCausalLM.from_quantized(
            model_name,
            quantize_config=quantize_config,
        )

    return model_name


class QuantizedModelManager:
    """Manager for quantized models."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".ct" / "models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_model(
        self,
        model_name: str,
        quantization: Optional[str] = None,
    ) -> Path:
        """Download a model to cache."""
        from huggingface_hub import snapshot_download

        model_dir = self.cache_dir / model_name.replace("/", "_")

        if not model_dir.exists():
            logger.info(f"Downloading {model_name}...")
            snapshot_download(
                repo_id=model_name,
                local_dir=model_dir,
            )

        return model_dir

    def list_cached_models(self) -> list[str]:
        """List cached models."""
        return [d.name for d in self.cache_dir.iterdir() if d.is_dir()]

    def get_model_size(self, model_name: str) -> int:
        """Get model size in bytes."""
        model_dir = self.cache_dir / model_name.replace("/", "_")
        if not model_dir.exists():
            return 0

        return sum(f.stat().st_size for f in model_dir.rglob("*"))