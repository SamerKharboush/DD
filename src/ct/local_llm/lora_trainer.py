"""
LoRA Fine-Tuning for CellType-Agent.

Implements Low-Rank Adaptation training on:
- BixBench task distribution
- Session traces with feedback
- Domain-specific data
"""

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ct.local_llm.lora")


@dataclass
class LoRAConfig:
    """LoRA training configuration."""
    base_model: str = "meta-llama/Llama-3-70B-Instruct"
    output_dir: str = "ct-lora-adapter"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 4096
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    use_gradient_checkpointing: bool = True
    quantize_base: bool = True  # Use 4-bit quantization


@dataclass
class TrainingData:
    """Training data for LoRA."""
    samples: list[dict] = field(default_factory=list)

    def add_sample(
        self,
        query: str,
        tool_calls: list[dict],
        conclusion: str,
        rating: Optional[int] = None,
    ):
        """Add a training sample."""
        # Format for instruction tuning
        formatted = {
            "instruction": query,
            "input": "",
            "output": self._format_output(tool_calls, conclusion),
            "quality_score": rating / 5.0 if rating else 0.7,
        }
        self.samples.append(formatted)

    def _format_output(self, tool_calls: list[dict], conclusion: str) -> str:
        """Format tool calls and conclusion."""
        parts = []

        for tc in tool_calls:
            tool_name = tc.get("tool", "unknown")
            params = tc.get("parameters", {})
            result = tc.get("result_summary", "")

            parts.append(f"Tool: {tool_name}")
            parts.append(f"Parameters: {json.dumps(params)}")
            parts.append(f"Result: {result}")
            parts.append("")

        parts.append(f"Conclusion: {conclusion}")

        return "\n".join(parts)

    def to_jsonl(self, output_path: Path) -> int:
        """Export to JSONL format."""
        with open(output_path, "w") as f:
            for sample in self.samples:
                f.write(json.dumps(sample) + "\n")
        return len(self.samples)


class LoRATrainer:
    """
    LoRA fine-tuning trainer.

    Usage:
        trainer = LoRATrainer()
        trainer.prepare_data(session_traces)
        trainer.train()
        trainer.merge_and_export()
    """

    def __init__(
        self,
        config: Optional[LoRAConfig] = None,
        data_dir: Optional[Path] = None,
    ):
        """
        Initialize LoRA trainer.

        Args:
            config: Training configuration
            data_dir: Directory for training data
        """
        self.config = config or LoRAConfig()
        self.data_dir = Path(data_dir) if data_dir else Path.home() / ".ct" / "lora_data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.training_data = TrainingData()
        self._is_trained = False

    def prepare_data_from_sessions(
        self,
        min_rating: int = 3,
        min_quality_score: float = 0.5,
        limit: Optional[int] = None,
    ) -> int:
        """
        Prepare training data from logged sessions.

        Args:
            min_rating: Minimum rating to include
            min_quality_score: Minimum quality score
            limit: Maximum samples to prepare

        Returns:
            Number of samples prepared
        """
        from ct.session_logging import SessionLogger

        logger_instance = SessionLogger()
        training_samples = logger_instance.get_training_data(
            min_quality_score=min_quality_score,
            limit=limit or 100000,
        )

        count = 0
        for sample in training_samples:
            rating = sample.get("user_rating")
            if rating and rating >= min_rating:
                self.training_data.add_sample(
                    query=sample["query"],
                    tool_calls=sample.get("tool_calls", []),
                    conclusion=sample.get("conclusion", ""),
                    rating=rating,
                )
                count += 1

        logger.info(f"Prepared {count} training samples")
        return count

    def prepare_data_from_bixbench(
        self,
        benchmark_file: Optional[Path] = None,
    ) -> int:
        """
        Prepare training data from BixBench benchmark.

        Args:
            benchmark_file: Path to benchmark JSON

        Returns:
            Number of samples prepared
        """
        if benchmark_file is None:
            # Use default BixBench location
            benchmark_file = Path.home() / ".ct" / "data" / "bixbench.json"

        if not benchmark_file.exists():
            logger.warning(f"BixBench file not found: {benchmark_file}")
            return 0

        with open(benchmark_file) as f:
            benchmark_data = json.load(f)

        count = 0
        for item in benchmark_data.get("questions", []):
            self.training_data.add_sample(
                query=item.get("question", ""),
                tool_calls=[],  # Would need to generate these
                conclusion=item.get("answer", ""),
                rating=5,  # High quality benchmark data
            )
            count += 1

        logger.info(f"Prepared {count} BixBench samples")
        return count

    def prepare_data_from_file(self, data_file: Path) -> int:
        """Load training data from JSONL file."""
        count = 0

        with open(data_file) as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    self.training_data.samples.append(sample)
                    count += 1
                except json.JSONDecodeError:
                    continue

        logger.info(f"Loaded {count} samples from {data_file}")
        return count

    def train(
        self,
        output_dir: Optional[Path] = None,
        use_unsloth: bool = True,
    ) -> dict:
        """
        Run LoRA training.

        Args:
            output_dir: Output directory for adapter
            use_unsloth: Use Unsloth for faster training

        Returns:
            Training results
        """
        output_dir = output_dir or self.data_dir / self.config.output_dir

        # Export training data
        train_file = self.data_dir / "train.jsonl"
        num_samples = self.training_data.to_jsonl(train_file)

        if num_samples < 100:
            logger.warning(f"Only {num_samples} training samples - recommend 15K+ for good results")

        logger.info(f"Starting LoRA training with {num_samples} samples...")

        start_time = time.time()

        try:
            if use_unsloth:
                results = self._train_with_unsloth(train_file, output_dir)
            else:
                results = self._train_with_axolotl(train_file, output_dir)

            self._is_trained = True
            results["num_samples"] = num_samples
            results["training_time_seconds"] = time.time() - start_time

            logger.info(f"Training complete in {results['training_time_seconds']:.1f}s")
            return results

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {"error": str(e)}

    def _train_with_unsloth(
        self,
        train_file: Path,
        output_dir: Path,
    ) -> dict:
        """Train using Unsloth library."""
        try:
            from unsloth import FastLanguageModel
            from trl import SFTTrainer
            from transformers import TrainingArguments
            import torch

            # Load base model
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.base_model,
                max_seq_length=self.config.max_seq_length,
                dtype=None,  # Auto-detect
                load_in_4bit=self.config.quantize_base,
            )

            # Add LoRA adapters
            model = FastLanguageModel.get_peft_model(
                model,
                r=self.config.lora_rank,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                use_gradient_checkpointing=self.config.use_gradient_checkpointing,
                random_state=42,
            )

            # Training arguments
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                warmup_ratio=self.config.warmup_ratio,
                weight_decay=self.config.weight_decay,
                logging_steps=10,
                save_steps=100,
                save_total_limit=3,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                optim="adamw_8bit",
            )

            # Create trainer
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=self._load_dataset(train_file),
                dataset_text_field="text",
                max_seq_length=self.config.max_seq_length,
                args=training_args,
            )

            # Train
            trainer.train()

            # Save adapter
            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))

            return {
                "status": "success",
                "adapter_path": str(output_dir),
            }

        except ImportError:
            logger.warning("Unsloth not available, falling back to Axolotl")
            return self._train_with_axolotl(train_file, output_dir)

    def _train_with_axolotl(
        self,
        train_file: Path,
        output_dir: Path,
    ) -> dict:
        """Train using Axolotl library."""
        # Create Axolotl config
        config = {
            "base_model": self.config.base_model,
            "base_model_config": self.config.base_model,
            "model_type": "LlamaForCausalLM",
            "tokenizer_type": "LlamaTokenizer",
            "load_in_8bit": False,
            "load_in_4bit": self.config.quantize_base,
            "strict": False,
            "datasets": [
                {
                    "path": str(train_file),
                    "type": "alpaca",
                }
            ],
            "dataset_prepared_path": str(self.data_dir / "prepared"),
            "val_set_size": 0.05,
            "output_dir": str(output_dir),
            "adapter": "lora",
            "lora_r": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "sequence_len": self.config.max_seq_length,
            "sample_packing": True,
            "eval_sample_packing": False,
            "num_epochs": self.config.num_epochs,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "micro_batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "optimizer": "adamw_bnb_8bit",
            "lr_scheduler": "cosine",
            "warmup_ratio": self.config.warmup_ratio,
            "save_strategy": "epoch",
        }

        config_path = self.data_dir / "axolotl_config.yaml"
        import yaml
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Run Axolotl training
        result = subprocess.run(
            ["axolotl", "train", str(config_path)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return {"error": result.stderr}

        return {
            "status": "success",
            "adapter_path": str(output_dir),
        }

    def _load_dataset(self, train_file: Path):
        """Load training dataset."""
        from datasets import Dataset

        samples = []
        with open(train_file) as f:
            for line in f:
                sample = json.loads(line)
                # Format for instruction tuning
                text = f"### Instruction:\n{sample['instruction']}\n\n### Response:\n{sample['output']}"
                samples.append({"text": text})

        return Dataset.from_list(samples)

    def merge_and_export(
        self,
        adapter_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Merge LoRA adapter with base model and export.

        Args:
            adapter_path: Path to trained adapter
            output_path: Output path for merged model

        Returns:
            Path to merged model
        """
        adapter_path = adapter_path or self.data_dir / self.config.output_dir
        output_path = output_path or self.data_dir / "merged_model"

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                torch_dtype="auto",
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)

            # Load adapter
            model = PeftModel.from_pretrained(base_model, str(adapter_path))

            # Merge
            model = model.merge_and_unload()

            # Save
            model.save_pretrained(str(output_path))
            tokenizer.save_pretrained(str(output_path))

            logger.info(f"Merged model saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Merge failed: {e}")
            raise

    def export_for_ollama(
        self,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Export model for Ollama."""
        output_path = output_path or self.data_dir / "ollama_model"

        # Create Modelfile
        modelfile = f"""FROM {self.config.base_model}
PARAMETER temperature {self.config.temperature}
PARAMETER num_ctx {self.config.max_seq_length}
SYSTEM You are CellType-Agent, an AI assistant for drug discovery research.
"""

        modelfile_path = output_path / "Modelfile"
        output_path.mkdir(parents=True, exist_ok=True)

        with open(modelfile_path, "w") as f:
            f.write(modelfile)

        logger.info(f"Ollama Modelfile created at {modelfile_path}")
        return modelfile_path

    def get_training_stats(self) -> dict:
        """Get training statistics."""
        return {
            "num_samples": len(self.training_data.samples),
            "is_trained": self._is_trained,
            "config": {
                "lora_rank": self.config.lora_rank,
                "learning_rate": self.config.learning_rate,
                "base_model": self.config.base_model,
            },
        }