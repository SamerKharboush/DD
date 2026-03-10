"""
RLEF Trainer for CellType-Agent.

Implements Reinforcement Learning from Experimental Feedback:
- Collects user feedback on agent responses
- Trains reward model from preferences
- Fine-tunes agent with PPO/DPO
- Enables continuous self-improvement
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ct.rlef.trainer")


@dataclass
class RLEFConfig:
    """Configuration for RLEF training."""
    reward_model: str = "distilroberta-base"
    policy_model: str = "meta-llama/Llama-3-8B-Instruct"
    output_dir: str = "rlef-checkpoints"
    learning_rate: float = 1e-5
    batch_size: int = 8
    ppo_epochs: int = 4
    kl_penalty: float = 0.1
    clip_range: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_length: int = 2048
    temperature: float = 1.0
    top_p: float = 0.9
    use_dpo: bool = True


@dataclass
class FeedbackSample:
    """A single feedback sample."""
    query: str
    response: str
    tool_calls: list[dict] = field(default_factory=list)
    conclusion: str = ""
    rating: int = 0
    comments: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "response": self.response,
            "tool_calls": self.tool_calls,
            "conclusion": self.conclusion,
            "rating": self.rating,
            "comments": self.comments,
            "timestamp": self.timestamp,
        }


class RLEFTrainer:
    """
    RLEF Training pipeline.

    Workflow:
    1. Collect feedback from sessions
    2. Build preference pairs
    3. Train reward model
    4. Fine-tune policy with PPO/DPO
    5. Evaluate and deploy
    """

    def __init__(
        self,
        config: Optional[RLEFConfig] = None,
        data_dir: Optional[Path] = None,
    ):
        self.config = config or RLEFConfig()
        self.data_dir = Path(data_dir) if data_dir else Path.home() / ".ct" / "rlef"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.feedback_samples: list[FeedbackSample] = []
        self.preference_pairs: list[tuple[dict, dict]] = []
        self._reward_model = None
        self._policy_model = None
        self._tokenizer = None

    def load_feedback_from_sessions(
        self,
        min_rating: int = 1,
        limit: Optional[int] = None,
    ) -> int:
        """Load feedback from session logs."""
        from ct.session_logging import SessionLogger

        logger_instance = SessionLogger()
        sessions = logger_instance.get_training_data(
            min_quality_score=min_rating / 5.0,
            limit=limit or 100000,
        )

        count = 0
        for session in sessions:
            rating = session.get("user_rating", 0)
            if rating >= min_rating:
                sample = FeedbackSample(
                    query=session.get("query", ""),
                    response=session.get("response", ""),
                    tool_calls=session.get("tool_calls", []),
                    conclusion=session.get("conclusion", ""),
                    rating=rating,
                    comments=session.get("feedback_text", ""),
                )
                self.feedback_samples.append(sample)
                count += 1

        logger.info(f"Loaded {count} feedback samples")
        return count

    def load_feedback_from_file(self, filepath: Path) -> int:
        """Load feedback from JSONL file."""
        count = 0
        with open(filepath) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    sample = FeedbackSample(
                        query=data.get("query", ""),
                        response=data.get("response", ""),
                        tool_calls=data.get("tool_calls", []),
                        conclusion=data.get("conclusion", ""),
                        rating=data.get("rating", 0),
                        comments=data.get("comments", ""),
                    )
                    self.feedback_samples.append(sample)
                    count += 1
                except json.JSONDecodeError:
                    continue

        logger.info(f"Loaded {count} samples from {filepath}")
        return count

    def build_preference_pairs(self) -> int:
        """Build preference pairs from feedback."""
        from collections import defaultdict

        query_groups = defaultdict(list)
        for sample in self.feedback_samples:
            query_key = sample.query.lower().strip()[:100]
            query_groups[query_key].append(sample)

        self.preference_pairs = []

        for query_key, samples in query_groups.items():
            if len(samples) < 2:
                continue

            samples_sorted = sorted(samples, key=lambda s: s.rating, reverse=True)

            for i, high in enumerate(samples_sorted):
                for low in samples_sorted[i+1:]:
                    if high.rating > low.rating:
                        self.preference_pairs.append((
                            high.to_dict(),
                            low.to_dict(),
                        ))

        logger.info(f"Created {len(self.preference_pairs)} preference pairs")
        return len(self.preference_pairs)

    def train_reward_model(self) -> dict:
        """Train reward model from preferences."""
        if not self.preference_pairs:
            logger.warning("No preference pairs - run build_preference_pairs() first")
            return {"error": "No preference pairs"}

        logger.info(f"Training reward model on {len(self.preference_pairs)} pairs...")

        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                TrainingArguments,
                Trainer,
            )
            from datasets import Dataset
            import torch

            self._reward_model = AutoModelForSequenceClassification.from_pretrained(
                self.config.reward_model,
                num_labels=1,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.reward_model)

            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            def format_pair(pair):
                chosen, rejected = pair
                return {
                    "chosen": f"Query: {chosen['query']}\nResponse: {chosen['response']}",
                    "rejected": f"Query: {rejected['query']}\nResponse: {rejected['response']}",
                }

            data = [format_pair(p) for p in self.preference_pairs]
            dataset = Dataset.from_list(data)

            def tokenize_fn(examples):
                chosen_tokens = self._tokenizer(
                    examples["chosen"],
                    truncation=True,
                    max_length=self.config.max_length,
                    padding="max_length",
                )
                rejected_tokens = self._tokenizer(
                    examples["rejected"],
                    truncation=True,
                    max_length=self.config.max_length,
                    padding="max_length",
                )
                return {
                    "chosen_input_ids": chosen_tokens["input_ids"],
                    "chosen_attention_mask": chosen_tokens["attention_mask"],
                    "rejected_input_ids": rejected_tokens["input_ids"],
                    "rejected_attention_mask": rejected_tokens["attention_mask"],
                }

            tokenized_dataset = dataset.map(tokenize_fn, batched=True)

            output_dir = self.data_dir / "reward_model"
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=3,
                per_device_train_batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                save_strategy="epoch",
                logging_steps=10,
            )

            class RewardTrainer(Trainer):
                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    chosen_rewards = model(
                        input_ids=inputs["chosen_input_ids"],
                        attention_mask=inputs["chosen_attention_mask"],
                    ).logits

                    rejected_rewards = model(
                        input_ids=inputs["rejected_input_ids"],
                        attention_mask=inputs["rejected_attention_mask"],
                    ).logits

                    loss = -torch.nn.functional.logsigmoid(
                        chosen_rewards - rejected_rewards
                    ).mean()

                    return (loss, {"chosen_reward": chosen_rewards.mean()}) if return_outputs else loss

            trainer = RewardTrainer(
                model=self._reward_model,
                args=training_args,
                train_dataset=tokenized_dataset,
            )

            trainer.train()

            self._reward_model.save_pretrained(str(output_dir))
            self._tokenizer.save_pretrained(str(output_dir))

            return {
                "status": "success",
                "model_path": str(output_dir),
                "num_pairs": len(self.preference_pairs),
            }

        except ImportError as e:
            logger.error(f"Missing dependencies: {e}")
            return {"error": str(e)}

    def train_policy_dpo(self) -> dict:
        """Train policy with Direct Preference Optimization."""
        if not self.preference_pairs:
            logger.warning("No preference pairs - run build_preference_pairs() first")
            return {"error": "No preference pairs"}

        logger.info(f"Training policy with DPO on {len(self.preference_pairs)} pairs...")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from datasets import Dataset
            import torch

            self._policy_model = AutoModelForCausalLM.from_pretrained(
                self.config.policy_model,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.policy_model)

            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            ref_model = AutoModelForCausalLM.from_pretrained(
                self.config.policy_model,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            ref_model.eval()

            def format_pair(pair):
                chosen, rejected = pair
                prompt = f"### Instruction:\n{chosen['query']}\n\n### Response:\n"
                return {
                    "prompt": prompt,
                    "chosen": chosen["response"],
                    "rejected": rejected["response"],
                }

            data = [format_pair(p) for p in self.preference_pairs[:1000]]
            dataset = Dataset.from_list(data)

            optimizer = torch.optim.AdamW(
                self._policy_model.parameters(),
                lr=self.config.learning_rate,
            )

            output_dir = self.data_dir / self.config.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)

            global_step = 0
            for epoch in range(self.config.ppo_epochs):
                for batch in dataset.iter(batch_size=self.config.batch_size):
                    prompt_enc = self._tokenizer(
                        batch["prompt"],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.config.max_length,
                    ).to(self._policy_model.device)

                    chosen_enc = self._tokenizer(
                        [p + c for p, c in zip(batch["prompt"], batch["chosen"])],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.config.max_length,
                    ).to(self._policy_model.device)

                    rejected_enc = self._tokenizer(
                        [p + r for p, r in zip(batch["prompt"], batch["rejected"])],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.config.max_length,
                    ).to(self._policy_model.device)

                    with torch.no_grad():
                        ref_chosen_logp = self._compute_log_prob(ref_model, chosen_enc)
                        ref_rejected_logp = self._compute_log_prob(ref_model, rejected_enc)

                    policy_chosen_logp = self._compute_log_prob(self._policy_model, chosen_enc)
                    policy_rejected_logp = self._compute_log_prob(self._policy_model, rejected_enc)

                    chosen_ratio = policy_chosen_logp - ref_chosen_logp
                    rejected_ratio = policy_rejected_logp - ref_rejected_logp

                    loss = -torch.nn.functional.logsigmoid(
                        self.config.kl_penalty * (chosen_ratio - rejected_ratio)
                    ).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    global_step += 1
                    if global_step % 10 == 0:
                        logger.info(f"Step {global_step}, Loss: {loss.item():.4f}")

            self._policy_model.save_pretrained(str(output_dir))
            self._tokenizer.save_pretrained(str(output_dir))

            return {
                "status": "success",
                "model_path": str(output_dir),
                "global_steps": global_step,
            }

        except ImportError as e:
            logger.error(f"Missing dependencies: {e}")
            return {"error": str(e)}

    def _compute_log_prob(self, model, encodings):
        """Compute log probability of sequence."""
        import torch

        outputs = model(
            input_ids=encodings["input_ids"],
            attention_mask=encodings["attention_mask"],
            labels=encodings["input_ids"],
        )

        return -outputs.loss * encodings["input_ids"].shape[1]

    def train(
        self,
        method: str = "dpo",
        session_file: Optional[str] = None,
    ) -> dict:
        """Main training entry point."""
        if session_file:
            self.load_feedback_from_file(Path(session_file))
        elif not self.feedback_samples:
            self.load_feedback_from_sessions()

        if not self.feedback_samples:
            return {"error": "No feedback samples available"}

        self.build_preference_pairs()

        if method == "dpo":
            return self.train_policy_dpo()
        else:
            return self.train_reward_model()

    def export_model(self, output_path: Optional[Path] = None) -> Path:
        """Export trained model for use with local LLM."""
        output_path = output_path or self.data_dir / "rlef_adapter"

        if self._policy_model:
            self._policy_model.save_pretrained(str(output_path))
            self._tokenizer.save_pretrained(str(output_path))
            logger.info(f"Model exported to {output_path}")
            return output_path

        raise ValueError("No trained model to export")

    def get_training_stats(self) -> dict:
        """Get training statistics."""
        return {
            "num_feedback_samples": len(self.feedback_samples),
            "num_preference_pairs": len(self.preference_pairs),
            "config": {
                "reward_model": self.config.reward_model,
                "policy_model": self.config.policy_model,
                "method": "dpo" if self.config.use_dpo else "ppo",
            },
        }


def create_training_data_from_sessions(
    output_file: str,
    min_rating: int = 3,
) -> int:
    """Create training data file from session logs."""
    trainer = RLEFTrainer()
    count = trainer.load_feedback_from_sessions(min_rating=min_rating)

    output_path = Path(output_file)
    with open(output_path, "w") as f:
        for sample in trainer.feedback_samples:
            f.write(json.dumps(sample.to_dict()) + "\n")

    logger.info(f"Wrote {count} samples to {output_path}")
    return count