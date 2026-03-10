"""
Preference Optimizer for RLEF.

Implements advanced preference optimization methods:
- Direct Preference Optimization (DPO)
- Kahneman-Tversky Optimization (KTO)
- Odds Ratio Preference Optimization (ORPO)
"""

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger("ct.rlef.preference")


@dataclass
class PreferenceConfig:
    """Configuration for preference optimization."""
    beta: float = 0.1  # KL penalty coefficient
    learning_rate: float = 5e-7
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_length: int = 2048
    max_prompt_length: int = 1024
    gamma: float = 1.0  # For KTO
    label_smoothing: float = 0.0


class PreferenceOptimizer:
    """
    Advanced preference optimization methods.

    Supports:
    - DPO: Direct Preference Optimization
    - KTO: Kahneman-Tversky Optimization
    - ORPO: Odds Ratio Preference Optimization

    Usage:
        optimizer = PreferenceOptimizer(policy_model, ref_model)
        loss = optimizer.compute_dpo_loss(chosen, rejected)
    """

    def __init__(
        self,
        policy_model,
        reference_model,
        config: Optional[PreferenceConfig] = None,
    ):
        """
        Initialize preference optimizer.

        Args:
            policy_model: The model being trained
            reference_model: The frozen reference model
            config: Optimization configuration
        """
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.config = config or PreferenceConfig()

    def compute_dpo_loss(
        self,
        chosen_logprobs: torch.Tensor,
        rejected_logprobs: torch.Tensor,
        reference_chosen_logprobs: torch.Tensor,
        reference_rejected_logprobs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute DPO loss.

        DPO loss: -log(sigmoid(beta * (log(p_chosen/p_ref_chosen) - log(p_rejected/p_ref_rejected))))

        Args:
            chosen_logprobs: Log probs of chosen responses under policy
            rejected_logprobs: Log probs of rejected responses under policy
            reference_chosen_logprobs: Log probs of chosen under reference
            reference_rejected_logprobs: Log probs of rejected under reference

        Returns:
            DPO loss tensor
        """
        # Compute log ratios
        chosen_logratios = chosen_logprobs - reference_chosen_logprobs
        rejected_logratios = rejected_logprobs - reference_rejected_logprobs

        # DPO loss
        logits = self.config.beta * (chosen_logratios - rejected_logratios)
        loss = -F.logsigmoid(logits).mean()

        return loss

    def compute_kto_loss(
        self,
        policy_logprobs: torch.Tensor,
        reference_logprobs: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KTO (Kahneman-Tversky Optimization) loss.

        KTO uses prospect theory to weight gains and losses differently.

        Args:
            policy_logprobs: Log probs under policy
            reference_logprobs: Log probs under reference
            labels: +1 for preferred, -1 for dispreferred

        Returns:
            KTO loss tensor
        """
        # KL divergence
        kl = policy_logprobs - reference_logprobs

        # Prospect theory weighting
        # Losses are weighted more heavily than gains (loss aversion)
        lambda_plus = 1.0  # Weight for gains
        lambda_minus = self.config.gamma  # Weight for losses (typically > 1)

        # Compute weighted rewards
        rewards = self.config.beta * kl

        # Apply prospect theory weighting
        weighted_rewards = torch.where(
            labels > 0,
            lambda_plus * F.sigmoid(rewards),
            -lambda_minus * F.sigmoid(-rewards),
        )

        # KTO loss
        loss = -weighted_rewards.mean()

        return loss

    def compute_orpo_loss(
        self,
        chosen_logprobs: torch.Tensor,
        rejected_logprobs: torch.Tensor,
        sft_loss: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ORPO (Odds Ratio Preference Optimization) loss.

        ORPO combines SFT with preference optimization using odds ratio.

        Args:
            chosen_logprobs: Log probs of chosen responses
            rejected_logprobs: Log probs of rejected responses
            sft_loss: Supervised fine-tuning loss

        Returns:
            ORPO loss tensor
        """
        # Compute odds ratio
        log_odds = chosen_logprobs - rejected_logprobs
        log_odds_ratio = log_odds - torch.log1p(-torch.exp(log_odds))

        # ORPO combines SFT loss with preference term
        orpo_loss = -F.logsigmoid(log_odds_ratio).mean()

        # Combined loss
        total_loss = sft_loss + self.config.beta * orpo_loss

        return total_loss

    def compute_ipo_loss(
        self,
        chosen_logprobs: torch.Tensor,
        rejected_logprobs: torch.Tensor,
        reference_chosen_logprobs: torch.Tensor,
        reference_rejected_logprobs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute IPO (Identity Preference Optimization) loss.

        IPO is a simplified variant of DPO with better theoretical properties.

        Args:
            chosen_logprobs: Log probs of chosen under policy
            rejected_logprobs: Log probs of rejected under policy
            reference_chosen_logprobs: Log probs of chosen under reference
            reference_rejected_logprobs: Log probs of rejected under reference

        Returns:
            IPO loss tensor
        """
        # Compute log ratios
        chosen_logratios = chosen_logprobs - reference_chosen_logprobs
        rejected_logratios = rejected_logprobs - reference_rejected_logprobs

        # IPO loss: (log_ratio - 1/(2*beta))^2
        target = 1.0 / (2 * self.config.beta)

        loss = 0.5 * (
            (chosen_logratios - target) ** 2 +
            (rejected_logratios + target) ** 2
        ).mean()

        return loss

    def optimize_step(
        self,
        batch: dict,
        optimizer: torch.optim.Optimizer,
        method: str = "dpo",
    ) -> dict:
        """
        Perform one optimization step.

        Args:
            batch: Batch containing chosen/rejected sequences
            optimizer: Optimizer to use
            method: Optimization method (dpo, kto, orpo, ipo)

        Returns:
            Dictionary with loss and metrics
        """
        self.policy_model.train()

        # Forward pass
        with torch.no_grad():
            ref_outputs = self.reference_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

        policy_outputs = self.policy_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        # Compute log probs
        policy_logprobs = self._compute_sequence_logprobs(
            policy_outputs.logits,
            batch["input_ids"],
        )
        ref_logprobs = self._compute_sequence_logprobs(
            ref_outputs.logits,
            batch["input_ids"],
        )

        # Split into chosen and rejected
        batch_size = batch["input_ids"].shape[0] // 2
        chosen_logprobs = policy_logprobs[:batch_size]
        rejected_logprobs = policy_logprobs[batch_size:]
        ref_chosen_logprobs = ref_logprobs[:batch_size]
        ref_rejected_logprobs = ref_logprobs[batch_size:]

        # Compute loss based on method
        if method == "dpo":
            loss = self.compute_dpo_loss(
                chosen_logprobs, rejected_logprobs,
                ref_chosen_logprobs, ref_rejected_logprobs,
            )
        elif method == "kto":
            labels = torch.cat([
                torch.ones(batch_size),
                -torch.ones(batch_size),
            ]).to(policy_logprobs.device)
            loss = self.compute_kto_loss(
                policy_logprobs, ref_logprobs, labels,
            )
        elif method == "ipo":
            loss = self.compute_ipo_loss(
                chosen_logprobs, rejected_logprobs,
                ref_chosen_logprobs, ref_rejected_logprobs,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute metrics
        with torch.no_grad():
            accuracy = (chosen_logprobs > rejected_logprobs).float().mean()
            margin = (chosen_logprobs - rejected_logprobs).mean()

        return {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "margin": margin.item(),
        }

    def _compute_sequence_logprobs(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probability of each sequence."""
        # Shift for autoregressive modeling
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        # Compute log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_logprobs = log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)

        # Sum over sequence
        return token_logprobs.sum(dim=-1)


class OnlinePreferenceLearner:
    """
    Online preference learning with continuous updates.

    Learns from user feedback in real-time and updates
    the model incrementally.
    """

    def __init__(
        self,
        model,
        optimizer: Optional[torch.optim.Optimizer] = None,
        buffer_size: int = 1000,
    ):
        """
        Initialize online learner.

        Args:
            model: Model to update
            optimizer: Optimizer to use
            buffer_size: Size of replay buffer
        """
        self.model = model
        self.optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=1e-6)
        self.buffer_size = buffer_size

        self.replay_buffer: list[dict] = []
        self.update_count = 0

    def add_feedback(
        self,
        query: str,
        chosen_response: str,
        rejected_response: str,
    ):
        """
        Add new feedback to buffer.

        Args:
            query: User query
            chosen_response: Preferred response
            rejected_response: Dispreferred response
        """
        sample = {
            "query": query,
            "chosen": chosen_response,
            "rejected": rejected_response,
            "timestamp": self.update_count,
        }

        self.replay_buffer.append(sample)

        # Maintain buffer size
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)

    def update(self, batch_size: int = 4) -> dict:
        """
        Perform online update.

        Args:
            batch_size: Batch size for update

        Returns:
            Update metrics
        """
        if len(self.replay_buffer) < batch_size:
            return {"error": "Insufficient samples"}

        # Sample from buffer
        import random
        batch_samples = random.sample(self.replay_buffer, batch_size)

        # TODO: Implement actual update
        self.update_count += 1

        return {
            "update_count": self.update_count,
            "buffer_size": len(self.replay_buffer),
        }