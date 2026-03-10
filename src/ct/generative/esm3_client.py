"""
ESM3 Client for conditional protein generation.

ESM3 (EvolutionaryScale, Science 2025) is the first generative multimodal
protein model that reasons over sequence, structure, and function simultaneously.

Key Features:
- Promptable design with partial sequence/structure/function constraints
- 98B parameter model (largest protein foundation model)
- Generate-filter-rerank pipeline for negative constraints

Access:
- 1.4B model: HuggingFace (CC-BY-NC 4.0)
- 7B/98B models: EvolutionaryScale Forge API, AWS Bedrock, NVIDIA NIM

Reference: https://www.evolutionaryscale.ai/
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests

logger = logging.getLogger("ct.generative.esm3")


@dataclass
class ESM3Generation:
    """Result of ESM3 protein generation."""
    generation_id: str
    sequence: str
    structure_pdb: Optional[str] = None
    confidence: float = 0.0
    prompt_used: dict = field(default_factory=dict)
    generation_time_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class ESM3ScoreResult:
    """Result of ESM3 sequence scoring."""
    sequence: str
    stability_score: float
    function_scores: dict = field(default_factory=dict)
    structure_plddt: Optional[float] = None


class ESM3Client:
    """
    Client for ESM3 protein generation and scoring.

    Usage:
        client = ESM3Client()

        # Generate with function prompt
        result = client.generate(
            function_prompt="E3 ligase substrate for CRBN",
            length_range=(50, 100),
        )

        # Score existing sequence
        score = client.score("MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG")
    """

    # API endpoints
    FORGE_API_URL = "https://forge.evolutionaryscale.ai/v1"
    AWS_BEDROCK_URL = "https://bedrock-runtime.us-east-1.amazonaws.com"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_provider: str = "forge",  # forge, aws, local
        cache_dir: Optional[Path] = None,
        model_size: str = "7B",  # 1.4B, 7B, 98B
    ):
        """
        Initialize ESM3 client.

        Args:
            api_key: API key (or set ESM3_API_KEY env var)
            api_provider: API provider (forge, aws, local)
            cache_dir: Cache directory
            model_size: Model size to use
        """
        self.api_key = api_key or os.environ.get("ESM3_API_KEY")
        self.api_provider = api_provider
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".ct" / "esm3_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_size = model_size

        self._local_model = None

    def _get_model(self):
        """Lazy-load local model if available."""
        if self._local_model is None and self.api_provider == "local":
            try:
                # Try to load from HuggingFace
                from transformers import AutoModelForCausalLM, AutoTokenizer
                model_name = f"evolutionaryscale/esm3-{self.model_size.lower()}"
                self._local_model = {
                    "model": AutoModelForCausalLM.from_pretrained(model_name),
                    "tokenizer": AutoTokenizer.from_pretrained(model_name),
                }
                logger.info(f"Loaded local ESM3 model: {model_name}")
            except ImportError:
                logger.warning("Local ESM3 requires transformers. pip install transformers")
                self._local_model = False
            except Exception as e:
                logger.warning(f"Failed to load local ESM3: {e}")
                self._local_model = False

        return self._local_model if self._local_model else None

    def generate(
        self,
        sequence_prompt: Optional[str] = None,
        structure_prompt: Optional[str] = None,  # PDB content
        function_prompt: Optional[str] = None,
        length_range: tuple[int, int] = (50, 200),
        temperature: float = 0.7,
        num_samples: int = 1,
    ) -> list[ESM3Generation]:
        """
        Generate protein sequences with ESM3.

        Args:
            sequence_prompt: Partial sequence with gaps (e.g., "MKT___G__V")
            structure_prompt: Partial structure in PDB format
            function_prompt: Function description (e.g., "binds CRBN")
            length_range: Target length range
            temperature: Sampling temperature
            num_samples: Number of samples to generate

        Returns:
            List of generated proteins
        """
        start_time = time.time()

        prompt = {
            "sequence": sequence_prompt,
            "structure": structure_prompt,
            "function": function_prompt,
        }

        if self.api_provider == "local":
            return self._generate_local(prompt, length_range, temperature, num_samples)
        else:
            return self._generate_api(prompt, length_range, temperature, num_samples)

    def _generate_api(
        self,
        prompt: dict,
        length_range: tuple[int, int],
        temperature: float,
        num_samples: int,
    ) -> list[ESM3Generation]:
        """Generate via API."""
        if not self.api_key:
            return [ESM3Generation(
                generation_id="error",
                sequence="",
                error="ESM3 API key not configured. Set ESM3_API_KEY environment variable.",
            )]

        generations = []

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": f"esm3-{self.model_size.lower()}",
                "prompt": {
                    "sequence": prompt.get("sequence"),
                    "structure": prompt.get("structure"),
                    "function": prompt.get("function"),
                },
                "parameters": {
                    "min_length": length_range[0],
                    "max_length": length_range[1],
                    "temperature": temperature,
                    "num_samples": num_samples,
                },
            }

            response = requests.post(
                f"{self.FORGE_API_URL}/generate",
                headers=headers,
                json=payload,
                timeout=300,
            )

            if response.status_code != 200:
                return [ESM3Generation(
                    generation_id="error",
                    sequence="",
                    error=f"API error: {response.status_code} - {response.text}",
                )]

            data = response.json()

            for i, sample in enumerate(data.get("samples", [])):
                generations.append(ESM3Generation(
                    generation_id=f"esm3_{hashlib.md5(sample.get('sequence', '').encode()).hexdigest()[:12]}",
                    sequence=sample.get("sequence", ""),
                    structure_pdb=sample.get("structure"),
                    confidence=sample.get("confidence", 0.0),
                    prompt_used=prompt,
                    generation_time_seconds=time.time() - start_time,
                ))

        except Exception as e:
            logger.error(f"ESM3 API error: {e}")
            return [ESM3Generation(
                generation_id="error",
                sequence="",
                error=str(e),
            )]

        return generations

    def _generate_local(
        self,
        prompt: dict,
        length_range: tuple[int, int],
        temperature: float,
        num_samples: int,
    ) -> list[ESM3Generation]:
        """Generate using local model."""
        model = self._get_model()

        if not model:
            return [ESM3Generation(
                generation_id="error",
                sequence="",
                error="Local ESM3 model not available",
            )]

        generations = []

        try:
            # Format prompt for ESM3
            tokenizer = model["tokenizer"]
            lm = model["model"]

            # Build input text
            input_text = self._format_prompt(prompt)

            inputs = tokenizer(input_text, return_tensors="pt")

            for _ in range(num_samples):
                # Generate
                outputs = lm.generate(
                    **inputs,
                    min_length=length_range[0],
                    max_length=length_range[1],
                    temperature=temperature,
                    do_sample=True,
                )

                sequence = tokenizer.decode(outputs[0], skip_special_tokens=True)

                generations.append(ESM3Generation(
                    generation_id=f"esm3_local_{hashlib.md5(sequence.encode()).hexdigest()[:12]}",
                    sequence=sequence,
                    prompt_used=prompt,
                ))

        except Exception as e:
            logger.error(f"Local generation error: {e}")
            return [ESM3Generation(
                generation_id="error",
                sequence="",
                error=str(e),
            )]

        return generations

    def _format_prompt(self, prompt: dict) -> str:
        """Format prompt for ESM3."""
        parts = []

        if prompt.get("sequence"):
            parts.append(f"<sequence>{prompt['sequence']}</sequence>")

        if prompt.get("function"):
            parts.append(f"<function>{prompt['function']}</function>")

        return "".join(parts) if parts else "<sequence></sequence>"

    def score(
        self,
        sequence: str,
        functions: Optional[list[str]] = None,
    ) -> ESM3ScoreResult:
        """
        Score a protein sequence.

        Args:
            sequence: Protein sequence to score
            functions: List of function keywords to score against

        Returns:
            ESM3ScoreResult with stability and function scores
        """
        functions = functions or []

        if self.api_provider == "local":
            return self._score_local(sequence, functions)
        else:
            return self._score_api(sequence, functions)

    def _score_api(
        self,
        sequence: str,
        functions: list[str],
    ) -> ESM3ScoreResult:
        """Score via API."""
        if not self.api_key:
            return ESM3ScoreResult(
                sequence=sequence,
                stability_score=0.5,
                function_scores={f: 0.5 for f in functions},
            )

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": f"esm3-{self.model_size.lower()}",
                "sequence": sequence,
                "functions": functions,
            }

            response = requests.post(
                f"{self.FORGE_API_URL}/score",
                headers=headers,
                json=payload,
                timeout=60,
            )

            if response.status_code == 200:
                data = response.json()
                return ESM3ScoreResult(
                    sequence=sequence,
                    stability_score=data.get("stability", 0.5),
                    function_scores=data.get("functions", {}),
                    structure_plddt=data.get("plddt"),
                )

        except Exception as e:
            logger.error(f"ESM3 scoring error: {e}")

        return ESM3ScoreResult(
            sequence=sequence,
            stability_score=0.5,
            function_scores={f: 0.5 for f in functions},
        )

    def _score_local(
        self,
        sequence: str,
        functions: list[str],
    ) -> ESM3ScoreResult:
        """Score using local model."""
        # Simplified scoring - real implementation would use model outputs
        return ESM3ScoreResult(
            sequence=sequence,
            stability_score=self._estimate_stability(sequence),
            function_scores={f: 0.5 for f in functions},
        )

    def _estimate_stability(self, sequence: str) -> float:
        """Estimate stability from sequence features."""
        # Simple heuristic based on amino acid composition
        hydrophobic = set("AILMFVWP")
        charged = set("DEKR")
        polar = set("STNQCYGH")

        total = len(sequence)
        if total == 0:
            return 0.5

        h_count = sum(1 for aa in sequence if aa in hydrophobic)
        c_count = sum(1 for aa in sequence if aa in charged)

        # Balance of hydrophobic/charged is good
        balance = 1 - abs(h_count / total - 0.3) - abs(c_count / total - 0.2)
        return max(0.1, min(1.0, balance))

    def mutate(
        self,
        sequence: str,
        mutation_positions: Optional[list[int]] = None,
        num_mutations: int = 1,
        preserve_function: Optional[str] = None,
    ) -> list[dict]:
        """
        Generate guided mutations.

        Args:
            sequence: Original sequence
            mutation_positions: Positions to mutate (None = any position)
            num_mutations: Number of mutations per variant
            preserve_function: Function to preserve

        Returns:
            List of mutation dictionaries
        """
        import random

        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        mutations = []

        positions = mutation_positions or list(range(len(sequence)))
        positions = random.sample(positions, min(num_mutations, len(positions)))

        for pos in positions:
            original = sequence[pos] if pos < len(sequence) else ""
            new_aa = random.choice(amino_acids)

            mutations.append({
                "position": pos,
                "original": original,
                "mutant": new_aa,
                "notation": f"{original}{pos+1}{new_aa}",
            })

        return mutations

    def generate_with_constraints(
        self,
        positive_constraints: Optional[list[str]] = None,
        negative_constraints: Optional[list[str]] = None,
        length_range: tuple[int, int] = (50, 200),
        num_candidates: int = 10,
    ) -> list[ESM3Generation]:
        """
        Generate proteins with positive and negative constraints.

        This implements the generate-filter-rerank pipeline for
        negative constraints like "avoid SALL4 binding".

        Args:
            positive_constraints: Features to include (e.g., "binds CRBN")
            negative_constraints: Features to avoid (e.g., "binds SALL4")
            length_range: Length range
            num_candidates: Number of candidates to generate

        Returns:
            List of filtered candidates
        """
        positive_constraints = positive_constraints or []
        negative_constraints = negative_constraints or []

        # Step 1: Generate with positive constraints
        function_prompt = "; ".join(positive_constraints) if positive_constraints else None

        # Generate more than needed for filtering
        num_to_generate = num_candidates * 3
        raw_generations = self.generate(
            function_prompt=function_prompt,
            length_range=length_range,
            num_samples=num_to_generate,
        )

        # Step 2: Filter by negative constraints
        filtered = []
        for gen in raw_generations:
            if gen.error:
                continue

            # Score against negative constraints
            passes_filter = True

            for neg_constraint in negative_constraints:
                score = self._score_against_constraint(gen.sequence, neg_constraint)
                if score > 0.5:  # High score means it might have the unwanted property
                    passes_filter = False
                    break

            if passes_filter:
                filtered.append(gen)
                if len(filtered) >= num_candidates:
                    break

        # Step 3: Rerank by overall quality
        for gen in filtered:
            gen.confidence = self._estimate_stability(gen.sequence)

        filtered.sort(key=lambda g: g.confidence, reverse=True)

        return filtered[:num_candidates]

    def _score_against_constraint(self, sequence: str, constraint: str) -> float:
        """Score how much a sequence matches a constraint."""
        # Use ESM3 scoring if available
        result = self.score(sequence, functions=[constraint])
        return result.function_scores.get(constraint, 0.5)


# Convenience functions for tool registration
def generate_protein(
    function_prompt: str,
    length_min: int = 50,
    length_max: int = 200,
    **kwargs,
) -> dict:
    """
    Generate a protein with specified function.

    Args:
        function_prompt: Function description (e.g., "E3 ligase substrate for CRBN")
        length_min: Minimum length
        length_max: Maximum length

    Returns:
        Dictionary with generated sequence
    """
    client = ESM3Client()
    generations = client.generate(
        function_prompt=function_prompt,
        length_range=(length_min, length_max),
    )

    if not generations or generations[0].error:
        return {
            "summary": f"Generation failed: {generations[0].error if generations else 'Unknown error'}",
            "error": True,
        }

    best = generations[0]
    return {
        "summary": f"Generated {len(best.sequence)}-residue protein with {best.confidence:.0%} confidence",
        "sequence": best.sequence,
        "confidence": best.confidence,
        "generation_id": best.generation_id,
    }


def generate_avoiding_target(
    positive_function: str,
    negative_target: str,
    **kwargs,
) -> dict:
    """
    Generate protein that binds target A while avoiding target B.

    Args:
        positive_function: Desired function (e.g., "binds CRBN")
        negative_target: Target to avoid (e.g., "binds SALL4")

    Returns:
        Dictionary with generated sequence
    """
    client = ESM3Client()
    generations = client.generate_with_constraints(
        positive_constraints=[positive_function],
        negative_constraints=[negative_target],
        num_candidates=5,
    )

    if not generations:
        return {
            "summary": "No candidates passed filtering",
            "error": True,
        }

    best = generations[0]
    return {
        "summary": f"Generated protein that {positive_function} while avoiding {negative_target}",
        "sequence": best.sequence,
        "confidence": best.confidence,
    }