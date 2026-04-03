"""Binoculars zero-shot AI text detection — Alternative model pairs.

Uses Qwen2.5-7B (base) + Qwen2.5-7B-Instruct (instruct) instead of Falcon-7B.
Falls back to other ungated model pairs if needed.

Same scoring logic as detector.py (Binoculars ICML 2024), adapted for:
  - MPS (Apple Silicon Metal backend)
  - float16 (no BitsAndBytes quantization)
  - Alternative model pairs that may better detect modern AI text

Model pairs (in order of preference):
  1. Qwen/Qwen2.5-7B + Qwen/Qwen2.5-7B-Instruct (ungated, recommended)
  2. meta-llama/Llama-3.1-8B + meta-llama/Llama-3.1-8B-Instruct (gated)
  3. mistralai/Mistral-7B-v0.3 + mistralai/Mistral-7B-Instruct-v0.3
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# Model pair configurations
MODEL_PAIRS = {
    "qwen2.5-7b": {
        "observer": "Qwen/Qwen2.5-7B",
        "performer": "Qwen/Qwen2.5-7B-Instruct",
    },
    "llama-3.1-8b": {
        "observer": "meta-llama/Llama-3.1-8B",
        "performer": "meta-llama/Llama-3.1-8B-Instruct",
    },
    "mistral-7b-v0.3": {
        "observer": "mistralai/Mistral-7B-v0.3",
        "performer": "mistralai/Mistral-7B-Instruct-v0.3",
    },
    "gemma-2-2b": {
        "observer": "google/gemma-2-2b",
        "performer": "google/gemma-2-2b-it",
    },
}

DEFAULT_PAIR = "qwen2.5-7b"

# Thresholds from original Binoculars paper (may need recalibration)
THRESHOLD_LOW_FPR = 0.8536
THRESHOLD_ACCURACY = 0.9015

MIN_TOKENS = 50


def get_device(requested: str = "auto") -> torch.device:
    """Select best available device for Mac Studio."""
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


class BinocularsAltDetector:
    """Loads two models (base + instruct) and computes Binoculars scores.

    Mac Studio version: uses MPS backend with float16, no quantization.
    """

    def __init__(
        self,
        model_pair: str = DEFAULT_PAIR,
        device: Optional[torch.device] = None,
    ) -> None:
        if model_pair not in MODEL_PAIRS:
            raise ValueError(
                f"Unknown model pair '{model_pair}'. "
                f"Available: {list(MODEL_PAIRS.keys())}"
            )
        self.model_pair_name = model_pair
        pair = MODEL_PAIRS[model_pair]
        self.observer_name = pair["observer"]
        self.performer_name = pair["performer"]

        self.observer = None
        self.performer = None
        self.tokenizer = None
        self.device = device or get_device()

    def load_models(self) -> None:
        """Load both models. Call once at startup."""
        logger.info(
            "Loading Binoculars models (%s) on %s...",
            self.model_pair_name,
            self.device,
        )

        if self.device.type == "cpu":
            logger.warning("Running on CPU — inference will be very slow")

        dtype = torch.float16 if self.device.type == "mps" else torch.float32

        logger.info("Loading tokenizer from: %s", self.observer_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.observer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Loading observer: %s (dtype=%s)", self.observer_name, dtype)
        self.observer = AutoModelForCausalLM.from_pretrained(
            self.observer_name,
            torch_dtype=dtype,
            device_map=None,
        ).to(self.device)
        self.observer.eval()

        logger.info("Loading performer: %s (dtype=%s)", self.performer_name, dtype)
        self.performer = AutoModelForCausalLM.from_pretrained(
            self.performer_name,
            torch_dtype=dtype,
            device_map=None,
        ).to(self.device)
        self.performer.eval()

        logger.info(
            "Binoculars models (%s) loaded on %s",
            self.model_pair_name,
            self.device,
        )

    def compute_score(self, text: str) -> tuple[float, int]:
        """Compute the Binoculars score for a text chunk.

        Returns:
            (score, token_count) where score approaches 1.0 for AI-generated
            text and is lower for human-written text.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        token_count = input_ids.shape[1]

        with torch.no_grad():
            logits_observer = self.observer(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
            logits_performer = self.performer(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits

        # Shift: predict next token from current position
        shift_logits_observer = logits_observer[:, :-1].contiguous()
        shift_logits_performer = logits_performer[:, :-1].contiguous()
        labels = input_ids[:, 1:].contiguous()

        # Standard perplexity under observer (log PPL)
        # On MPS, cross_entropy may need float32 for numerical stability
        if self.device.type == "mps":
            ppl = F.cross_entropy(
                shift_logits_observer.float().view(-1, shift_logits_observer.size(-1)),
                labels.view(-1),
                reduction="mean",
            )
        else:
            ppl = F.cross_entropy(
                shift_logits_observer.view(-1, shift_logits_observer.size(-1)),
                labels.view(-1),
                reduction="mean",
            )

        # Cross-perplexity: how surprised observer is by performer's distribution
        performer_probs = F.softmax(shift_logits_performer.float(), dim=-1)
        observer_log_probs = F.log_softmax(shift_logits_observer.float(), dim=-1)
        x_ppl = -(performer_probs * observer_log_probs).sum(dim=-1).mean()

        score = (ppl / x_ppl).item()
        return score, token_count

    def classify(self, score: float, threshold_mode: str = "low_fpr") -> str:
        """Classify a score as ai_generated, human, or uncertain."""
        if score >= THRESHOLD_ACCURACY:
            return "ai_generated"
        elif score < THRESHOLD_LOW_FPR:
            return "human"
        else:
            if threshold_mode == "accuracy":
                return "human"
            return "uncertain"

    @property
    def is_loaded(self) -> bool:
        return self.observer is not None and self.performer is not None
