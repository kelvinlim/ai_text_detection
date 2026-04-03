"""Binoculars zero-shot AI text detection — Mac Studio (MPS) version.

Adapted from grant_assist/binoculars-service/app/detector.py.
Key changes from the Cloud Run/CUDA version:
  - CUDA → MPS (Apple Silicon Metal backend)
  - Removed BitsAndBytes 4-bit quantization (not supported on MPS)
  - Uses float16 on MPS (Mac Studio has enough unified memory)
  - Falls back to CPU if MPS unavailable

Implements the scoring method from:
  Binoculars: Zero-Shot Detection of LLM-Generated Text (ICML 2024)
  https://arxiv.org/abs/2401.12070
"""

import logging

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

OBSERVER_MODEL = "tiiuae/falcon-7b"
PERFORMER_MODEL = "tiiuae/falcon-7b-instruct"

THRESHOLD_LOW_FPR = 0.8536
THRESHOLD_ACCURACY = 0.9015

MIN_TOKENS = 50  # below this, results are unreliable


def get_device(requested: str = "auto") -> torch.device:
    """Select best available device for Mac Studio."""
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


class BinocularsDetector:
    """Loads two Falcon-7B models and computes Binoculars scores.

    Mac Studio version: uses MPS backend with float16, no quantization.
    Memory requirement: ~28-32 GB unified memory for two 7B models in fp16.
    """

    def __init__(self, device: torch.device | None = None) -> None:
        self.observer = None
        self.performer = None
        self.tokenizer = None
        self.device = device or get_device()

    def load_models(self) -> None:
        """Load both models. Call once at startup (~60-90s on Mac Studio)."""
        logger.info("Loading Binoculars models on %s...", self.device)

        if self.device.type == "cpu":
            logger.warning("Running on CPU — inference will be very slow")

        # No quantization on MPS; use float16 for memory efficiency
        # float16 is well-supported on MPS for inference
        dtype = torch.float16 if self.device.type == "mps" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(OBSERVER_MODEL)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Loading observer: %s (dtype=%s)", OBSERVER_MODEL, dtype)
        self.observer = AutoModelForCausalLM.from_pretrained(
            OBSERVER_MODEL,
            torch_dtype=dtype,
            device_map=None,  # manual placement for MPS
        ).to(self.device)
        self.observer.eval()

        logger.info("Loading performer: %s (dtype=%s)", PERFORMER_MODEL, dtype)
        self.performer = AutoModelForCausalLM.from_pretrained(
            PERFORMER_MODEL,
            torch_dtype=dtype,
            device_map=None,
        ).to(self.device)
        self.performer.eval()

        logger.info("Binoculars models loaded successfully on %s", self.device)

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
