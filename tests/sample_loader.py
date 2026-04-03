"""YAML-based sample loader for AI text detection test samples.

Loads all .yaml sample files from tests/samples/ subdirectories and provides
backward-compatible constants and query functions matching the original
test_samples.py, test_samples_expanded.py, and test_samples_generated.py APIs.

Usage:
    from tests.sample_loader import ALL_SAMPLES, HUMAN_SAMPLES, AI_SAMPLES
    from tests.sample_loader import get_sample_by_id, get_samples_by_label
"""

from pathlib import Path

import yaml

_SAMPLES_DIR = Path(__file__).resolve().parent / "samples"

_REQUIRED_FIELDS = {"id", "label", "section", "text", "category"}


def load_sample(filepath: Path) -> dict:
    """Load a single YAML sample file and return it as a dict."""
    data = yaml.safe_load(filepath.read_text(encoding="utf-8"))
    missing = _REQUIRED_FIELDS - set(data.keys())
    if missing:
        raise ValueError(
            f"Sample {filepath.name} missing required fields: {missing}")
    data["text"] = data["text"].strip()
    if "prompt" in data:
        data["prompt"] = data["prompt"].strip()
    return data


def load_all() -> list[dict]:
    """Load all .yaml samples from all subdirectories of tests/samples/."""
    samples = []
    for yaml_file in sorted(_SAMPLES_DIR.rglob("*.yaml")):
        samples.append(load_sample(yaml_file))
    return samples


def _by_category(samples: list[dict], category: str) -> list[dict]:
    return [s for s in samples if s["category"] == category]


# ---------------------------------------------------------------------------
# Eagerly load all samples at import time
# ---------------------------------------------------------------------------
_ALL_LOADED = load_all()

# --- Original 17 samples (from test_samples.py) ---
HUMAN_SAMPLES = _by_category(_ALL_LOADED, "human")
AI_SAMPLES = _by_category(_ALL_LOADED, "ai")
MIXED_SAMPLES = _by_category(_ALL_LOADED, "mixed")
EDGE_CASES = _by_category(_ALL_LOADED, "edge_case")
ALL_SAMPLES = HUMAN_SAMPLES + AI_SAMPLES + MIXED_SAMPLES + EDGE_CASES

# --- Expanded 20 samples (from test_samples_expanded.py) ---
HUMAN_SAMPLES_EXPANDED = _by_category(_ALL_LOADED, "human_expanded")
AI_SAMPLES_EXPANDED = _by_category(_ALL_LOADED, "ai_expanded")
ALL_EXPANDED_SAMPLES = HUMAN_SAMPLES_EXPANDED + AI_SAMPLES_EXPANDED

# --- Ollama 6 samples (from test_samples_generated.py) ---
OLLAMA_SAMPLES = _by_category(_ALL_LOADED, "ollama")
ALL_OLLAMA_SAMPLES = OLLAMA_SAMPLES


# ---------------------------------------------------------------------------
# Query functions (backward-compatible)
# ---------------------------------------------------------------------------

def get_samples_by_label(label: str) -> list[dict]:
    """Return all original samples with a given label."""
    return [s for s in ALL_SAMPLES if s["label"] == label]


def get_sample_by_id(sample_id: str) -> dict | None:
    """Return a single sample by ID from any collection."""
    for s in _ALL_LOADED:
        if s["id"] == sample_id:
            return s
    return None


def get_expanded_samples_by_label(label: str) -> list[dict]:
    """Return expanded samples with a given label."""
    return [s for s in ALL_EXPANDED_SAMPLES if s["label"] == label]
