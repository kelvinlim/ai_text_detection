"""Test samples for Binoculars AI text detection.

Backward-compatible shim — samples are now stored as individual YAML files
in tests/samples/<section>/. This module re-exports from the YAML loader.

Categories:
1. HUMAN_SAMPLES — authentic academic/grant writing
2. AI_SAMPLES — LLM-generated grant text
3. MIXED_SAMPLES — human text with AI-edited portions
4. EDGE_CASES — short text, references, tables, non-English
"""

from tests.sample_loader import (
    HUMAN_SAMPLES,
    AI_SAMPLES,
    MIXED_SAMPLES,
    EDGE_CASES,
    ALL_SAMPLES,
    get_samples_by_label,
    get_sample_by_id,
)

__all__ = [
    "HUMAN_SAMPLES",
    "AI_SAMPLES",
    "MIXED_SAMPLES",
    "EDGE_CASES",
    "ALL_SAMPLES",
    "get_samples_by_label",
    "get_sample_by_id",
]


# Quick summary when run directly
if __name__ == "__main__":
    print(f"Total test samples: {len(ALL_SAMPLES)}")
    from collections import Counter
    counts = Counter(s["label"] for s in ALL_SAMPLES)
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")
    print()
    for sample in ALL_SAMPLES:
        words = len(sample["text"].split())
        print(f"  [{sample['label']:>12}] {sample['id']:<35} ({words} words)")
