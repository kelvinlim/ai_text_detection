"""Expanded test samples for Binoculars threshold calibration.

Backward-compatible shim — samples are now stored as individual YAML files
in tests/samples/<section>/. This module re-exports from the YAML loader.
"""

from tests.sample_loader import (
    HUMAN_SAMPLES_EXPANDED,
    AI_SAMPLES_EXPANDED,
    ALL_EXPANDED_SAMPLES,
    get_expanded_samples_by_label,
)

__all__ = [
    "HUMAN_SAMPLES_EXPANDED",
    "AI_SAMPLES_EXPANDED",
    "ALL_EXPANDED_SAMPLES",
    "get_expanded_samples_by_label",
]


if __name__ == "__main__":
    print(f"Expanded test samples: {len(ALL_EXPANDED_SAMPLES)}")
    from collections import Counter
    counts = Counter(s["label"] for s in ALL_EXPANDED_SAMPLES)
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")
    print()
    for sample in ALL_EXPANDED_SAMPLES:
        words = len(sample["text"].split())
        print(f"  [{sample['label']:>12}] {sample['id']:<35} ({words} words)")
