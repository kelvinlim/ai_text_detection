#!/usr/bin/env python3
"""Run Binoculars detection on all test samples and report results.

Usage:
    python scripts/run_test_samples.py [--device auto|mps|cpu] [--threshold-mode low_fpr|accuracy]

This script loads the Binoculars models, scores every sample in test_samples.py,
and prints a comparison table of expected vs actual labels.
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_samples import ALL_SAMPLES, get_samples_by_label


def main():
    parser = argparse.ArgumentParser(description="Score test samples with Binoculars")
    parser.add_argument(
        "--device", default="auto", choices=["auto", "mps", "cpu"],
        help="Inference device (default: auto → MPS if available, else CPU)",
    )
    parser.add_argument(
        "--threshold-mode", default="low_fpr", choices=["low_fpr", "accuracy"],
        help="Classification threshold mode (default: low_fpr)",
    )
    parser.add_argument(
        "--skip-load", action="store_true",
        help="Dry run — print samples without loading models",
    )
    args = parser.parse_args()

    # --- Dry run mode ---
    if args.skip_load:
        print("=" * 80)
        print("TEST SAMPLES (dry run — models not loaded)")
        print("=" * 80)
        for sample in ALL_SAMPLES:
            words = len(sample["text"].split())
            print(f"  [{sample['label']:>12}] {sample['id']:<35} ({words} words)")
        print(f"\nTotal: {len(ALL_SAMPLES)} samples")
        return

    # --- Load models ---
    import torch

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    print("Loading Binoculars models (this takes 1-2 minutes)...")

    from binoculars_local.detector import BinocularsDetector

    detector = BinocularsDetector(device=device)
    load_start = time.monotonic()
    detector.load_models()
    load_time = time.monotonic() - load_start
    print(f"Models loaded in {load_time:.1f}s\n")

    # --- Score all samples ---
    THRESHOLD_LOW_FPR = 0.8536
    THRESHOLD_ACCURACY = 0.9015

    results = []
    total_start = time.monotonic()

    print("=" * 90)
    print(f"{'ID':<35} {'Expected':>12} {'Score':>8} {'Actual':>14} {'Match':>6} {'Time':>6}")
    print("-" * 90)

    for sample in ALL_SAMPLES:
        text = sample["text"].strip()
        expected = sample["label"]

        chunk_start = time.monotonic()
        score, token_count = detector.compute_score(text)
        chunk_time = time.monotonic() - chunk_start

        # Classify
        if token_count < 50:
            actual = "uncertain"
        elif score >= THRESHOLD_ACCURACY:
            actual = "ai_generated"
        elif score < THRESHOLD_LOW_FPR:
            actual = "human"
        else:
            actual = "uncertain"

        # Check match (mixed and uncertain are always "partial match")
        if expected in ("mixed", "skip", "uncertain"):
            match = "~"
        elif expected == actual:
            match = "YES"
        else:
            match = "NO"

        results.append({
            "id": sample["id"],
            "expected": expected,
            "actual": actual,
            "score": score,
            "token_count": token_count,
            "time": chunk_time,
            "match": match,
        })

        print(
            f"  {sample['id']:<35} {expected:>12} {score:>8.4f} {actual:>14} "
            f"{match:>6} {chunk_time:>5.1f}s"
        )

    total_time = time.monotonic() - total_start

    # --- Summary ---
    print("=" * 90)
    print(f"\nTotal scoring time: {total_time:.1f}s")
    print(f"Average per sample: {total_time / len(ALL_SAMPLES):.1f}s")

    # Accuracy on labeled samples (excluding mixed/skip/uncertain expected)
    testable = [r for r in results if r["expected"] in ("human", "ai_generated")]
    correct = sum(1 for r in testable if r["match"] == "YES")
    if testable:
        print(f"\nAccuracy on labeled samples: {correct}/{len(testable)} "
              f"({100 * correct / len(testable):.0f}%)")

    # Score distributions
    human_scores = [r["score"] for r in results if r["expected"] == "human"]
    ai_scores = [r["score"] for r in results if r["expected"] == "ai_generated"]

    if human_scores:
        print(f"\nHuman sample scores:  min={min(human_scores):.4f}  "
              f"max={max(human_scores):.4f}  mean={sum(human_scores)/len(human_scores):.4f}")
    if ai_scores:
        print(f"AI sample scores:     min={min(ai_scores):.4f}  "
              f"max={max(ai_scores):.4f}  mean={sum(ai_scores)/len(ai_scores):.4f}")

    # Separation
    if human_scores and ai_scores:
        gap = min(ai_scores) - max(human_scores)
        if gap > 0:
            print(f"Score gap (separation): {gap:.4f} — good separation")
        else:
            print(f"Score overlap: {abs(gap):.4f} — thresholds may need calibration")

    print(f"\nModel load time: {load_time:.1f}s")
    print(f"Device: {device}")


if __name__ == "__main__":
    main()
