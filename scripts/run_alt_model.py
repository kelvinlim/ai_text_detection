#!/usr/bin/env python3
"""Run Binoculars detection with alternative model pairs on all test samples.

Usage:
    python scripts/run_alt_model.py [--model-pair qwen2.5-7b] [--device auto|mps|cpu]

Compares results against the Falcon-7B baseline from RESULTS.md.
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_samples import ALL_SAMPLES

# Falcon-7B baseline scores from RESULTS.md for comparison
FALCON_BASELINE = {
    "human_specific_aims": 0.6458,
    "human_significance": 0.7876,
    "human_innovation": 0.8004,
    "human_approach": 0.6964,
    "human_preliminary_data": 0.6540,
    "ai_specific_aims": 0.5981,
    "ai_significance": 0.6316,
    "ai_innovation": 0.6593,
    "ai_approach": 0.5095,
    "ai_preliminary_data": 0.6237,
    "mixed_human_start_ai_end": 0.7409,
    "mixed_ai_rewrite_of_human": 0.6697,
    "mixed_human_with_ai_sentences": 0.6597,
    "edge_too_short": 0.5902,
    "edge_references": 0.6805,
    "edge_technical_jargon": 0.4420,
    "edge_non_english": 0.5523,
}


def main():
    from binoculars_local.detector_llama import MODEL_PAIRS

    parser = argparse.ArgumentParser(
        description="Score test samples with Binoculars (alternative model pairs)"
    )
    parser.add_argument(
        "--model-pair",
        default="qwen2.5-7b",
        choices=list(MODEL_PAIRS.keys()),
        help="Model pair to use (default: qwen2.5-7b)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "mps", "cpu"],
        help="Inference device (default: auto)",
    )
    args = parser.parse_args()

    # --- Load models ---
    import torch

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    pair_info = MODEL_PAIRS[args.model_pair]
    print(f"Model pair: {args.model_pair}")
    print(f"  Observer:  {pair_info['observer']}")
    print(f"  Performer: {pair_info['performer']}")
    print(f"Device: {device}")
    print("Loading models (this may take several minutes on first run)...")

    from binoculars_local.detector_llama import BinocularsAltDetector

    detector = BinocularsAltDetector(model_pair=args.model_pair, device=device)
    load_start = time.monotonic()
    detector.load_models()
    load_time = time.monotonic() - load_start
    print(f"Models loaded in {load_time:.1f}s\n")

    # --- Score all samples ---
    THRESHOLD_LOW_FPR = 0.8536
    THRESHOLD_ACCURACY = 0.9015

    results = []
    total_start = time.monotonic()

    print("=" * 100)
    print(
        f"{'ID':<35} {'Expected':>12} {'Score':>8} {'Falcon':>8} "
        f"{'Actual':>14} {'Match':>6} {'Time':>6}"
    )
    print("-" * 100)

    for sample in ALL_SAMPLES:
        text = sample["text"].strip()
        expected = sample["label"]

        chunk_start = time.monotonic()
        score, token_count = detector.compute_score(text)
        chunk_time = time.monotonic() - chunk_start

        # Classify using original thresholds
        if token_count < 50:
            actual = "uncertain"
        elif score >= THRESHOLD_ACCURACY:
            actual = "ai_generated"
        elif score < THRESHOLD_LOW_FPR:
            actual = "human"
        else:
            actual = "uncertain"

        # Check match
        if expected in ("mixed", "skip", "uncertain"):
            match = "~"
        elif expected == actual:
            match = "YES"
        else:
            match = "NO"

        falcon_score = FALCON_BASELINE.get(sample["id"], None)
        falcon_str = f"{falcon_score:.4f}" if falcon_score else "  N/A "

        results.append(
            {
                "id": sample["id"],
                "expected": expected,
                "actual": actual,
                "score": score,
                "falcon_score": falcon_score,
                "token_count": token_count,
                "time": chunk_time,
                "match": match,
            }
        )

        print(
            f"  {sample['id']:<35} {expected:>12} {score:>8.4f} {falcon_str:>8} "
            f"{actual:>14} {match:>6} {chunk_time:>5.1f}s"
        )

    total_time = time.monotonic() - total_start

    # --- Summary ---
    print("=" * 100)
    print(f"\nModel pair: {args.model_pair}")
    print(f"Total scoring time: {total_time:.1f}s")
    print(f"Average per sample: {total_time / len(ALL_SAMPLES):.1f}s")

    # Accuracy on labeled samples
    testable = [r for r in results if r["expected"] in ("human", "ai_generated")]
    correct = sum(1 for r in testable if r["match"] == "YES")
    if testable:
        print(
            f"\nAccuracy on labeled samples: {correct}/{len(testable)} "
            f"({100 * correct / len(testable):.0f}%)"
        )

    # Score distributions
    human_scores = [r["score"] for r in results if r["expected"] == "human"]
    ai_scores = [r["score"] for r in results if r["expected"] == "ai_generated"]

    if human_scores:
        print(
            f"\nHuman sample scores:  min={min(human_scores):.4f}  "
            f"max={max(human_scores):.4f}  mean={sum(human_scores)/len(human_scores):.4f}"
        )
    if ai_scores:
        print(
            f"AI sample scores:     min={min(ai_scores):.4f}  "
            f"max={max(ai_scores):.4f}  mean={sum(ai_scores)/len(ai_scores):.4f}"
        )

    # Separation analysis
    if human_scores and ai_scores:
        gap = min(ai_scores) - max(human_scores)
        if gap > 0:
            print(f"Score gap (separation): {gap:.4f} -- GOOD separation!")
        else:
            print(f"Score overlap: {abs(gap):.4f} -- thresholds may need calibration")

        # Check if direction is correct (AI should score HIGHER than human)
        mean_human = sum(human_scores) / len(human_scores)
        mean_ai = sum(ai_scores) / len(ai_scores)
        if mean_ai > mean_human:
            print(
                f"Direction: CORRECT (AI mean {mean_ai:.4f} > Human mean {mean_human:.4f})"
            )
        else:
            print(
                f"Direction: WRONG (AI mean {mean_ai:.4f} < Human mean {mean_human:.4f})"
            )

    # Comparison with Falcon baseline
    print("\n" + "=" * 60)
    print("COMPARISON WITH FALCON-7B BASELINE")
    print("=" * 60)

    falcon_human = [
        FALCON_BASELINE[s["id"]]
        for s in ALL_SAMPLES
        if s["label"] == "human" and s["id"] in FALCON_BASELINE
    ]
    falcon_ai = [
        FALCON_BASELINE[s["id"]]
        for s in ALL_SAMPLES
        if s["label"] == "ai_generated" and s["id"] in FALCON_BASELINE
    ]

    print(f"\n{'Metric':<35} {'Falcon-7B':>12} {args.model_pair:>12}")
    print("-" * 60)
    if human_scores and falcon_human:
        print(
            f"  Human mean score               "
            f"{sum(falcon_human)/len(falcon_human):>12.4f} "
            f"{sum(human_scores)/len(human_scores):>12.4f}"
        )
    if ai_scores and falcon_ai:
        print(
            f"  AI mean score                  "
            f"{sum(falcon_ai)/len(falcon_ai):>12.4f} "
            f"{sum(ai_scores)/len(ai_scores):>12.4f}"
        )
    if falcon_human and falcon_ai and human_scores and ai_scores:
        falcon_gap = sum(falcon_ai) / len(falcon_ai) - sum(falcon_human) / len(
            falcon_human
        )
        new_gap = sum(ai_scores) / len(ai_scores) - sum(human_scores) / len(
            human_scores
        )
        print(f"  AI-Human gap (mean)            {falcon_gap:>12.4f} {new_gap:>12.4f}")

        falcon_correct_human = sum(
            1 for s in falcon_human if s < THRESHOLD_LOW_FPR
        )
        falcon_correct_ai = sum(1 for s in falcon_ai if s >= THRESHOLD_ACCURACY)
        new_correct_human = sum(1 for s in human_scores if s < THRESHOLD_LOW_FPR)
        new_correct_ai = sum(1 for s in ai_scores if s >= THRESHOLD_ACCURACY)
        print(
            f"  Correct human classifications   "
            f"{falcon_correct_human}/{len(falcon_human):>10} "
            f"{new_correct_human}/{len(human_scores):>10}"
        )
        print(
            f"  Correct AI classifications      "
            f"{falcon_correct_ai}/{len(falcon_ai):>10} "
            f"{new_correct_ai}/{len(ai_scores):>10}"
        )

    print(f"\nModel load time: {load_time:.1f}s")
    print(f"Device: {device}")


if __name__ == "__main__":
    main()
