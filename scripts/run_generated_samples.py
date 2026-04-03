#!/usr/bin/env python3
"""Score Ollama-generated samples alongside original samples.

Compares Binoculars scores across different AI text sources:
- Human-written (original test_samples.py)
- Claude-generated (original test_samples.py)
- Ollama-generated (test_samples_generated.py — Qwen3, Gemma3, Llama3.1)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_samples import HUMAN_SAMPLES, AI_SAMPLES
from tests.test_samples_generated import ALL_OLLAMA_SAMPLES


def main():
    import torch
    from binoculars_local.detector import BinocularsDetector

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading Binoculars models...")

    detector = BinocularsDetector(device=device)
    detector.load_models()
    print("Models loaded.\n")

    # Score all samples
    groups = {
        "Human (original)": HUMAN_SAMPLES,
        "Claude AI (original)": AI_SAMPLES,
        "Ollama AI (generated)": ALL_OLLAMA_SAMPLES,
    }

    all_results = {}

    print("=" * 85)
    print(f"{'ID':<42} {'Source':>14} {'Score':>8} {'Tokens':>7}")
    print("-" * 85)

    for group_name, samples in groups.items():
        scores = []
        for sample in samples:
            start = time.monotonic()
            score, token_count = detector.compute_score(sample["text"])
            elapsed = time.monotonic() - start

            source = sample.get("source", group_name.split()[0].lower())
            scores.append(score)

            print(f"  {sample['id']:<42} {source:>14} {score:>8.4f} {token_count:>7}")

        all_results[group_name] = scores
        print()

    # Summary
    print("=" * 85)
    print("\nSCORE DISTRIBUTIONS BY SOURCE")
    print("-" * 50)

    for group_name, scores in all_results.items():
        mean = sum(scores) / len(scores)
        mn, mx = min(scores), max(scores)
        print(f"  {group_name:<25} n={len(scores):>2}  "
              f"mean={mean:.4f}  min={mn:.4f}  max={mx:.4f}")

    # Key comparison
    human_mean = sum(all_results["Human (original)"]) / len(all_results["Human (original)"])
    claude_mean = sum(all_results["Claude AI (original)"]) / len(all_results["Claude AI (original)"])
    ollama_mean = sum(all_results["Ollama AI (generated)"]) / len(all_results["Ollama AI (generated)"])

    print(f"\n  Human mean:  {human_mean:.4f}")
    print(f"  Claude mean: {claude_mean:.4f}  (diff from human: {claude_mean - human_mean:+.4f})")
    print(f"  Ollama mean: {ollama_mean:.4f}  (diff from human: {ollama_mean - human_mean:+.4f})")

    print("\n  Expected: AI scores should be HIGHER than human scores")
    if ollama_mean > human_mean:
        print("  Result: Ollama AI scores ARE higher — detection may work")
    else:
        print("  Result: Ollama AI scores are NOT higher — same problem as Claude")

    if claude_mean > human_mean:
        print("  Result: Claude AI scores ARE higher")
    else:
        print("  Result: Claude AI scores are NOT higher")


if __name__ == "__main__":
    main()
