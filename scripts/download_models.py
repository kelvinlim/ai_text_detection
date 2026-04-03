#!/usr/bin/env python3
"""Pre-download Falcon-7B models to local HuggingFace cache.

Run once before using the detector:
    python scripts/download_models.py

Downloads ~28 GB total to ~/.cache/huggingface/hub/
"""

import sys
import time


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    models = [
        ("tiiuae/falcon-7b", "observer"),
        ("tiiuae/falcon-7b-instruct", "performer"),
    ]

    for model_name, role in models:
        print(f"\nDownloading {role}: {model_name}...")
        start = time.monotonic()

        print(f"  Tokenizer...", end=" ", flush=True)
        AutoTokenizer.from_pretrained(model_name)
        print("done")

        print(f"  Model (float16)...", end=" ", flush=True)
        AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="float16"
        )
        elapsed = time.monotonic() - start
        print(f"done ({elapsed:.0f}s)")

    print("\nAll models cached. Ready to run.")


if __name__ == "__main__":
    main()
