# AI Text Detection Project

## Overview

This project evaluates **Binoculars** (ICML 2024) as a zero-shot detector for AI-generated text in NIH grant applications. It runs locally on Mac Studio (M2 Max, 64 GB) and MacBook Pro (M4 Max, 36 GB) using Apple MPS (Metal) acceleration. Scores are reproducible across both machines using Apple MPS (Metal) acceleration.

## Tech Stack

- **Python 3.10+**, PyTorch with MPS backend
- **Transformers** (HuggingFace) for model loading and inference
- **FastAPI** for the local detection server
- **Pydantic** for config and request/response schemas
- No CUDA, no BitsAndBytes — runs natively on Apple Silicon in float16

## Project Structure

```
ai_text_detection/
├── binoculars_local/           # Core detection library
│   ├── detector.py             # Falcon-7B Binoculars scorer (MPS)
│   ├── detector_llama.py       # Multi-model-pair scorer (Qwen, Llama, Mistral, Gemma)
│   ├── server.py               # FastAPI server (same API as grant_assist Cloud Run service)
│   └── config.py               # Environment-based configuration
├── tests/
│   ├── samples/                # YAML sample files organized by section
│   │   ├── specific_aims/      # Specific Aims samples (human, AI, multi-model)
│   │   ├── significance/       # Significance samples
│   │   ├── innovation/         # Innovation samples
│   │   ├── approach/           # Approach samples
│   │   ├── preliminary_data/   # Preliminary Data samples
│   │   ├── background/         # Background samples
│   │   ├── rigor/              # Rigor & Reproducibility samples
│   │   ├── environment/        # Environment samples
│   │   ├── budget/             # Budget Justification samples
│   │   ├── biosketch/          # Biosketch Narrative samples
│   │   ├── mixed/              # Mixed human+AI samples
│   │   └── edge_cases/         # Edge case samples
│   ├── sample_loader.py        # YAML sample loader (provides backward-compatible API)
│   ├── test_samples.py         # Shim: re-exports 17 original samples from loader
│   ├── test_samples_expanded.py # Shim: re-exports 20 expanded samples from loader
│   └── test_samples_generated.py # Shim: re-exports 6 Ollama samples from loader
├── scripts/
│   ├── run_test_samples.py     # Basic test runner
│   ├── run_generated_samples.py # Cross-architecture comparison
│   ├── run_alt_model.py        # Alternative model pair runner
│   ├── calibrate_thresholds.py # Statistical threshold calibration
│   ├── diagnose_scores.py      # PPL/X-PPL component decomposition
│   ├── download_models.py      # HuggingFace model downloader
│   ├── add_sample.py           # Helper to create new YAML sample files
│   └── migrate_samples_to_yaml.py # One-time migration (already run)
├── grant_assist/               # Symlink to the Grant Assist repo
├── RESULTS.md                  # Full experiment results and analysis
├── MacStudioPlan.md            # Implementation plan and architecture decisions
└── requirements.txt            # Python dependencies
```

## Key Models

- **Falcon-7B** pair: `tiiuae/falcon-7b` + `tiiuae/falcon-7b-instruct` (~28 GB in fp16)
- **Qwen2.5-7B** pair: `Qwen/Qwen2.5-7B` + `Qwen/Qwen2.5-7B-Instruct` (~30 GB in fp16)
- Models cached in `~/.cache/huggingface/hub/`

## Quick Start

```bash
pip install -r requirements.txt
python scripts/download_models.py        # ~28 GB one-time download
python scripts/run_test_samples.py       # score all samples
python -m binoculars_local.server        # start local API at :8080
```

## Critical Finding: Inverted Thresholds

On grant/academic text, Binoculars scores are **inverted** from the paper's predictions:
- AI text scores **lower** (mean 0.63) than human text (mean 0.74)
- Published thresholds (0.85/0.90) detect **nothing** — all scores fall below them
- With calibrated inverted thresholds (~0.68), accuracy reaches **87%**

See `RESULTS.md` for full analysis and `scripts/diagnose_scores.py` for the mathematical explanation.

## Grant Assist Integration

The local server (`binoculars_local/server.py`) exposes the same `POST /detect` API as the Cloud Run service in `grant_assist/binoculars-service/`. Point `ai_detection_url` in Grant Assist config to `http://localhost:8080` for local development.

## Common Tasks

- **Run detection on new text:** `python scripts/run_test_samples.py`
- **Test alternative model pair:** `python scripts/run_alt_model.py --model-pair qwen2.5-7b`
- **Recalibrate thresholds:** `python scripts/calibrate_thresholds.py`
- **Diagnose score components:** `python scripts/diagnose_scores.py`
- **Add test sample from chatbot output:** `python scripts/add_sample.py --id <id> --section "Specific Aims" --source <model> --prompt "..." --text-file output.txt`
- **Add test sample interactively:** `python scripts/add_sample.py --interactive`
- **Add test sample from clipboard (macOS):** `pbpaste | python scripts/add_sample.py --id <id> --section "..." --source <model> --text-file -`
