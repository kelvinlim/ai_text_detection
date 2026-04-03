# AI Text Detection

Zero-shot AI-generated text detection for NIH grant applications using **Binoculars** (ICML 2024).

## Hardware

- **Mac Studio** — Apple M2 Max, 64 GB unified memory
- **MacBook Pro** — Apple M4 Max, 36 GB unified memory
- Runs natively on Apple Silicon via MPS (Metal) backend in float16
- Scores are reproducible across both machines (differences ≤ 0.0001)

## Tech Stack

- **Python 3.10+**, PyTorch with MPS backend
- **Transformers** (HuggingFace) for model loading and inference
- **FastAPI** for the local detection server
- **Pydantic** for config and request/response schemas
- No CUDA, no BitsAndBytes — runs natively on Apple Silicon in float16

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

## Project Structure

```
ai_text_detection/
├── binoculars_local/           # Core detection library
│   ├── detector.py             # Falcon-7B Binoculars scorer (MPS)
│   ├── detector_llama.py       # Multi-model-pair scorer (Qwen, Llama, Mistral, Gemma)
│   ├── server.py               # FastAPI server (same API as grant_assist Cloud Run service)
│   └── config.py               # Environment-based configuration
├── tests/
│   ├── test_samples.py         # 17 original samples (5 human, 5 AI, 3 mixed, 4 edge)
│   ├── test_samples_expanded.py # 20 additional calibration samples
│   └── test_samples_generated.py # 6 Ollama-generated samples (Qwen3, Gemma3, Llama3.1)
├── scripts/
│   ├── run_test_samples.py     # Basic test runner
│   ├── run_generated_samples.py # Cross-architecture comparison
│   ├── run_alt_model.py        # Alternative model pair runner
│   ├── calibrate_thresholds.py # Statistical threshold calibration
│   ├── diagnose_scores.py      # PPL/X-PPL component decomposition
│   └── download_models.py      # HuggingFace model downloader
├── grant_assist/               # Symlink to the Grant Assist repo
├── RESULTS.md                  # Full experiment results and analysis
├── MacStudioPlan.md            # Implementation plan and architecture decisions
└── requirements.txt            # Python dependencies
```

## Common Tasks

- **Run detection on new text:** `python scripts/run_test_samples.py`
- **Test alternative model pair:** `python scripts/run_alt_model.py --model-pair qwen2.5-7b`
- **Recalibrate thresholds:** `python scripts/calibrate_thresholds.py`
- **Diagnose score components:** `python scripts/diagnose_scores.py`
- **Add test samples:** Edit `tests/test_samples.py` or `tests/test_samples_expanded.py`

## Grant Assist Integration

The local server (`binoculars_local/server.py`) exposes the same `POST /detect` API as the Cloud Run service in `grant_assist/binoculars-service/`. Point `ai_detection_url` in Grant Assist config to `http://localhost:8080` for local development.
