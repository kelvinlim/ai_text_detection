# Binoculars AI Detection — Mac Studio Implementation Plan

## Critical Issue: Ollama Cannot Run Binoculars

Binoculars requires **full logit distributions** (softmax over the entire vocabulary, ~65K tokens) from both models at every token position. Ollama's API only returns **top-20 logprobs** — not the full distribution. This makes standard Binoculars scoring impossible through Ollama alone.

| Requirement | Ollama Provides | Needed |
|---|---|---|
| Full vocab logits per token | Top-20 only | All ~65K |
| Input token logprobs | No | Yes |
| Cross-perplexity calculation | Impossible | Required |

**This is a hard constraint, not a configuration issue.** The cross-perplexity formula requires summing over the full vocabulary:

```
x_ppl = -Σ_v P_performer(v) · log P_observer(v)   # sum over ALL vocab tokens
```

With only top-20, ~99.97% of the distribution is missing.

---

## Recommended Approach: Three Options for Mac Studio

### Option A: `transformers` + MPS Backend (Recommended — Easiest Port)

**Why:** The existing `detector.py` code works almost unchanged. Just swap CUDA → MPS and remove BitsAndBytes (poorly supported on macOS). Mac Studio has enough unified memory for two 7B models in float16 (~28GB).

**Pros:**
- Minimal code changes from existing `binoculars-service/app/detector.py`
- Full logit access guaranteed
- Well-tested library, same as the Binoculars paper used
- MPS acceleration on Apple Silicon

**Cons:**
- Larger memory footprint than quantized (28GB fp16 vs 8GB 4-bit)
- Slower than MLX on Apple Silicon
- Model download from HuggingFace (~14GB per model)

**Mac Studio Requirements:**
- M2 Max (32GB+), M2 Ultra, M4 Max, or M4 Ultra
- 64GB+ unified memory recommended (comfortable headroom)
- 32GB works but tight with two fp16 7B models

### Option B: MLX (Best Performance on Apple Silicon)

**Why:** Apple's MLX framework is purpose-built for Apple Silicon. Native Metal acceleration, efficient memory usage, full logit access.

**Pros:**
- Fastest inference on Mac Studio
- Memory-efficient (unified memory architecture aware)
- Full logit access
- Growing model ecosystem

**Cons:**
- Requires model conversion to MLX format (or find pre-converted)
- Falcon-7B MLX availability uncertain — may need to convert or use alternative model pair
- Newer ecosystem, less battle-tested

### Option C: `llama-cpp-python` + Metal (GGUF Models)

**Why:** Uses GGUF quantized models (same format Ollama uses internally). The `logits_all=True` parameter returns full vocabulary logits. Metal acceleration on Mac.

**Pros:**
- Efficient quantized inference (smaller memory footprint)
- `logits_all=True` gives full logit distributions
- Can use Ollama-downloaded model files directly
- Good Metal acceleration

**Cons:**
- More complex logit extraction code
- Need to manage two model instances manually
- Falcon-7B GGUF availability for the original (non-H1R) variant is limited
- May need to convert models to GGUF yourself

---

## Chosen Path: Option A (`transformers` + MPS)

Option A is recommended because:
1. **Smallest diff** from the existing working code in `binoculars-service/`
2. **Guaranteed correctness** — same library the Binoculars paper used
3. **Mac Studio M2 Ultra (64GB+) or M4 has plenty of memory** for fp16
4. **MPS backend is mature** in PyTorch 2.x

If performance becomes an issue later, Option B (MLX) is the upgrade path.

---

## Implementation Plan

### Phase 1: Mac Studio Binoculars Service (~1-2 days)

Adapt the existing `binoculars-service/` to run on Mac Studio without GPU containers.

#### 1.1 Modified `detector.py` — Key Changes

```python
# Changes from the Cloud Run version:
# 1. CUDA → MPS device selection
# 2. Remove BitsAndBytes 4-bit quantization (not supported on MPS)
# 3. Use float16 on MPS (fits in Mac Studio unified memory)
# 4. Add CPU fallback for compatibility

import torch

def _get_device() -> torch.device:
    """Select best available device on Mac Studio."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# Model loading — no quantization, use float16
load_kwargs = {
    "torch_dtype": torch.float16,
    "device_map": None,  # manual device placement for MPS
}

observer = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b", **load_kwargs)
observer = observer.to(device)  # move to MPS
observer.eval()

performer = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct", **load_kwargs)
performer = performer.to(device)
performer.eval()
```

#### 1.2 Memory Budget (Mac Studio)

| Component | Memory (fp16) | Memory (fp32) |
|---|---|---|
| Falcon-7B observer | ~14 GB | ~28 GB |
| Falcon-7B-Instruct performer | ~14 GB | ~28 GB |
| **Total (fp16)** | **~28 GB** | — |
| Inference overhead (activations, KV cache) | ~2-4 GB | — |
| **Working total** | **~30-32 GB** | — |

- **32 GB Mac Studio:** Tight but feasible in fp16. Close other apps.
- **64 GB Mac Studio:** Comfortable. Recommended.
- **96-192 GB Mac Studio:** No constraints whatsoever.

#### 1.3 Project Structure

```
ai_text_detection/
├── MacStudioPlan.md          ← this file
├── binoculars_local/
│   ├── __init__.py
│   ├── detector.py           # MPS-adapted Binoculars scorer
│   ├── chunker.py            # Reuse from grant_assist (copy or symlink)
│   ├── models.py             # Pydantic schemas
│   ├── server.py             # FastAPI app for local serving
│   └── config.py             # Local configuration
├── tests/
│   ├── __init__.py
│   ├── test_samples.py       # Test sample data
│   ├── test_detector.py      # Unit tests for scoring logic
│   ├── test_chunker.py       # Unit tests for chunking
│   └── test_integration.py   # End-to-end API tests
├── scripts/
│   ├── download_models.py    # Pre-download models from HuggingFace
│   ├── run_server.sh         # Start local FastAPI server
│   └── run_test_samples.py   # Score test samples and print results
├── requirements.txt          # Python dependencies
└── README.md
```

#### 1.4 Dependencies (`requirements.txt`)

```
torch>=2.2.0
transformers>=4.36.0
accelerate>=0.25.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
pytest>=8.0.0
httpx>=0.27.0
```

Note: `bitsandbytes` is **removed** (not needed without 4-bit quantization on MPS).

#### 1.5 Model Download Script

```python
# scripts/download_models.py
"""Pre-download Falcon-7B models to local HuggingFace cache."""
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = ["tiiuae/falcon-7b", "tiiuae/falcon-7b-instruct"]

for model_name in MODELS:
    print(f"Downloading {model_name}...")
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="float16")
    print(f"  ✓ {model_name} cached")
```

Run once: `python scripts/download_models.py` (~28 GB download, cached in `~/.cache/huggingface/`)

#### 1.6 Configuration

```python
# binoculars_local/config.py
from pydantic_settings import BaseSettings

class BinocularsConfig(BaseSettings):
    model_config = {"env_prefix": "BINOCULARS_"}

    observer_model: str = "tiiuae/falcon-7b"
    performer_model: str = "tiiuae/falcon-7b-instruct"
    device: str = "auto"              # "auto", "mps", "cpu"
    dtype: str = "float16"            # "float16" or "float32"
    threshold_mode: str = "low_fpr"   # "accuracy" or "low_fpr"
    host: str = "0.0.0.0"
    port: int = 8080
    max_chunks_per_request: int = 100
```

### Phase 2: Ollama as Supplementary Tool (Optional)

While Ollama can't run Binoculars directly, it can still be useful:

1. **Model exploration** — Use Ollama to quickly test different model pairs before committing to HuggingFace downloads
2. **Baseline comparison** — Use Ollama's top-K logprobs as a fast (but less accurate) pre-filter
3. **Other AI detection approaches** — Prompt-based detection ("analyze this text for AI patterns") as a complementary signal

```bash
# Install Ollama on Mac Studio
brew install ollama

# Pull Falcon models (for exploration, not for Binoculars scoring)
ollama pull falcon:7b
```

### Phase 3: Alternative Model Pairs to Explore

The original paper uses Falcon-7B, but other pairs may work better:

| Observer (Base) | Performer (Instruct) | Tokenizer Match | Notes |
|---|---|---|---|
| `tiiuae/falcon-7b` | `tiiuae/falcon-7b-instruct` | ✓ Same | Paper default |
| `meta-llama/Llama-3.1-8B` | `meta-llama/Llama-3.1-8B-Instruct` | ✓ Same | Newer, potentially better on modern AI text |
| `mistralai/Mistral-7B-v0.3` | `mistralai/Mistral-7B-Instruct-v0.3` | ✓ Same | Strong base model |
| `google/gemma-2-9b` | `google/gemma-2-9b-it` | ✓ Same | Google's pair |

**Recommendation:** Start with Falcon-7B (validated by the paper), then test Llama-3.1-8B pair, which may detect newer AI models better since it's a more recent architecture.

### Phase 4: Testing & Calibration (~1-2 days)

See `tests/test_samples.py` for the test sample data.

#### 4.1 Test Matrix

| Test Category | # Samples | Expected Outcome |
|---|---|---|
| Human-written grant sections | 5 | score < 0.85, label = "human" |
| AI-generated grant sections | 5 | score > 0.90, label = "ai_generated" |
| AI-edited human text (mixed) | 3 | mixed scores per chunk |
| Edge cases (short, references) | 4 | uncertain / skipped |
| **Total** | **17** | |

#### 4.2 Calibration Workflow

1. Run all test samples through the detector
2. Record scores and compare against expected labels
3. Plot score distributions for human vs AI samples
4. If the default thresholds don't separate well on grant text, compute new thresholds:
   - Find the score that gives ≤1% FPR on the human samples
   - Find the score that gives ≥90% TPR on the AI samples
5. Update `THRESHOLD_LOW_FPR` and `THRESHOLD_ACCURACY` if needed

#### 4.3 Performance Benchmarks to Collect

| Metric | Target |
|---|---|
| Model load time (cold start) | < 60s |
| Score per chunk (512 tokens) | < 2s on MPS |
| Full document (50 chunks) | < 90s |
| Memory usage (steady state) | < 35 GB |

### Phase 5: Integration with Grant Assist

Once the local service is validated:

1. Update `grant_assist/backend/app/config.py` to point `ai_detection_url` at `http://localhost:8080`
2. The API contract (`POST /detect`) remains identical
3. No frontend changes needed — the response schema is the same
4. For development: run `binoculars_local/server.py` on the Mac Studio
5. For production: deploy the Cloud Run version (Phase 1 of `AIDetection.md`)

---

## Quick Start

```bash
# 1. Install dependencies
cd /Users/kolim/Projects/ai_text_detection
pip install -r requirements.txt

# 2. Download models (~28 GB, one-time)
python scripts/download_models.py

# 3. Run test samples (no server needed)
python scripts/run_test_samples.py

# 4. Start local server (optional, for Grant Assist integration)
python -m binoculars_local.server
# → http://localhost:8080/docs
```

---

## Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| 32GB Mac Studio runs out of memory | Service crashes | Use `float32` CPU fallback or upgrade to 64GB |
| MPS backend bugs with Falcon-7B | Incorrect scores | Fall back to CPU; file PyTorch issue |
| Falcon-7B thresholds wrong for grant text | Poor accuracy | Calibrate on test samples (Phase 4) |
| Model download blocked by HuggingFace auth | Can't start | Accept Falcon-7B license on HuggingFace, use `huggingface-cli login` |
| Newer AI text (GPT-4o, Claude 4) evades detection | False negatives | Test with Llama-3.1 model pair; add ensemble approach |
