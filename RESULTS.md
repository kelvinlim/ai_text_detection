# Binoculars Test Results — Mac Studio (MPS)

## Run Date: 2026-04-02

## Environment
- Device: Apple MPS (Mac Studio)
- PyTorch with Metal backend, float16
- Average scoring time: 2.5s per chunk (Falcon-7B), 0.9s (Qwen2.5-7B)

---

## Experiment 1: Falcon-7B on Original Samples (17 samples)

Models: `tiiuae/falcon-7b` (observer) + `tiiuae/falcon-7b-instruct` (performer)

| Sample ID | Expected | Score | Classified As | Match |
|---|---|---|---|---|
| human_specific_aims | human | 0.6458 | human | YES |
| human_significance | human | 0.7876 | human | YES |
| human_innovation | human | 0.8004 | human | YES |
| human_approach | human | 0.6964 | human | YES |
| human_preliminary_data | human | 0.6540 | human | YES |
| ai_specific_aims | ai_generated | 0.5981 | human | NO |
| ai_significance | ai_generated | 0.6316 | human | NO |
| ai_innovation | ai_generated | 0.6593 | human | NO |
| ai_approach | ai_generated | 0.5095 | human | NO |
| ai_preliminary_data | ai_generated | 0.6237 | human | NO |
| mixed_human_start_ai_end | mixed | 0.7409 | — | — |
| mixed_ai_rewrite_of_human | mixed | 0.6697 | — | — |
| mixed_human_with_ai_sentences | mixed | 0.6597 | — | — |
| edge_too_short | uncertain | 0.5902 | uncertain | — |
| edge_references | skip | 0.6805 | — | — |
| edge_technical_jargon | human | 0.4420 | human | YES |
| edge_non_english | uncertain | 0.5523 | — | — |

**Accuracy with published thresholds: 6/11 (55%)** — all AI samples misclassified.

---

## Experiment 2: Cross-Architecture AI Samples via Ollama

Tested whether Falcon-7B detects AI text from different model families.
Text generated locally via Ollama from Qwen3:8B, Gemma3:12B, and Llama3.1:8B.

| Source | n | Mean Score | Min | Max |
|---|---|---|---|---|
| Human (original) | 5 | 0.7168 | 0.6458 | 0.8004 |
| Ollama AI (mixed models) | 6 | 0.6290 | 0.5649 | 0.6870 |
| Claude AI (original) | 5 | 0.6044 | 0.5095 | 0.6593 |

**Conclusion:** AI text from ALL model architectures scores lower than human text.
The inverted direction is not Claude-specific — it's a fundamental property of
how Falcon-7B processes formulaic academic writing.

---

## Experiment 3: Qwen2.5-7B Alternative Model Pair

Models: `Qwen/Qwen2.5-7B` (observer) + `Qwen/Qwen2.5-7B-Instruct` (performer)

| Metric | Falcon-7B | Qwen2.5-7B |
|---|---|---|
| Human mean | 0.6710 | **1.0429** |
| AI mean | 0.6044 | **0.8890** |
| AI - Human gap | -0.0666 | **-0.1540** |
| Direction | WRONG | WRONG |
| Accuracy | 55% | **9%** |

Scores shifted dramatically higher but direction stayed wrong. Human text still
scores higher than AI text. Qwen2.5-7B made the problem **worse**, not better.

---

## Experiment 4: Threshold Calibration (30 samples)

Expanded to 30 labeled samples (15 human, 15 AI) using Falcon-7B.

### Score Distributions

| Metric | Human (n=15) | AI (n=15) |
|---|---|---|
| Mean | **0.7385** | **0.6274** |
| Std | 0.0518 | 0.0476 |
| Median | 0.7431 | 0.6316 |
| Min | 0.6458 | 0.5095 |
| Max | 0.8224 | 0.7099 |

### Statistical Tests

| Test | Statistic | p-value | Interpretation |
|---|---|---|---|
| Mann-Whitney U | 208.0 | **0.00008** | Significant separation |
| Welch's t-test | 5.91 | **0.000002** | Significant separation |
| Cohen's d | 2.23 | — | Large effect size |

**The distributions ARE statistically separable**, but with inverted direction.

### Calibrated Grant-Specific Thresholds

Classification is **inverted**: score < threshold → AI, score > threshold → human.

| Method | Threshold | Sensitivity | Specificity | FPR | Accuracy |
|---|---|---|---|---|---|
| Youden's J (optimal) | **0.6845** | 93.3% | 80.0% | 20.0% | **86.7%** |
| Conservative (≤1% FPR) | **0.6387** | 60.0% | 100% | 0% | 80.0% |

Published thresholds (0.8536 / 0.9015): **0% detection rate** on grant text.

---

## Key Findings

### 1. Binoculars CAN detect AI-generated grant text
The score separation is statistically significant (p < 0.0001, Cohen's d = 2.23).
With calibrated thresholds, 87% accuracy is achievable.

### 2. The score direction is INVERTED from the paper
AI text scores **lower** than human text on grant writing. The paper predicts the
opposite. This likely occurs because:
- Grant writing is inherently formulaic — both models agree on it
- AI-generated grant text is even MORE predictable/formulaic than human grant text
- The base model (observer) finds AI text easier to predict, lowering PPL
- Lower PPL → lower score (PPL / X-PPL)

### 3. Published thresholds are useless for this domain
All 37 samples scored below 0.85. The published thresholds detect 0% of AI text.

### 4. The problem is architecture-independent
AI text from Claude, Qwen3, Gemma3, and Llama3.1 all score lower than human text.
Switching the detector model pair to Qwen2.5-7B made it worse, not better.

### 5. There IS an overlap zone
Scores between 0.645 and 0.710 contain both human and AI samples. Samples in this
range should be classified as "uncertain."

---

## Recommendations

### For Grant Assist Integration

1. **Use inverted Falcon-7B thresholds:**
   - score < 0.64 → `ai_generated` (high confidence, ~0% FPR)
   - 0.64 ≤ score < 0.69 → `uncertain` (overlap zone)
   - score ≥ 0.69 → `human` (high confidence)

2. **Validate on real grant data** before production use. The 30-sample calibration
   is a starting point, not a production-ready threshold.

3. **Consider ensemble approach:** Combine Binoculars score with stylometric features
   (sentence length variance, vocabulary richness, burstiness) for higher accuracy.

4. **Present results with appropriate uncertainty.** Label as "AI writing indicators"
   not "AI detection" — false positives on human text are damaging in grant review.

### For Further Research

5. **Collect real-world samples:** Score known human grants + AI-rewritten versions
   of the same grants to test on authentic data.

6. **Test domain-specific fine-tuned model pairs:** A model pair fine-tuned on
   biomedical text might produce better separation.

7. **Investigate why direction inverts:** Run on non-academic text (news, fiction,
   social media) to confirm whether inversion is domain-specific.

---

## Files Created

| File | Purpose |
|---|---|
| `binoculars_local/detector.py` | Falcon-7B detector (MPS) |
| `binoculars_local/detector_llama.py` | Multi-model-pair detector (Qwen, Llama, Mistral, Gemma) |
| `binoculars_local/server.py` | Local FastAPI server |
| `binoculars_local/config.py` | Configuration |
| `tests/test_samples.py` | 17 original test samples |
| `tests/test_samples_expanded.py` | 20 additional calibration samples |
| `tests/test_samples_generated.py` | 6 Ollama-generated samples |
| `scripts/run_test_samples.py` | Basic test runner |
| `scripts/run_generated_samples.py` | Cross-architecture comparison |
| `scripts/run_alt_model.py` | Alternative model pair runner |
| `scripts/calibrate_thresholds.py` | Statistical calibration |
| `scripts/download_models.py` | Model download utility |
