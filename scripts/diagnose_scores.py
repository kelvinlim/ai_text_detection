#!/usr/bin/env python3
"""Diagnose WHY Binoculars scores invert on grant text.

Decomposes the score into its two components:
  score = PPL / X-PPL

- PPL (perplexity): How surprised is the observer by the actual tokens?
- X-PPL (cross-perplexity): How much do observer and performer agree?

The paper assumes: AI text → PPL ≈ X-PPL → score ≈ 1.0
We need to understand why grant text violates this.
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_samples import HUMAN_SAMPLES, AI_SAMPLES
from binoculars_local.detector import BinocularsDetector


class DiagnosticDetector(BinocularsDetector):
    """Extended detector that returns PPL and X-PPL separately."""

    def compute_score_diagnostic(self, text: str) -> dict:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=512, padding=False,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        token_count = input_ids.shape[1]

        with torch.no_grad():
            logits_observer = self.observer(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
            logits_performer = self.performer(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits

        shift_obs = logits_observer[:, :-1].contiguous()
        shift_perf = logits_performer[:, :-1].contiguous()
        labels = input_ids[:, 1:].contiguous()

        # PPL: how surprised is the observer by the actual next tokens?
        ppl = F.cross_entropy(
            shift_obs.float().view(-1, shift_obs.size(-1)),
            labels.view(-1),
            reduction="mean",
        ).item()

        # X-PPL: expected cross-entropy between performer's distribution
        # and observer's distribution (how much do they agree?)
        performer_probs = F.softmax(shift_perf.float(), dim=-1)
        observer_log_probs = F.log_softmax(shift_obs.float(), dim=-1)
        x_ppl = -(performer_probs * observer_log_probs).sum(dim=-1).mean().item()

        # Also compute performer's own PPL for comparison
        ppl_performer = F.cross_entropy(
            shift_perf.float().view(-1, shift_perf.size(-1)),
            labels.view(-1),
            reduction="mean",
        ).item()

        # KL divergence between performer and observer (measures model disagreement)
        observer_probs = F.softmax(shift_obs.float(), dim=-1)
        kl_div = F.kl_div(
            observer_log_probs, performer_probs,
            reduction="batchmean", log_target=False,
        ).item()

        score = ppl / x_ppl

        return {
            "score": score,
            "ppl_observer": ppl,
            "ppl_performer": ppl_performer,
            "x_ppl": x_ppl,
            "kl_divergence": kl_div,
            "token_count": token_count,
        }


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading models...")

    detector = DiagnosticDetector(device=device)
    detector.load_models()
    print("Models loaded.\n")

    print("=" * 110)
    print(f"{'ID':<30} {'Label':>5} {'Score':>7} {'PPL_obs':>9} {'PPL_perf':>9} "
          f"{'X-PPL':>9} {'KL(P||O)':>9} {'PPL/XPPL':>9}")
    print("-" * 110)

    human_results = []
    ai_results = []

    for samples, label in [(HUMAN_SAMPLES, "H"), (AI_SAMPLES, "AI")]:
        for sample in samples:
            r = detector.compute_score_diagnostic(sample["text"])
            row = (
                f"  {sample['id']:<30} {label:>5} {r['score']:>7.4f} "
                f"{r['ppl_observer']:>9.4f} {r['ppl_performer']:>9.4f} "
                f"{r['x_ppl']:>9.4f} {r['kl_divergence']:>9.4f} "
                f"{r['ppl_observer']/r['x_ppl']:>9.4f}"
            )
            print(row)
            if label == "H":
                human_results.append(r)
            else:
                ai_results.append(r)
        print()

    # Averages
    print("=" * 110)
    print("\nCOMPONENT AVERAGES")
    print("-" * 70)

    def avg(results, key):
        return sum(r[key] for r in results) / len(results)

    metrics = [
        ("PPL (observer)", "ppl_observer",
         "How hard is the text for the base model to predict?"),
        ("PPL (performer)", "ppl_performer",
         "How hard is the text for the instruct model to predict?"),
        ("X-PPL (cross)", "x_ppl",
         "How much do the two models' distributions disagree?"),
        ("KL divergence", "kl_divergence",
         "Direct measure of distribution divergence"),
        ("Score (PPL/X-PPL)", "score",
         "The Binoculars score"),
    ]

    print(f"\n  {'Metric':<25} {'Human':>10} {'AI':>10} {'Diff':>10}  Interpretation")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}  {'-'*30}")

    for name, key, desc in metrics:
        h = avg(human_results, key)
        a = avg(ai_results, key)
        diff = a - h
        print(f"  {name:<25} {h:>10.4f} {a:>10.4f} {diff:>+10.4f}  {desc}")

    # The explanation
    print("\n" + "=" * 110)
    print("DIAGNOSIS")
    print("=" * 110)

    h_ppl = avg(human_results, "ppl_observer")
    a_ppl = avg(ai_results, "ppl_observer")
    h_xppl = avg(human_results, "x_ppl")
    a_xppl = avg(ai_results, "x_ppl")
    h_kl = avg(human_results, "kl_divergence")
    a_kl = avg(ai_results, "kl_divergence")

    ppl_drop_pct = (a_ppl - h_ppl) / h_ppl * 100
    xppl_drop_pct = (a_xppl - h_xppl) / h_xppl * 100

    print(f"""
  PPL (observer) drops by {ppl_drop_pct:+.1f}% from human to AI text.
  X-PPL drops by {xppl_drop_pct:+.1f}% from human to AI text.

  The score = PPL / X-PPL inverts when PPL drops MORE than X-PPL.
  PPL dropped {abs(ppl_drop_pct):.1f}% vs X-PPL dropped {abs(xppl_drop_pct):.1f}%.
""")

    if abs(ppl_drop_pct) > abs(xppl_drop_pct):
        print("  --> PPL dropped MORE than X-PPL. This explains the inversion.")
        print()
        print("  WHY: AI-generated grant text is extremely predictable to the")
        print("  observer model (very low PPL). The observer finds it EASIER to")
        print("  predict the actual tokens than to match the performer's distribution.")
        print("  The cross-perplexity (X-PPL) has a natural floor set by the inherent")
        print("  divergence between base and instruct models — it can't drop as fast.")
        print()
        print("  In the paper's original test domain (news, Wikipedia, Reddit),")
        print("  AI text was predictable but not dramatically more so than human text.")
        print("  In grant writing, AI text is MAXIMALLY formulaic, pushing PPL so")
        print("  low that the ratio inverts.")
    else:
        print("  --> X-PPL dropped MORE than PPL. Different mechanism at play.")

    print(f"""
  KL divergence (model disagreement):
    Human text: {h_kl:.4f}
    AI text:    {a_kl:.4f}

  {'Models AGREE more on AI text' if a_kl < h_kl else 'Models DISAGREE more on AI text'}.
  {'This is consistent with the paper theory (both models see AI text similarly).' if a_kl < h_kl else 'This contradicts the paper — models disagree more on AI text.'}
""")


if __name__ == "__main__":
    main()
