#!/usr/bin/env python3
"""One-time migration: convert hardcoded Python samples to YAML files.

Creates tests/samples/<section>/<id>.yaml for each of the 42 samples.
Verifies round-trip correctness after writing.

Usage:
    python scripts/migrate_samples_to_yaml.py [--dry-run]
"""

import argparse
import sys
from pathlib import Path

import yaml

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.test_samples import HUMAN_SAMPLES, AI_SAMPLES, MIXED_SAMPLES, EDGE_CASES
from tests.test_samples_expanded import HUMAN_SAMPLES_EXPANDED, AI_SAMPLES_EXPANDED
from tests.test_samples_generated import OLLAMA_SAMPLES

SAMPLES_DIR = ROOT / "tests" / "samples"

# --- Section folder mapping ---
# Maps grant section names to directory names
SECTION_TO_DIR = {
    "Specific Aims": "specific_aims",
    "Significance": "significance",
    "Innovation": "innovation",
    "Approach": "approach",
    "Preliminary Data": "preliminary_data",
    "Background": "background",
    "Rigor and Reproducibility": "rigor",
    "Environment": "environment",
    "Budget Justification": "budget",
    "Biosketch Narrative": "biosketch",
    # Mixed/edge sections
    "Significance (mixed)": "mixed",
    "Innovation (mixed)": "mixed",
    "Approach (mixed)": "mixed",
    "Short Fragment": "edge_cases",
    "References": "edge_cases",
    "Methods (Dense Jargon)": "edge_cases",
    "Spanish Abstract": "edge_cases",
}

# --- Reconstructed prompts for AI samples ---
RECONSTRUCTED_PROMPTS = {
    # Original AI samples
    "ai_specific_aims": (
        "Write a Specific Aims section for an NIH R01 grant proposal about "
        "neuroinflammation in Alzheimer's disease, focusing on the NLRP3 "
        "inflammasome pathway in microglia."
    ),
    "ai_significance": (
        "Write a Significance section for an NIH grant proposal about cardiac "
        "regeneration using engineered extracellular vesicles derived from "
        "iPSC-cardiomyocytes."
    ),
    "ai_innovation": (
        "Write an Innovation section for an NIH grant proposal about how "
        "epitranscriptomic modifications regulate immune cell metabolism in "
        "solid tumor immunology, using single-cell multi-omics."
    ),
    "ai_approach": (
        "Write an Approach section for an NIH grant proposal about BDNF-TrkB "
        "signaling in activity-dependent synaptic plasticity, using rat "
        "hippocampal neuronal cultures and electrophysiology."
    ),
    "ai_preliminary_data": (
        "Write a Preliminary Data section for an NIH grant about "
        "tumor-associated macrophage subpopulations in non-small cell lung "
        "cancer, using single-cell RNA sequencing."
    ),
    # Expanded AI samples
    "ai_exp_background_1": (
        "Write a Background section for an NIH grant about the gut-brain axis "
        "and how the intestinal microbiome influences CNS function and "
        "neuropsychiatric conditions."
    ),
    "ai_exp_background_2": (
        "Write a Background section for an NIH grant about challenges of "
        "CAR-T cell therapy in solid tumors, focusing on the immunosuppressive "
        "tumor microenvironment."
    ),
    "ai_exp_rigor_1": (
        "Write a Rigor and Reproducibility section for an NIH grant, covering "
        "sex as a biological variable, power analysis, randomization, blinding, "
        "and antibody validation."
    ),
    "ai_exp_rigor_2": (
        "Write a Rigor and Reproducibility section for an NIH grant, covering "
        "quality control for RNA-seq, immunohistochemistry validation, blinded "
        "analysis, and statistical methods."
    ),
    "ai_exp_environment_1": (
        "Write an Environment section for an NIH grant describing a "
        "well-equipped biomedical research center with tissue culture, imaging "
        "core, AAALAC animal facility, and HPC cluster."
    ),
    "ai_exp_environment_2": (
        "Write an Environment section for an NIH grant describing a biomedical "
        "engineering department with shared core facilities, proximity to a "
        "medical school, and support for early-career investigators."
    ),
    "ai_exp_budget_1": (
        "Write a Budget Justification section for an NIH grant covering PI "
        "effort, a postdoc, a graduate student, supplies for molecular biology, "
        "and animal costs."
    ),
    "ai_exp_budget_2": (
        "Write a Budget Justification section for an NIH grant covering "
        "conference travel, open-access publication costs, and a benchtop flow "
        "cytometer equipment purchase."
    ),
    "ai_exp_biosketch_1": (
        "Write a Biosketch personal statement for an NIH grant PI who studies "
        "neurodegeneration and Alzheimer's disease, established their lab in "
        "2017, has R01 and R21 funding."
    ),
    "ai_exp_biosketch_2": (
        "Write a Biosketch personal statement for an NIH grant PI with an "
        "interdisciplinary background in biomedical engineering and tumor "
        "immunology, with an NSF CAREER Award."
    ),
    # Ollama samples
    "ollama_qwen3_specific_aims": (
        "Write a Specific Aims section for an NIH grant about "
        "neuroinflammation in Alzheimer's disease."
    ),
    "ollama_qwen3_significance": (
        "Write a Significance section for an NIH grant about cardiac "
        "regeneration."
    ),
    "ollama_gemma3_specific_aims": (
        "Write a Specific Aims section for an NIH grant about "
        "microglia-derived exosomes in Alzheimer's disease."
    ),
    "ollama_gemma3_significance": (
        "Write a Significance section for an NIH grant about cardiac "
        "regeneration after myocardial infarction."
    ),
    "ollama_llama31_specific_aims": (
        "Write a Specific Aims section for an NIH grant about "
        "neuroinflammation in Alzheimer's disease."
    ),
    "ollama_llama31_approach": (
        "Write an Approach section for an NIH grant about CRISPR gene editing "
        "for sickle cell disease."
    ),
}


def section_dir(section: str) -> str:
    """Map a section name to its directory name."""
    if section in SECTION_TO_DIR:
        return SECTION_TO_DIR[section]
    # Fallback: slugify
    return section.lower().replace(" ", "_").replace("&", "and")


def build_yaml_doc(sample: dict, category: str, origin: str,
                   source: str | None = None,
                   date_created: str | None = None) -> dict:
    """Build an ordered dict for YAML output."""
    doc = {
        "id": sample["id"],
        "label": sample["label"],
        "section": sample["section"],
        "category": category,
    }
    if origin:
        doc["origin"] = origin
    if source:
        doc["source"] = source
    if date_created:
        doc["date_created"] = date_created
    if sample.get("description"):
        doc["description"] = sample["description"]

    # Add reconstructed prompt for AI samples
    prompt = RECONSTRUCTED_PROMPTS.get(sample["id"])
    if prompt:
        doc["prompt"] = prompt
        doc["prompt_reconstructed"] = True

    doc["text"] = sample["text"].strip()
    return doc


class LiteralStr(str):
    """String subclass that YAML will dump as a literal block scalar."""
    pass


def literal_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(LiteralStr, literal_representer)


def to_yaml_str(doc: dict) -> str:
    """Serialize a sample doc to YAML with literal block scalars for text."""
    # Wrap long string fields in LiteralStr for block scalar style
    doc = dict(doc)
    doc["text"] = LiteralStr(doc["text"])
    if "prompt" in doc:
        doc["prompt"] = LiteralStr(doc["prompt"])
    if "description" in doc and "\n" in doc.get("description", ""):
        doc["description"] = LiteralStr(doc["description"])

    return yaml.dump(doc, default_flow_style=False, allow_unicode=True,
                     sort_keys=False, width=80)


def migrate_samples(samples, category, origin, source=None, date_created=None,
                    dry_run=False):
    """Migrate a list of samples to YAML files."""
    results = []
    for sample in samples:
        sample_source = sample.get("source", source)
        doc = build_yaml_doc(sample, category, origin, sample_source,
                             date_created)
        dir_name = section_dir(sample["section"])
        out_dir = SAMPLES_DIR / dir_name
        out_path = out_dir / f"{sample['id']}.yaml"

        if dry_run:
            print(f"  [DRY RUN] {out_path.relative_to(ROOT)}")
            results.append((sample, doc, out_path))
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        yaml_str = to_yaml_str(doc)
        out_path.write_text(yaml_str, encoding="utf-8")
        print(f"  wrote {out_path.relative_to(ROOT)}")
        results.append((sample, doc, out_path))

    return results


def verify_roundtrip(results):
    """Verify that each YAML file loads back with matching text."""
    errors = 0
    for sample, doc, out_path in results:
        loaded = yaml.safe_load(out_path.read_text(encoding="utf-8"))
        original_text = sample["text"].strip()
        loaded_text = loaded["text"].strip()
        if original_text != loaded_text:
            print(f"  MISMATCH: {sample['id']}")
            print(f"    original len={len(original_text)}")
            print(f"    loaded   len={len(loaded_text)}")
            errors += 1
        if loaded["id"] != sample["id"]:
            print(f"  ID MISMATCH: {loaded['id']} != {sample['id']}")
            errors += 1
        if loaded["label"] != sample["label"]:
            print(f"  LABEL MISMATCH: {sample['id']}")
            errors += 1
    return errors


def main():
    parser = argparse.ArgumentParser(description="Migrate samples to YAML")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be created without writing")
    args = parser.parse_args()

    print("=" * 70)
    print("MIGRATING TEST SAMPLES TO YAML")
    print("=" * 70)

    all_results = []

    # Original human samples
    print("\nOriginal human samples (5):")
    all_results += migrate_samples(
        HUMAN_SAMPLES, category="human", origin="handwritten",
        date_created="2026-04-01", dry_run=args.dry_run)

    # Original AI samples
    print("\nOriginal AI samples (5):")
    all_results += migrate_samples(
        AI_SAMPLES, category="ai", origin="claude_generated",
        source="claude-3.5-sonnet", date_created="2026-04-01",
        dry_run=args.dry_run)

    # Mixed samples
    print("\nMixed samples (3):")
    all_results += migrate_samples(
        MIXED_SAMPLES, category="mixed", origin="mixed_edited",
        date_created="2026-04-01", dry_run=args.dry_run)

    # Edge cases
    print("\nEdge cases (4):")
    all_results += migrate_samples(
        EDGE_CASES, category="edge_case", origin="handwritten",
        date_created="2026-04-01", dry_run=args.dry_run)

    # Expanded human samples
    print("\nExpanded human samples (10):")
    all_results += migrate_samples(
        HUMAN_SAMPLES_EXPANDED, category="human_expanded",
        origin="handwritten", date_created="2026-04-01",
        dry_run=args.dry_run)

    # Expanded AI samples
    print("\nExpanded AI samples (10):")
    all_results += migrate_samples(
        AI_SAMPLES_EXPANDED, category="ai_expanded",
        origin="claude_generated", source="claude-3.5-sonnet",
        date_created="2026-04-01", dry_run=args.dry_run)

    # Ollama samples
    print("\nOllama samples (6):")
    all_results += migrate_samples(
        OLLAMA_SAMPLES, category="ollama", origin="ollama_generated",
        date_created="2026-04-02", dry_run=args.dry_run)

    print(f"\nTotal: {len(all_results)} samples")

    if args.dry_run:
        print("\nDry run complete. No files written.")
        return

    # Verify round-trip
    print("\n" + "-" * 70)
    print("VERIFYING ROUND-TRIP...")
    errors = verify_roundtrip(all_results)
    if errors:
        print(f"\n{errors} ERRORS found!")
        sys.exit(1)
    else:
        print(f"All {len(all_results)} samples verified OK.")

    print("\nMigration complete.")


if __name__ == "__main__":
    main()
