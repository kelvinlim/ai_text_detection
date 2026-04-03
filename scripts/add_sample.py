#!/usr/bin/env python3
"""Add a new test sample as a YAML file.

Creates a YAML sample file in tests/samples/<section>/ with full provenance
metadata. Accepts text from a file, inline argument, or stdin.

Usage:
    # From a text file:
    python scripts/add_sample.py \\
        --id ai_specific_aims_gpt4o \\
        --label ai_generated \\
        --section "Specific Aims" \\
        --source gpt-4o \\
        --prompt "Write a Specific Aims page for an NIH R01 about..." \\
        --text-file output.txt

    # Inline text:
    python scripts/add_sample.py \\
        --id human_aims_new \\
        --label human \\
        --section "Specific Aims" \\
        --text "The goal of this research..."

    # Interactive mode:
    python scripts/add_sample.py --interactive

    # Pipe from clipboard (macOS):
    pbpaste | python scripts/add_sample.py \\
        --id ai_aims_claude4 \\
        --label ai_generated \\
        --section "Specific Aims" \\
        --source claude-sonnet-4-20250514 \\
        --prompt "Write Specific Aims for..." \\
        --text-file -
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "tests" / "samples"

# Known section-to-directory mappings
SECTION_DIR_MAP = {
    "specific aims": "specific_aims",
    "significance": "significance",
    "innovation": "innovation",
    "approach": "approach",
    "preliminary data": "preliminary_data",
    "background": "background",
    "rigor and reproducibility": "rigor",
    "rigor": "rigor",
    "environment": "environment",
    "budget justification": "budget",
    "budget": "budget",
    "biosketch narrative": "biosketch",
    "biosketch": "biosketch",
}


class LiteralStr(str):
    pass


def literal_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(LiteralStr, literal_representer)


def section_to_dir(section: str) -> str:
    """Map a section name to its directory name."""
    key = section.lower().strip()
    if key in SECTION_DIR_MAP:
        return SECTION_DIR_MAP[key]
    return key.replace(" ", "_").replace("&", "and")


def infer_origin(source: str | None, label: str) -> str:
    """Infer origin from source model name and label."""
    if label == "human":
        return "handwritten"
    if label == "mixed":
        return "mixed_edited"
    if not source:
        return "unknown"
    s = source.lower()
    if "claude" in s or "anthropic" in s:
        return "claude_generated"
    if "gpt" in s or "openai" in s or "o1" in s or "o3" in s:
        return "openai_generated"
    if "gemini" in s or "google" in s:
        return "google_generated"
    # Check common local/ollama models
    for name in ("llama", "qwen", "gemma", "mistral", "phi", "deepseek"):
        if name in s:
            return "ollama_generated"
    return "ai_generated"


def interactive_mode() -> dict:
    """Collect sample metadata interactively."""
    print("=== Add New Sample (Interactive) ===\n")

    sample_id = input("Sample ID (e.g. ai_specific_aims_gpt4o): ").strip()
    if not sample_id:
        print("Error: ID is required.")
        sys.exit(1)

    label = input("Label [ai_generated/human/mixed/uncertain/skip]: ").strip()
    if not label:
        label = "ai_generated"

    section = input("Grant section (e.g. Specific Aims): ").strip()
    if not section:
        print("Error: Section is required.")
        sys.exit(1)

    source = input("Source model (e.g. gpt-4o, claude-sonnet-4-20250514) [blank for human]: ").strip() or None

    prompt = input("Prompt used (or press Enter to skip): ").strip() or None

    print("\nPaste the sample text below. Enter a blank line then Ctrl-D (or Ctrl-Z on Windows) to finish:")
    lines = []
    try:
        for line in sys.stdin:
            lines.append(line.rstrip("\n"))
    except EOFError:
        pass
    text = "\n".join(lines).strip()

    if not text:
        print("Error: No text provided.")
        sys.exit(1)

    return {
        "id": sample_id,
        "label": label,
        "section": section,
        "source": source,
        "prompt": prompt,
        "text": text,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Add a new test sample as a YAML file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode — prompts for all fields")
    parser.add_argument("--id", help="Unique sample ID")
    parser.add_argument("--label", default="ai_generated",
                        choices=["human", "ai_generated", "mixed",
                                 "uncertain", "skip"],
                        help="Sample label (default: ai_generated)")
    parser.add_argument("--section", help="Grant section name")
    parser.add_argument("--category",
                        help="Category for grouping (auto-inferred if omitted)")
    parser.add_argument("--source", help="Model name (e.g. gpt-4o)")
    parser.add_argument("--prompt", help="Prompt used to generate the text")
    parser.add_argument("--origin",
                        help="Origin type (auto-inferred if omitted)")
    parser.add_argument("--description", help="Description or notes")
    parser.add_argument("--text", help="Sample text (inline)")
    parser.add_argument("--text-file",
                        help="Read sample text from file (use - for stdin)")
    args = parser.parse_args()

    if args.interactive:
        info = interactive_mode()
    else:
        if not args.id or not args.section:
            parser.error("--id and --section are required (or use --interactive)")

        # Read text
        if args.text_file:
            if args.text_file == "-":
                text = sys.stdin.read().strip()
            else:
                text = Path(args.text_file).read_text(encoding="utf-8").strip()
        elif args.text:
            text = args.text.strip()
        else:
            parser.error("Provide text via --text, --text-file, or pipe to --text-file -")

        info = {
            "id": args.id,
            "label": args.label,
            "section": args.section,
            "source": args.source,
            "prompt": args.prompt,
            "text": text,
        }

    # Build YAML document
    doc = {
        "id": info["id"],
        "label": info["label"],
        "section": info["section"],
        "category": args.category if not args.interactive and args.category else info["label"],
        "origin": (args.origin if not args.interactive and args.origin
                   else infer_origin(info.get("source"), info["label"])),
    }

    if info.get("source"):
        doc["source"] = info["source"]

    doc["date_created"] = str(date.today())

    if args.description if not args.interactive else None:
        doc["description"] = args.description

    if info.get("prompt"):
        doc["prompt"] = LiteralStr(info["prompt"])

    doc["text"] = LiteralStr(info["text"])

    # Determine output path
    dir_name = section_to_dir(info["section"])
    out_dir = SAMPLES_DIR / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{info['id']}.yaml"

    if out_path.exists():
        print(f"Warning: {out_path.relative_to(ROOT)} already exists!")
        confirm = input("Overwrite? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            sys.exit(1)

    yaml_str = yaml.dump(doc, default_flow_style=False, allow_unicode=True,
                         sort_keys=False, width=80)
    out_path.write_text(yaml_str, encoding="utf-8")

    print(f"\nCreated: {out_path.relative_to(ROOT)}")
    print(f"  ID:       {info['id']}")
    print(f"  Label:    {info['label']}")
    print(f"  Section:  {info['section']}")
    print(f"  Source:   {info.get('source', '(none)')}")
    print(f"  Words:    {len(info['text'].split())}")
    if info.get("prompt"):
        prompt_preview = info["prompt"][:80] + ("..." if len(info["prompt"]) > 80 else "")
        print(f"  Prompt:   {prompt_preview}")


if __name__ == "__main__":
    main()
