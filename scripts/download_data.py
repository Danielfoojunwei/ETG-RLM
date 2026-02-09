#!/usr/bin/env python3
"""Download benchmark datasets for the ETG canonical evaluation.

Downloads and prepares the following datasets:
    1. TruthfulQA (817 instances) -- truthfulness under adversarial priors
    2. HaluEval (1000 instances) -- hallucination detection benchmark
    3. HotpotQA (500 instances) -- multi-hop reasoning
    4. Natural Questions (1000 instances) -- open-domain factoid QA
    5. ELI5 (500 instances) -- long-form explanatory answers

Usage:
    python scripts/download_data.py --output-dir data/ [--datasets all]
    python scripts/download_data.py --output-dir data/ --datasets truthfulqa hotpotqa

References:
    [1] Lin et al., "TruthfulQA," ACL 2022.
    [2] Li et al., "HaluEval," EMNLP 2023.
    [3] Yang et al., "HotpotQA," EMNLP 2018.
    [6] Kwiatkowski et al., "Natural Questions," TACL 2019.
    [9] Fan et al., "ELI5," ACL 2019.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetMeta:
    """Metadata for a benchmark dataset."""

    name: str
    description: str
    huggingface_id: str
    split: str
    n_instances: int
    citation: str


DATASETS: dict[str, DatasetMeta] = {
    "truthfulqa": DatasetMeta(
        name="TruthfulQA",
        description="Truthfulness under adversarial priors (817 questions)",
        huggingface_id="truthful_qa",
        split="validation",
        n_instances=817,
        citation="Lin et al., ACL 2022",
    ),
    "halueval": DatasetMeta(
        name="HaluEval",
        description="Hallucination evaluation benchmark (QA + summarization)",
        huggingface_id="pminervini/HaluEval",
        split="data",
        n_instances=1000,
        citation="Li et al., EMNLP 2023",
    ),
    "hotpotqa": DatasetMeta(
        name="HotpotQA",
        description="Multi-hop reasoning QA (distractor setting)",
        huggingface_id="hotpot_qa",
        split="validation",
        n_instances=500,
        citation="Yang et al., EMNLP 2018",
    ),
    "natural_questions": DatasetMeta(
        name="Natural Questions",
        description="Open-domain factoid QA from Google Search",
        huggingface_id="natural_questions",
        split="validation",
        n_instances=1000,
        citation="Kwiatkowski et al., TACL 2019",
    ),
    "eli5": DatasetMeta(
        name="ELI5",
        description="Long-form explanatory answers (Explain Like I'm 5)",
        huggingface_id="eli5",
        split="validation_asks",
        n_instances=500,
        citation="Fan et al., ACL 2019",
    ),
}


# ---------------------------------------------------------------------------
# Download functions
# ---------------------------------------------------------------------------


def download_dataset(
    meta: DatasetMeta,
    output_dir: Path,
    max_instances: int | None = None,
) -> Path:
    """Download a single dataset and save to disk.

    Args:
        meta: dataset metadata
        output_dir: directory to save the dataset
        max_instances: limit number of instances (None = use default)

    Returns:
        Path to the saved dataset file.
    """
    out_path = output_dir / f"{meta.name.lower().replace(' ', '_')}.jsonl"
    n = max_instances or meta.n_instances

    # Check if HuggingFace datasets is available
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError:
        print(f"  [STUB] HuggingFace 'datasets' not installed.")
        print(f"  [STUB] Would download: {meta.huggingface_id} ({meta.split})")
        print(f"  [STUB] Creating placeholder at: {out_path}")
        _write_placeholder(out_path, meta, n)
        return out_path

    print(f"  Downloading {meta.name} from HuggingFace: {meta.huggingface_id}")
    try:
        ds = load_dataset(meta.huggingface_id, split=meta.split)
        ds = ds.select(range(min(n, len(ds))))

        with open(out_path, "w") as f:
            for item in ds:
                f.write(json.dumps(item) + "\n")

        print(f"  Saved {len(ds)} instances to {out_path}")
    except Exception as e:
        print(f"  [ERROR] Failed to download {meta.name}: {e}")
        print(f"  [STUB] Creating placeholder at: {out_path}")
        _write_placeholder(out_path, meta, n)

    return out_path


def _write_placeholder(path: Path, meta: DatasetMeta, n: int) -> None:
    """Write a placeholder JSONL file for when the real download fails."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i in range(min(n, 5)):
            entry = {
                "instance_id": f"{meta.name.lower()}_{i:04d}",
                "question": f"[Placeholder question {i} for {meta.name}]",
                "answer": f"[Placeholder answer {i}]",
                "_meta": {
                    "dataset": meta.name,
                    "huggingface_id": meta.huggingface_id,
                    "split": meta.split,
                    "total_instances": meta.n_instances,
                    "citation": meta.citation,
                    "placeholder": True,
                },
            }
            f.write(json.dumps(entry) + "\n")


def download_all(
    output_dir: Path,
    dataset_names: list[str] | None = None,
) -> dict[str, Path]:
    """Download all specified datasets.

    Args:
        output_dir: base output directory
        dataset_names: list of dataset keys to download (None = all)

    Returns:
        Mapping from dataset name to output path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    names = dataset_names or list(DATASETS.keys())

    results: dict[str, Path] = {}
    for name in names:
        if name not in DATASETS:
            print(f"  [WARN] Unknown dataset: {name}. Skipping.")
            continue

        meta = DATASETS[name]
        print(f"\n{'='*60}")
        print(f"Dataset: {meta.name}")
        print(f"  {meta.description}")
        print(f"  Source: {meta.huggingface_id} ({meta.split})")
        print(f"  Instances: {meta.n_instances}")
        print(f"  Citation: {meta.citation}")
        print(f"{'='*60}")

        path = download_dataset(meta, output_dir)
        results[name] = path

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download benchmark datasets for ETG evaluation"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory to save datasets (default: data/)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to download (default: all). Options: "
             + ", ".join(DATASETS.keys()),
    )

    args = parser.parse_args()

    print("ETG Canonical Evaluation - Dataset Download")
    print(f"Output directory: {args.output_dir}")

    names = args.datasets
    if names and "all" in names:
        names = None

    results = download_all(args.output_dir, names)

    print(f"\n{'='*60}")
    print("Download Summary:")
    for name, path in results.items():
        print(f"  {name}: {path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
