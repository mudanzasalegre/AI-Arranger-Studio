from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for package_path in (
    ROOT / "packages" / "arranger_core",
    ROOT / "packages" / "dataset_tools",
    ROOT / "packages" / "training",
):
    sys.path.insert(0, str(package_path))

from training import train_baseline_statistical_models  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train baseline statistical role models from tokenized segments."
    )
    parser.add_argument(
        "--tokenized-segments",
        required=True,
        help="Path to tokenized_segments.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "statistical_baselines"),
        help="Directory where model artifacts and comparison reports will be written.",
    )
    parser.add_argument("--seed", type=int, default=1800)
    parser.add_argument("--ngram-order", type=int, default=3)
    args = parser.parse_args()

    summary = train_baseline_statistical_models(
        args.tokenized_segments,
        args.output_dir,
        seed=args.seed,
        ngram_order=args.ngram_order,
    )
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
