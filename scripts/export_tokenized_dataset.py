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

from training import export_tokenized_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export role-tokenized training datasets.")
    parser.add_argument("--pattern-index", required=True, help="Path to PatternIndex JSON.")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "tokenized_dataset"),
        help="Directory where tokenized dataset files will be written.",
    )
    parser.add_argument("--seed", type=int, default=1700)
    parser.add_argument("--min-quality", type=int, default=3)
    parser.add_argument(
        "--roles",
        nargs="*",
        default=None,
        help="Optional subset: melody bass piano_comping horn_responses drums.",
    )
    args = parser.parse_args()

    summary = export_tokenized_dataset(
        args.pattern_index,
        args.output_dir,
        seed=args.seed,
        min_quality=args.min_quality,
        roles=args.roles
        if args.roles
        else ("melody", "bass", "piano_comping", "horn_responses", "drums"),
    )
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
