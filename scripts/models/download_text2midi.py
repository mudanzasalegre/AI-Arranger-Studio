from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = ["pytorch_model.bin", "vocab_remi.pkl"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="amaai-lab/text2midi")
    parser.add_argument("--checkpoint-dir", default="models/checkpoints/text2midi")
    parser.add_argument("--flan-tokenizer", default="google/flan-t5-base")
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "Missing huggingface_hub. Run: python -m pip install huggingface_hub"
        ) from exc

    checkpoint_dir = ROOT / args.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for filename in FILES:
        path = hf_hub_download(
            repo_id=args.repo_id,
            filename=filename,
            local_dir=str(checkpoint_dir),
        )
        downloaded.append(path)

    tokenizer_cached = False
    tokenizer_error = None
    try:
        from transformers import T5Tokenizer

        T5Tokenizer.from_pretrained(args.flan_tokenizer)
        tokenizer_cached = True
    except Exception as exc:
        tokenizer_error = str(exc)

    report = {
        "status": "ok" if tokenizer_cached else "partial_ok",
        "repo_id": args.repo_id,
        "checkpoint_dir": str(checkpoint_dir),
        "files": downloaded,
        "flan_tokenizer": args.flan_tokenizer,
        "flan_tokenizer_cached": tokenizer_cached,
        "flan_tokenizer_error": tokenizer_error,
    }
    report_path = ROOT / "models/manifests/text2midi_download_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
