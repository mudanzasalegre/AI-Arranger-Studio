from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=os.environ.get("MIDIGPT_MODEL_NAME", "yellow"))
    parser.add_argument("--report", default="models/manifests/midigpt_download_report.json")
    args = parser.parse_args()

    try:
        from midigpt.inference import InferenceEngine
    except ImportError as exc:
        raise SystemExit(
            'MIDI-GPT is not installed. Run: python -m pip install "midigpt[inference]"'
        ) from exc

    engine = InferenceEngine.from_pretrained(args.model_name)
    report = {
        "status": "ok",
        "model_name": args.model_name,
        "engine_class": type(engine).__name__,
        "hf_home": os.environ.get("HF_HOME"),
        "hf_hub_cache": os.environ.get("HF_HUB_CACHE"),
    }
    path = ROOT / args.report
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
