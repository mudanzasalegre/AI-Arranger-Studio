from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=os.environ.get("MIDIGPT_MODEL_NAME", "yellow"))
    parser.add_argument("--output", default="outputs/model_artifacts/raw/midigpt_smoke.mid")
    parser.add_argument("--model-dim", type=int, default=4)
    args = parser.parse_args()

    try:
        from midigpt import Bar, Score, Track
        from midigpt.inference import InferenceConfig, InferenceEngine, GenerationRequest, TrackPrompt
    except ImportError as exc:
        raise SystemExit('MIDI-GPT is not installed. Run: python -m pip install "midigpt[inference]"') from exc

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    engine = InferenceEngine.from_pretrained(args.model_name)
    score = Score(tracks=[Track(bars=[Bar() for _ in range(args.model_dim)])])
    request = GenerationRequest(
        tracks=[TrackPrompt(id=0, bars=list(range(args.model_dim)))],
        config=InferenceConfig(model_dim=args.model_dim, mask_mode="attention"),
    )
    result = engine.session(score, request).run()
    result.to_midi(str(output_path))

    total_notes = sum(len(bar.notes) for track in result.tracks for bar in track.bars)
    report = {
        "status": "ok" if output_path.exists() and output_path.stat().st_size > 0 else "fail",
        "model_name": args.model_name,
        "output": str(output_path),
        "bytes": output_path.stat().st_size if output_path.exists() else 0,
        "total_notes": total_notes,
    }
    report_path = ROOT / "outputs/model_smoke/midigpt_smoke_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
