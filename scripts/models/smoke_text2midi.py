from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", default="models/external_repos/text2midi")
    parser.add_argument("--checkpoint-dir", default="models/checkpoints/text2midi")
    parser.add_argument("--output", default="outputs/model_artifacts/raw/text2midi_smoke.mid")
    parser.add_argument("--seed", type=int, default=2201)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--prompt",
        default=(
            "A short hard bop jazz MIDI in C minor, 132 BPM, with piano, bass, "
            "drums and alto saxophone."
        ),
    )
    parser.add_argument(
        "--allow-check-only",
        action="store_true",
        help="Only verify repo/checkpoints when inference wrapper is not implemented yet.",
    )
    args = parser.parse_args()

    repo_dir = ROOT / args.repo_dir
    checkpoint_dir = ROOT / args.checkpoint_dir
    output = ROOT / args.output
    required = [checkpoint_dir / "pytorch_model.bin", checkpoint_dir / "vocab_remi.pkl"]
    missing = [str(path) for path in required if not path.exists()]

    report = {
        "repo_dir": str(repo_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "output": str(output),
        "missing": missing,
        "inference_attempted": False,
        "status": "pending",
    }

    if not repo_dir.exists():
        report["status"] = "fail"
        report["error"] = f"Text2MIDI repo not found: {repo_dir}"
    elif missing:
        report["status"] = "fail"
        report["error"] = "Missing checkpoint files"
    else:
        wrapper = ROOT / "scripts/models/run_text2midi_inference.py"
        if wrapper.exists():
            report["inference_attempted"] = True
            output.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(wrapper),
                "--repo-dir",
                str(repo_dir),
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--output",
                str(output),
                "--prompt",
                args.prompt,
                "--seed",
                str(args.seed),
                "--max-len",
                str(args.max_len),
                "--temperature",
                str(args.temperature),
                "--device",
                args.device,
            ]
            completed = subprocess.run(
                cmd, cwd=str(ROOT), text=True, capture_output=True, check=False
            )
            report["returncode"] = completed.returncode
            report["stdout"] = completed.stdout[-4000:]
            report["stderr"] = completed.stderr[-4000:]
            report["status"] = (
                "ok"
                if completed.returncode == 0 and output.exists() and output.stat().st_size > 0
                else "fail"
            )
        elif args.allow_check_only:
            report["status"] = "check_only"
            report["note"] = (
                "run_text2midi_inference.py not implemented yet; repo and checkpoints are present."
            )
        else:
            report["status"] = "fail"
            report["error"] = (
                "scripts/models/run_text2midi_inference.py is not implemented yet. "
                "Use --allow-check-only during PR-22 setup."
            )

    report_path = ROOT / "outputs/model_smoke/text2midi_smoke_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] not in {"ok", "check_only"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
