from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

STEPS = [
    ("verify_models", [sys.executable, "scripts/models_pro/verify_all_models.py"]),
    ("golden_baseline", [sys.executable, "scripts/golden_generate.py"]),
    ("professional_benchmark", [sys.executable, "scripts/models/professional_generation_benchmark.py", "--config", "configs/professional_benchmarks.yaml"]),
]

def run(cmd: list[str]) -> dict:
    completed = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    return {
        "cmd": cmd,
        "returncode": completed.returncode,
        "ok": completed.returncode == 0,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }

def main() -> None:
    results = {}
    for name, cmd in STEPS:
        results[name] = run(cmd)
    status = "ok" if all(item["ok"] for item in results.values()) else "fail"
    report = {"status": status, "results": results}
    out = ROOT / "outputs/pro_benchmarks/pro_end_to_end_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if status != "ok":
        raise SystemExit(1)

if __name__ == "__main__":
    main()
