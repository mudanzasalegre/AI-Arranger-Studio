from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

SYSTEM = """You are a symbolic music planner. Return only JSON. Do not write notes, MIDI, MusicXML, lyrics or audio.
Required JSON keys: style, substyle, tempo, meter, key, form, ensemble, instruments, sections, generation_strategy.
"""
USER = """Plan a hard bop minor blues in C minor, 132 BPM, jazz sextet with alto sax, trumpet, trombone, piano, double bass and drums. Return JSON only."""


def extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/api")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()

    try:
        import httpx
    except ImportError as exc:
        raise SystemExit("Missing httpx. It is included in requirements.txt.") from exc

    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER},
        ],
        "stream": False,
        "format": "json",
    }
    url = args.base_url.rstrip("/") + "/chat"
    with httpx.Client(timeout=args.timeout) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

    content = data.get("message", {}).get("content") or data.get("response") or ""
    parsed = extract_json(content)
    required = ["style", "tempo", "meter", "key", "form", "ensemble", "instruments", "sections", "generation_strategy"]
    missing = [key for key in required if key not in parsed]
    report = {
        "status": "ok" if not missing else "fail",
        "model": args.model,
        "base_url": args.base_url,
        "missing": missing,
        "parsed": parsed,
    }
    report_path = ROOT / "outputs/model_smoke/ollama_planner_smoke_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
