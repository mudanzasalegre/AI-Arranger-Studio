from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def post(client, url: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = client.post(url, json=payload)
    try:
        data = response.json()
    except Exception:
        data = {"text": response.text}
    if response.status_code >= 400:
        raise RuntimeError(f"POST {url} failed {response.status_code}: {data}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--use-midigpt", action="store_true")
    parser.add_argument("--use-text2midi", action="store_true")
    args = parser.parse_args()

    try:
        import httpx
    except ImportError as exc:
        raise SystemExit("Missing httpx. Run `python -m pip install -r requirements.txt`.") from exc

    base = args.api.rstrip("/")
    summary: dict[str, Any] = {"status": "pending", "api": base, "steps": []}

    with httpx.Client(timeout=args.timeout) as client:
        health = client.get(base + "/health")
        health.raise_for_status()
        summary["steps"].append({"step": "health", "status": "ok", "response": health.json()})

        models = client.get(base + "/v1/ai/models").json()
        summary["steps"].append({"step": "models", "status": "ok", "models": models.get("models", [])})

        project_id = f"local_smoke_{int(time.time())}"
        generated = post(
            client,
            base + "/v1/projects/generate",
            {
                "project_id": project_id,
                "prompt": "hard bop nocturno en Do menor, 132 bpm, blues menor, sexteto con saxo alto, trompeta, trombon, piano, contrabajo y bateria",
                "seed": 2601,
                "options": {"export": True, "validate": True, "include_pdf": False},
            },
        )
        summary["steps"].append({"step": "generate_project", "status": "ok", "project_id": project_id, "response": generated})

        plan = post(
            client,
            base + f"/v1/projects/{project_id}/ai/plan",
            {"prompt": "haz el shout chorus más intenso pero sin cambiar pistas", "seed": 2602},
        )
        summary["steps"].append({"step": "ai_plan", "status": "ok", "response": plan})

        tracks = generated.get("project", {}).get("tracks", [])
        target_track = next((t["id"] for t in tracks if t.get("role") in {"melody", "horn_response"}), None)
        if target_track is None and tracks:
            target_track = tracks[0]["id"]
        if target_track is None:
            raise RuntimeError("No target track found for infill smoke")

        mock_take = post(
            client,
            base + f"/v1/projects/{project_id}/ai/infill",
            {
                "backend": "mock_symbolic",
                "track_id": target_track,
                "bars": [1],
                "instruction": "mock infill smoke",
                "seed": 2603,
            },
        )
        summary["steps"].append({"step": "mock_infill", "status": "ok", "response": mock_take})

        take_id = mock_take.get("take", {}).get("take_id")
        if take_id:
            accept = post(client, base + f"/v1/projects/{project_id}/takes/{take_id}/accept", {})
            summary["steps"].append({"step": "accept_mock_take", "status": "ok", "response": accept})

        if args.use_midigpt:
            midigpt_take = post(
                client,
                base + f"/v1/projects/{project_id}/ai/infill",
                {
                    "backend": "midigpt",
                    "track_id": target_track,
                    "bars": [2, 3, 4, 5],
                    "instruction": "local MIDI-GPT smoke infill, medium density, playable range",
                    "seed": 2604,
                },
            )
            summary["steps"].append({"step": "midigpt_infill", "status": "ok", "response": midigpt_take})

        if args.use_text2midi:
            sketch = post(
                client,
                base + "/v1/ai/text-to-midi-sketch",
                {
                    "backend": "text2midi",
                    "prompt": "Hard bop minor blues in C minor, 132 BPM, with piano, double bass, drums and alto sax.",
                    "seed": 2605,
                },
            )
            summary["steps"].append({"step": "text2midi_sketch", "status": "ok", "response": sketch})

        export = post(client, base + f"/v1/projects/{project_id}/export", {"include_pdf": False})
        summary["steps"].append({"step": "export", "status": "ok", "response": export})

    summary["status"] = "ok"
    out = ROOT / "outputs/model_smoke/ai_local_smoke_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
