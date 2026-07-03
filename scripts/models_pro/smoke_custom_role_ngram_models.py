from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for package in (
    "arranger_core",
    "dataset_tools",
    "model_backends",
    "training",
):
    path = str(ROOT / "packages" / package)
    if path not in sys.path:
        sys.path.insert(0, path)

from arranger_core import (  # noqa: E402
    ArtifactImporter,
    ArtifactStore,
    GenerationSpec,
    ProjectMerger,
    ValidationGate,
    generate_arrangement,
)
from model_backends import ModelGenerationRequest, build_model_backend_registry  # noqa: E402
from model_backends.config import AIModelsConfig, BackendConfig  # noqa: E402
from training import checkpoint_dir_for_role  # noqa: E402

DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "pro_benchmarks" / "custom_role_ngram_smoke"
DEFAULT_CHECKPOINT_ROOT = ROOT / "models" / "checkpoints" / "custom"
REQUIRED_CHECKPOINT_FILES = (
    "model.json",
    "tokenizer.json",
    "config.yaml",
    "training_manifest.yaml",
    "license_report.json",
    "metrics.json",
)
ROLE_BACKENDS = {
    "melody": "custom_jazz_melody_v001",
    "walking_bass": "custom_jazz_walking_bass_v001",
    "piano_comping": "custom_jazz_piano_comping_v001",
    "horn_responses": "custom_jazz_horn_responses_v001",
    "drums": "custom_jazz_drums_v001",
}
ROLE_TRACKS = {
    "melody": "alto_sax",
    "walking_bass": "double_bass",
    "piano_comping": "piano",
    "horn_responses": "trumpet_bflat",
    "drums": "drums",
}


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    output_dir = _repo_path(args.output_dir)
    checkpoint_root = _repo_path(args.checkpoint_root)
    _clean_output_dir(output_dir)

    if not _checkpoints_ready(checkpoint_root):
        if args.no_train_if_missing:
            raise SystemExit(
                f"Custom role n-gram checkpoints are incomplete under {checkpoint_root}"
            )
        _run_training(checkpoint_root=checkpoint_root, output_dir=output_dir / "training")

    config = _ai_config(checkpoint_root, output_dir=output_dir / "raw")
    registry = build_model_backend_registry(config=config, include_unavailable=True)
    project = generate_arrangement(
        GenerationSpec(ensemble="jazz_sextet", form="minor_blues_12", seed=args.seed),
        project_id="pr34-custom-role-ngram-smoke",
    )
    store = ArtifactStore(output_dir / "model_artifacts")
    importer = ArtifactImporter(artifact_store=store)
    merger = ProjectMerger()
    gate = ValidationGate()

    role_reports: dict[str, dict[str, Any]] = {}
    for index, (role, backend_id) in enumerate(ROLE_BACKENDS.items()):
        backend = registry.get(backend_id)
        target_track_id = ROLE_TRACKS[role]
        bars = [1, 2]
        result = backend.generate(
            ModelGenerationRequest(
                request_id=f"pr34_{role}",
                task="infill_bars",
                role_intent={"role": role, "density": "medium_high"},
                track_id=target_track_id,
                bars=bars,
                density="medium_high",
                seed=args.seed + index,
                metadata={"export_mode": "commercial"},
            )
        )
        records = store.store_generation_result(result, project_id=project.project_id)
        midi_record = _record_by_type(records, "midi")
        token_record = _record_by_type(records, "tokens")
        token_payload = json.loads(Path(token_record.raw_path).read_text(encoding="utf-8"))
        _assert_non_dummy_token_payload(token_payload, role=role)

        imported = importer.import_record(
            midi_record,
            project=project,
            target_track_id=target_track_id,
            target_bars=bars,
        )
        candidate = merger.merge(
            project,
            imported,
            target_track_id=target_track_id,
            target_bars=bars,
            locked_tracks=_locked_tracks(project, target_track_id),
        )
        validation = gate.validate_candidate(
            base_project=project,
            candidate_project=candidate,
            target_track_id=target_track_id,
            target_bars=bars,
            locked_tracks=_locked_tracks(project, target_track_id),
        )
        validation_path = output_dir / "validation" / f"{role}_validation.json"
        _write_json(validation_path, validation)
        candidate_path = output_dir / "candidates" / f"{role}_candidate_project.json"
        candidate.save_json(candidate_path)
        if validation["status"] not in {"pass", "pass_with_warnings"}:
            raise RuntimeError(f"Validation failed for {role}: {validation['errors']}")
        store.mark_validated(
            midi_record,
            validated_path=validation_path,
            metadata={"validation_status": validation["status"]},
        )
        store.mark_validated(
            token_record,
            validated_path=Path(token_record.raw_path),
            metadata={"validation_status": "pass", "token_payload_checked": True},
        )

        role_reports[role] = {
            "backend_id": backend_id,
            "track_id": target_track_id,
            "bars": bars,
            "confidence": result.confidence,
            "midi_artifact": midi_record.raw_path,
            "tokens_artifact": token_record.raw_path,
            "candidate_project": str(candidate_path),
            "validation_report": str(validation_path),
            "validation_status": validation["status"],
            "token_count": len(token_payload["target_tokens"]),
            "generation_source": token_payload["generation_source"],
            "model_type": token_payload["model"]["model_type"],
        }

    report = {
        "status": "ok",
        "script": "smoke_custom_role_ngram_models",
        "checkpoint_root": str(checkpoint_root),
        "output_dir": str(output_dir),
        "project_id": project.project_id,
        "roles": role_reports,
        "artifact_manifest": str(output_dir / "model_artifacts" / "artifact_manifest.json"),
    }
    _write_json(output_dir / "smoke_report.json", report)
    print(json.dumps(report, indent=2))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test custom role n-gram checkpoints through backend/import/merge gates."
    )
    parser.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=3420)
    parser.add_argument("--no-train-if-missing", action="store_true")
    return parser.parse_args(argv)


def _ai_config(checkpoint_root: Path, *, output_dir: Path) -> AIModelsConfig:
    return AIModelsConfig(
        backends={
            backend_id: BackendConfig(
                enabled=True,
                type="custom_role",
                adapter=(
                    "model_backends.custom_role.statistical_backend."
                    "StatisticalCustomRoleBackend"
                ),
                role=role,
                checkpoint_dir=str(checkpoint_dir_for_role(checkpoint_root, role)),
                commercial_use="unknown",
                tasks=["generate_track", "infill_bars", "generate_variation"],
            )
            for role, backend_id in ROLE_BACKENDS.items()
        },
        settings={"artifact_raw_dir": str(output_dir)},
    )


def _checkpoints_ready(checkpoint_root: Path) -> bool:
    for role in ROLE_BACKENDS:
        checkpoint_dir = checkpoint_dir_for_role(checkpoint_root, role)
        if any(not (checkpoint_dir / filename).exists() for filename in REQUIRED_CHECKPOINT_FILES):
            return False
    return True


def _run_training(*, checkpoint_root: Path, output_dir: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/models_pro/train_custom_role_ngram_models.py",
            "--checkpoint-root",
            str(checkpoint_root),
            "--output-dir",
            str(output_dir),
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Custom role n-gram training failed:\n"
            f"STDOUT:\n{completed.stdout[-4000:]}\nSTDERR:\n{completed.stderr[-4000:]}"
        )


def _record_by_type(records: list[Any], artifact_type: str) -> Any:
    for record in records:
        if record.artifact_type == artifact_type:
            return record
    raise RuntimeError(f"Generation result did not include {artifact_type!r} artifact")


def _assert_non_dummy_token_payload(payload: dict[str, Any], *, role: str) -> None:
    if payload.get("generation_source") != "statistical_custom_role_model":
        raise RuntimeError(f"{role} token payload did not come from the statistical backend")
    if payload.get("model", {}).get("model_type") != "custom_role_ngram":
        raise RuntimeError(f"{role} token payload has wrong model type")
    tokens = payload.get("target_tokens")
    if not isinstance(tokens, list) or len(tokens) < 4:
        raise RuntimeError(f"{role} token payload is empty or malformed")


def _locked_tracks(project: Any, target_track_id: str) -> list[str]:
    return [track.id for track in project.tracks if track.id != target_track_id]


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def _clean_output_dir(path: Path) -> None:
    outputs_root = (ROOT / "outputs").resolve()
    resolved = path.resolve()
    if path.exists():
        if resolved == outputs_root or outputs_root not in resolved.parents:
            raise RuntimeError(f"Refusing to clean output path outside outputs/: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
