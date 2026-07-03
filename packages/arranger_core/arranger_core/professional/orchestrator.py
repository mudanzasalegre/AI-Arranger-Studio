from __future__ import annotations

import json
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field

from arranger_core.ai.artifact_importer import ArtifactImporter
from arranger_core.ai.artifact_store import ArtifactStore
from arranger_core.ai.validation_gate import ValidationGate
from arranger_core.exporters import export_project, write_full_midi, write_validation_html
from arranger_core.merge.model_artifact_merger import ProjectMerger
from arranger_core.planning import LlmPlanner
from arranger_core.prompt_compiler import compile_prompt
from arranger_core.quality import ProQualityGate
from arranger_core.role_generators import generate_arrangement
from arranger_core.schema import ArrangementProject, GenerationSpec, NoteEvent, RestEvent, Track
from arranger_core.sketches import Text2MidiSketchImporter
from arranger_core.takes.manager import TakeManager
from arranger_core.takes.models import ModelArtifactRecord
from arranger_core.validators import MusicValidationError, validate_project, write_validation_json

CUSTOM_ROLE_BACKENDS = {
    "melody": "custom_jazz_melody_v001",
    "walking_bass": "custom_jazz_walking_bass_v001",
    "piano_comping": "custom_jazz_piano_comping_v001",
    "horn_responses": "custom_jazz_horn_responses_v001",
    "drums": "custom_jazz_drums_v001",
}
CUSTOM_ROLE_TARGETS = {
    "melody": (("melody",), [1, 2]),
    "walking_bass": (("walking_bass",), [1, 2]),
    "piano_comping": (("comping", "piano", "piano_comping"), [3, 4]),
    "horn_responses": (("horn_response", "horn_responses"), [9, 10]),
    "drums": (("drums",), [11, 12]),
}
MIDIGPT_DEFAULT_TARGETS = (
    {"track_role": "melody", "bars": [5, 6, 7, 8], "instruction": "melody refinement"},
    {"track_role": "horn_response", "bars": [9, 10, 11, 12], "instruction": "horn responses"},
    {"track_role": "piano_comping", "bars": [3, 4, 7, 8], "instruction": "piano comping"},
)
RATING_ORDER = {"A": 4, "B": 3, "C": 2, "D": 1}


class ProfessionalModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProfessionalGenerationOptions(ProfessionalModel):
    prompt: str
    profile: str = "pro"
    seed: int = 1234
    run_id: str | None = None
    output_root: str = "outputs/pro_benchmarks"
    ai_config_path: str = "configs/ai_models.pro.yaml"
    quality_thresholds_path: str = "configs/quality_thresholds.pro.yaml"
    export_mode: Literal["private", "commercial"] = "private"
    include_pdf: bool = False
    min_rating: Literal["A", "B", "C", "D"] = "B"
    use_llm_planner: bool = True
    use_rule_based_base: bool = True
    use_custom_role_models: bool = True
    use_midigpt_infill: bool = True
    use_text2midi_sketch: bool = False
    midigpt_targets: list[dict[str, Any]] = Field(default_factory=list)
    max_ai_attempts: int = Field(default=3, ge=1, le=3)
    clean: bool = True


class ProfessionalGenerationResult(ProfessionalModel):
    status: Literal["ok", "fail"]
    run_id: str
    output_dir: str
    project_id: str
    files: dict[str, str]
    validation: dict[str, Any]
    quality: dict[str, Any]
    model_trace: dict[str, Any]
    takes_manifest: dict[str, Any]
    export_manifest: dict[str, Any]
    steps: list[dict[str, Any]]
    fallbacks: list[dict[str, Any]]


class ProfessionalGenerationOrchestrator:
    def __init__(self, *, repo_root: str | Path | None = None) -> None:
        self.repo_root = Path(repo_root).resolve() if repo_root else _find_repo_root()

    def generate(self, options: ProfessionalGenerationOptions) -> ProfessionalGenerationResult:
        run_id = options.run_id or _default_run_id(options)
        output_dir = _repo_path(self.repo_root, options.output_root) / run_id
        if options.clean:
            _clean_child_dir(output_dir, _repo_path(self.repo_root, options.output_root))
        output_dir.mkdir(parents=True, exist_ok=True)

        artifact_store = ArtifactStore(output_dir / "model_artifacts")
        take_manager = TakeManager(output_dir)
        trace = _new_trace(run_id=run_id, options=options, output_dir=output_dir)
        steps: list[dict[str, Any]] = []
        fallbacks: list[dict[str, Any]] = []

        spec = compile_prompt(options.prompt, seed=options.seed)
        project = self._base_project(spec, run_id=run_id, options=options)
        project.save_json(output_dir / "arrangement_project.json")
        take_manager.ensure_base_take(project)
        steps.append({"step": "rule_based_base", "status": "ok"})

        if options.use_llm_planner:
            project, plan_step = self._run_planner(project, options, output_dir)
            steps.append(plan_step)
            if plan_step.get("fallback_used"):
                fallbacks.append(plan_step)

        if options.use_custom_role_models:
            project = self._run_custom_role_models(
                project,
                options=options,
                output_dir=output_dir,
                artifact_store=artifact_store,
                take_manager=take_manager,
                trace=trace,
                steps=steps,
                fallbacks=fallbacks,
            )

        if options.use_midigpt_infill:
            project = self._run_midigpt_infill(
                project,
                options=options,
                output_dir=output_dir,
                artifact_store=artifact_store,
                take_manager=take_manager,
                trace=trace,
                steps=steps,
                fallbacks=fallbacks,
            )

        if options.use_text2midi_sketch:
            steps.append(
                self._run_text2midi_sketch(
                    options=options,
                    output_dir=output_dir,
                    artifact_store=artifact_store,
                    trace=trace,
                )
            )

        validation = validate_project(project)
        project.validation_report = validation
        project.save_json(output_dir / "arrangement_project.json")
        write_validation_json(validation, output_dir / "validation_report.json")
        write_validation_html(validation, output_dir / "validation_report.html")
        trace["final_validation"] = _compact_validation(validation)

        quality_gate = ProQualityGate(
            thresholds_path=_repo_path(self.repo_root, options.quality_thresholds_path)
        )
        quality = quality_gate.evaluate(
            project,
            validation_report=validation,
            model_trace=trace,
            export_mode=options.export_mode,
            min_rating=options.min_rating,
            require_export_files=False,
        )
        _write_json(output_dir / "quality_report.json", quality)
        trace["quality"] = {
            "status": quality["status"],
            "rating": quality["rating"],
            "score": quality["score"],
        }

        export_manifest: dict[str, Any] = {}
        status: Literal["ok", "fail"] = "fail"
        if validation["status"] != "fail" and quality["status"] == "pass":
            try:
                export_manifest = export_project(
                    project,
                    output_dir,
                    include_pdf=options.include_pdf,
                    validation_policy="strict",
                    export_mode=options.export_mode,
                )
                steps.append({"step": "export", "status": "ok"})
                exported_takes_manifest = output_dir / "takes_manifest.json"
                takes_manifest = (
                    json.loads(exported_takes_manifest.read_text(encoding="utf-8"))
                    if exported_takes_manifest.exists()
                    else take_manager.list_takes(project=project)
                )
                quality = quality_gate.evaluate(
                    project,
                    validation_report=validation,
                    output_dir=output_dir,
                    export_manifest=export_manifest,
                    model_trace=trace,
                    takes_manifest=takes_manifest,
                    export_mode=options.export_mode,
                    min_rating=options.min_rating,
                    require_export_files=True,
                )
                _write_json(output_dir / "quality_report.json", quality)
                trace["quality"] = {
                    "status": quality["status"],
                    "rating": quality["rating"],
                    "score": quality["score"],
                }
                if quality["status"] == "pass":
                    status = "ok"
                    steps.append({"step": "pro_quality_gate", "status": "pass"})
                else:
                    steps.append(
                        {
                            "step": "pro_quality_gate",
                            "status": "fail",
                            "blocking_errors": quality.get("blocking_errors", []),
                        }
                    )
            except MusicValidationError as exc:
                validation = exc.report
                steps.append({"step": "export", "status": "fail", "error": str(exc)})
        else:
            steps.append(
                {
                    "step": "export",
                    "status": "skipped",
                    "reason": "validation_or_quality_failed",
                }
            )

        trace["status"] = status
        trace["steps"] = steps
        trace["fallbacks"] = fallbacks
        _write_json(output_dir / "model_trace.json", trace)
        exported_takes_manifest = output_dir / "takes_manifest.json"
        if exported_takes_manifest.exists():
            takes_manifest = json.loads(exported_takes_manifest.read_text(encoding="utf-8"))
        else:
            takes_manifest = take_manager.list_takes(project=project)
            _write_json(exported_takes_manifest, takes_manifest)
        _write_json(output_dir / "generation_summary.json", _summary_payload(
            status=status,
            run_id=run_id,
            options=options,
            validation=validation,
            quality=quality,
            export_manifest=export_manifest,
            trace=trace,
        ))
        (output_dir / "generation_summary.md").write_text(
            _summary_markdown(
                status=status,
                run_id=run_id,
                validation=validation,
                quality=quality,
                steps=steps,
                output_dir=output_dir,
            ),
            encoding="utf-8",
        )
        files = _result_files(output_dir, export_manifest)
        return ProfessionalGenerationResult(
            status=status,
            run_id=run_id,
            output_dir=str(output_dir),
            project_id=project.project_id,
            files=files,
            validation=validation,
            quality=quality,
            model_trace=trace,
            takes_manifest=takes_manifest,
            export_manifest=export_manifest,
            steps=steps,
            fallbacks=fallbacks,
        )

    def _base_project(
        self,
        spec: GenerationSpec,
        *,
        run_id: str,
        options: ProfessionalGenerationOptions,
    ) -> ArrangementProject:
        if not options.use_rule_based_base:
            raise ValueError("PR-35 requires a rule-based/retrieval base arrangement")
        project = generate_arrangement(spec, project_id=run_id)
        project.metadata = {
            **project.metadata,
            "professional_project": True,
            "professional_profile": options.profile,
            "professional_prompt": options.prompt,
            "generation_mode": "professional_orchestrator",
        }
        return project

    def _run_planner(
        self,
        project: ArrangementProject,
        options: ProfessionalGenerationOptions,
        output_dir: Path,
    ) -> tuple[ArrangementProject, dict[str, Any]]:
        provider = self._planner_provider(options)
        result = LlmPlanner(provider=provider).plan(
            prompt=options.prompt,
            project=project,
            seed=options.seed,
        )
        if result.status != "ok":
            return project, {
                "step": "llm_planner",
                "status": "fallback",
                "fallback_used": True,
                "validation": result.validation,
            }
        project.metadata = {
            **project.metadata,
            "song_plan": result.song_plan.model_dump(mode="json"),
            "active_plan_version": result.plan_version,
            "ai_planner": {
                "planner": result.planner,
                "fallback_used": result.fallback_used,
                "validation": result.validation,
                "song_plan_patch": result.song_plan_patch.model_dump(mode="json"),
            },
        }
        _write_json(output_dir / "song_plan.json", result.song_plan.model_dump(mode="json"))
        _write_json(
            output_dir / "plan_versions" / f"{result.plan_version}.json",
            result.model_dump(mode="json"),
        )
        project.save_json(output_dir / "arrangement_project.json")
        return project, {
            "step": "llm_planner",
            "status": "ok",
            "planner": result.planner,
            "plan_version": result.plan_version,
            "fallback_used": result.fallback_used,
            "validation": _compact_validation(result.validation),
        }

    def _planner_provider(self, options: ProfessionalGenerationOptions):
        try:
            from model_backends import build_model_backend_registry, load_ai_models_config
            from model_backends.errors import ModelBackendError
        except ImportError:
            return None
        try:
            config = load_ai_models_config(_repo_path(self.repo_root, options.ai_config_path))
            registry = build_model_backend_registry(
                config=config,
                include_disabled=False,
                include_unavailable=True,
            )
            return registry.get("local_llm_planner")
        except (KeyError, ModelBackendError):
            return None

    def _run_custom_role_models(
        self,
        project: ArrangementProject,
        *,
        options: ProfessionalGenerationOptions,
        output_dir: Path,
        artifact_store: ArtifactStore,
        take_manager: TakeManager,
        trace: dict[str, Any],
        steps: list[dict[str, Any]],
        fallbacks: list[dict[str, Any]],
    ) -> ArrangementProject:
        registry = self._backend_registry(options, artifact_store.root / "raw")
        records = _registry_records(registry)
        for role, backend_id in CUSTOM_ROLE_BACKENDS.items():
            track = _track_for_custom_role(project, role)
            if track is None:
                steps.append({"step": "custom_role_model", "role": role, "status": "skipped"})
                continue
            bars = _valid_bars(project, CUSTOM_ROLE_TARGETS[role][1])
            if not bars:
                continue
            step = self._apply_model_infill(
                project,
                backend_id=backend_id,
                backend_role=role,
                target_track=track,
                bars=bars,
                instruction=f"custom role {role} refinement",
                density="medium",
                temperature=0.65,
                seed=options.seed + 10 + len(steps),
                options=options,
                output_dir=output_dir,
                artifact_store=artifact_store,
                take_manager=take_manager,
                registry=registry,
                registry_records=records,
                trace=trace,
            )
            steps.append(step)
            if step["status"] == "accepted":
                project = ArrangementProject.load_json(output_dir / "arrangement_project.json")
            elif step.get("fallback"):
                fallbacks.append(step)
        return project

    def _run_midigpt_infill(
        self,
        project: ArrangementProject,
        *,
        options: ProfessionalGenerationOptions,
        output_dir: Path,
        artifact_store: ArtifactStore,
        take_manager: TakeManager,
        trace: dict[str, Any],
        steps: list[dict[str, Any]],
        fallbacks: list[dict[str, Any]],
    ) -> ArrangementProject:
        registry = self._backend_registry(options, artifact_store.root / "raw")
        records = _registry_records(registry)
        targets = options.midigpt_targets or list(MIDIGPT_DEFAULT_TARGETS)
        for target in targets:
            track = _track_for_midigpt_target(project, target)
            if track is None or track.role in {"walking_bass", "drums"}:
                step = {
                    "step": "midigpt_infill",
                    "status": "skipped",
                    "target": target,
                    "reason": "missing_or_disallowed_target",
                }
                steps.append(step)
                continue
            bars = _valid_bars(project, [int(bar) for bar in target.get("bars", [])])
            if not bars:
                continue
            accepted = False
            for attempt in range(1, options.max_ai_attempts + 1):
                if attempt == 3:
                    step = {
                        "step": "midigpt_infill",
                        "status": "fallback",
                        "fallback": "rule_based_existing_arrangement",
                        "attempt": attempt,
                        "track_id": track.id,
                        "bars": bars,
                    }
                    steps.append(step)
                    fallbacks.append(step)
                    break
                step = self._apply_model_infill(
                    project,
                    backend_id="midigpt",
                    backend_role=track.role,
                    target_track=track,
                    bars=bars,
                    instruction=str(target.get("instruction") or "MIDI-GPT selective infill"),
                    density=str(
                        target.get("density") or ("medium_high" if attempt == 1 else "medium")
                    ),
                    temperature=float(
                        target.get("temperature") or (0.95 if attempt == 1 else 0.65)
                    ),
                    seed=options.seed + 100 + len(steps) + attempt,
                    options=options,
                    output_dir=output_dir,
                    artifact_store=artifact_store,
                    take_manager=take_manager,
                    registry=registry,
                    registry_records=records,
                    trace=trace,
                    step_name="midigpt_infill",
                    attempt=attempt,
                )
                steps.append(step)
                if step["status"] == "accepted":
                    project = ArrangementProject.load_json(output_dir / "arrangement_project.json")
                    accepted = True
                    break
                if step.get("fallback"):
                    fallbacks.append(step)
            if not accepted:
                continue
        return project

    def _run_text2midi_sketch(
        self,
        *,
        options: ProfessionalGenerationOptions,
        output_dir: Path,
        artifact_store: ArtifactStore,
        trace: dict[str, Any],
    ) -> dict[str, Any]:
        registry = self._backend_registry(options, artifact_store.root / "raw")
        records = _registry_records(registry)
        if not _backend_ready(records, "text2midi"):
            return {
                "step": "text2midi_sketch",
                "status": "skipped",
                "reason": _backend_error(records, "text2midi"),
            }
        try:
            from model_backends import ModelGenerationRequest
        except ImportError as exc:
            return {"step": "text2midi_sketch", "status": "skipped", "reason": str(exc)}
        try:
            backend = registry.get("text2midi")
            result = backend.generate(
                ModelGenerationRequest(
                    request_id=f"sketch_{uuid4().hex[:10]}",
                    task="generate_full_sketch",
                    prompt=options.prompt,
                    instruction=options.prompt,
                    seed=options.seed,
                    metadata={"sketch_only": True, "sketch_reference": True},
                )
            )
            sketch_id = f"{output_dir.name}_sketch"
            stored = artifact_store.store_generation_result(result, project_id=sketch_id)
            midi_record = _record_by_type(stored, "midi")
            imported = Text2MidiSketchImporter().import_record(
                midi_record,
                prompt=options.prompt,
                seed=options.seed,
                sketch_id=sketch_id,
            )
            sketch_dir = output_dir / "sketch_reference"
            sketch_project_path = imported.project.save_json(
                sketch_dir / "arrangement_project.json"
            )
            validation_path = write_validation_json(
                imported.validation_report,
                sketch_dir / "validation_report.json",
            )
            for record in stored:
                artifact_store.mark_validated(
                    artifact_store.get(record.artifact_id),
                    validated_path=validation_path,
                    metadata={"sketch_reference": True, "validation": imported.validation_report},
                )
            step = {
                "step": "text2midi_sketch",
                "status": imported.status,
                "artifact_ids": [record.artifact_id for record in stored],
                "project_path": str(sketch_project_path),
                "validation": _compact_validation(imported.validation_report),
                "used_in_final": False,
            }
            trace["sketch_reference"] = step
            return step
        except Exception as exc:
            return {
                "step": "text2midi_sketch",
                "status": "fallback",
                "fallback": "sketch_reference_unavailable",
                "error": str(exc),
                "used_in_final": False,
            }

    def _apply_model_infill(
        self,
        project: ArrangementProject,
        *,
        backend_id: str,
        backend_role: str,
        target_track: Track,
        bars: list[int],
        instruction: str,
        density: str,
        temperature: float,
        seed: int,
        options: ProfessionalGenerationOptions,
        output_dir: Path,
        artifact_store: ArtifactStore,
        take_manager: TakeManager,
        registry: Any,
        registry_records: dict[str, dict[str, Any]],
        trace: dict[str, Any],
        step_name: str = "custom_role_model",
        attempt: int = 1,
    ) -> dict[str, Any]:
        if not _backend_ready(registry_records, backend_id):
            return {
                "step": step_name,
                "status": "fallback",
                "fallback": "rule_based_existing_arrangement",
                "backend_id": backend_id,
                "role": backend_role,
                "track_id": target_track.id,
                "bars": bars,
                "attempt": attempt,
                "reason": _backend_error(registry_records, backend_id),
            }
        try:
            from model_backends import ModelGenerationRequest
            backend = registry.get(backend_id)
            context_path = _write_context_midi(
                project,
                output_dir=output_dir,
                backend_id=backend_id,
            )
            model_request = ModelGenerationRequest(
                request_id=f"{step_name}_{backend_id}_{uuid4().hex[:10]}",
                task="infill_bars",
                project=project.model_dump(mode="json"),
                song_plan=_song_plan(project),
                section_plan=_section_context(_song_plan(project), bars),
                phrase_plan=_phrase_context(_song_plan(project), bars),
                groove_map=(_song_plan(project) or {}).get("groove_map"),
                role_intent={
                    "role": backend_role,
                    "project_track_role": target_track.role,
                    "instrument": target_track.instrument,
                    "track_id": target_track.id,
                    "bars": bars,
                    "density": density,
                    "instruction": instruction,
                },
                track_id=target_track.id,
                bars=bars,
                locked_tracks=_locked_tracks(project, target_track.id),
                instruction=instruction,
                density=density,  # type: ignore[arg-type]
                temperature=temperature,
                seed=seed,
                metadata={
                    "context_midi_path": str(context_path),
                    "target_role": backend_role,
                    "target_instrument": target_track.instrument,
                    "export_mode": options.export_mode,
                },
            )
            generation_result = _generation_result_with_existing_artifacts(
                backend.generate(model_request)
            )
            stored = artifact_store.store_generation_result(
                generation_result,
                project_id=project.project_id,
            )
            midi_record = _record_by_type(stored, "midi")
            imported = ArtifactImporter(artifact_store=artifact_store).import_record(
                midi_record,
                project=project,
                target_track_id=target_track.id,
                target_bars=bars,
            )
            candidate = ProjectMerger().merge(
                project,
                imported,
                target_track_id=target_track.id,
                target_bars=bars,
                locked_tracks=_locked_tracks(project, target_track.id),
            )
            validation = ValidationGate().validate_candidate(
                base_project=project,
                candidate_project=candidate,
                target_track_id=target_track.id,
                target_bars=bars,
                locked_tracks=_locked_tracks(project, target_track.id),
            )
            if validation["status"] == "fail":
                for record in stored:
                    artifact_store.mark_rejected(
                        artifact_store.get(record.artifact_id),
                        reason="validation_failed",
                        metadata={"validation": validation},
                    )
                return _fallback_step(
                    step_name,
                    backend_id=backend_id,
                    role=backend_role,
                    track_id=target_track.id,
                    bars=bars,
                    attempt=attempt,
                    error="validation_failed",
                )
            validation_path = output_dir / "take_validations" / f"{uuid4().hex[:12]}.json"
            _write_json(validation_path, validation)
            validated_records = [
                artifact_store.mark_validated(
                    artifact_store.get(record.artifact_id),
                    validated_path=(
                        validation_path
                        if record.artifact_type == "midi"
                        else Path(record.raw_path)
                    ),
                    metadata={"validation": validation},
                )
                for record in stored
            ]
            take = take_manager.create_pending_take(
                base_project=project,
                candidate_project=candidate,
                artifact_records=validated_records,
                validation_report=validation,
                track_id=target_track.id,
                bars=bars,
                instruction=instruction,
                seed=seed,
                metadata={
                    "model_trace": {
                        "backend": generation_result.backend_id,
                        "backend_id": generation_result.backend_id,
                        "task": generation_result.task,
                        "track_id": target_track.id,
                        "bars": bars,
                        "instruction": instruction,
                        "density": density,
                        "temperature": temperature,
                        "seed": seed,
                        "validation_status": validation["status"],
                        "commercial_use": _backend_commercial_use(registry_records, backend_id),
                    }
                },
            )
            accepted_take, accepted_project = take_manager.accept_take(take.take_id)
            accepted_project.save_json(output_dir / "arrangement_project.json")
            trace["model_artifacts"].append(
                {
                    "take_id": accepted_take.take_id,
                    "backend_id": generation_result.backend_id,
                    "task": generation_result.task,
                    "role": backend_role,
                    "track_id": target_track.id,
                    "bars": bars,
                    "artifact_ids": [record.artifact_id for record in validated_records],
                    "validation_status": validation["status"],
                    "commercial_use": _backend_commercial_use(registry_records, backend_id),
                    "attempt": attempt,
                }
            )
            return {
                "step": step_name,
                "status": "accepted",
                "backend_id": generation_result.backend_id,
                "role": backend_role,
                "track_id": target_track.id,
                "bars": bars,
                "take_id": accepted_take.take_id,
                "artifact_ids": [record.artifact_id for record in validated_records],
                "validation": _compact_validation(validation),
                "attempt": attempt,
            }
        except Exception as exc:
            return _fallback_step(
                step_name,
                backend_id=backend_id,
                role=backend_role,
                track_id=target_track.id,
                bars=bars,
                attempt=attempt,
                error=str(exc),
            )

    def _backend_registry(self, options: ProfessionalGenerationOptions, artifact_raw_dir: Path):
        from model_backends import build_model_backend_registry, load_ai_models_config

        config = load_ai_models_config(_repo_path(self.repo_root, options.ai_config_path))
        config = config.model_copy(
            update={
                "settings": {
                    **config.settings,
                    "artifact_raw_dir": str(artifact_raw_dir),
                }
            }
        )
        return build_model_backend_registry(
            config=config,
            include_disabled=False,
            include_unavailable=True,
        )


def _find_repo_root() -> Path:
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "packages").exists():
            return candidate
    return current


def _repo_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def _default_run_id(options: ProfessionalGenerationOptions) -> str:
    profile = _safe_stem(options.profile)
    return f"pro_{profile}_{options.seed}"


def _safe_stem(value: str) -> str:
    text = "".join(char.lower() if char.isalnum() else "_" for char in value)
    return "_".join(part for part in text.split("_") if part) or "run"


def _clean_child_dir(path: Path, parent: Path) -> None:
    resolved_path = path.resolve()
    resolved_parent = parent.resolve()
    if resolved_path == resolved_parent or resolved_parent not in resolved_path.parents:
        raise RuntimeError(f"Refusing to clean path outside parent: {resolved_path}")
    if resolved_path.exists():
        shutil.rmtree(resolved_path)
    resolved_path.mkdir(parents=True, exist_ok=True)


def _new_trace(
    *,
    run_id: str,
    options: ProfessionalGenerationOptions,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "0.1.0",
        "trace_type": "professional_generation_orchestrator",
        "run_id": run_id,
        "status": "running",
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output_dir),
        "profile": options.profile,
        "prompt": options.prompt,
        "seed": options.seed,
        "export_mode": options.export_mode,
        "text2midi_policy": "sketch_reference",
        "model_artifacts": [],
    }


def _load_thresholds(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _default_thresholds()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else _default_thresholds()


def _default_thresholds() -> dict[str, Any]:
    return {
        "global": {"max_blocking_errors": 0, "min_tracks": 3, "min_note_events": 80},
        "ratings": {
            "A": {"min_score": 0.88},
            "B": {"min_score": 0.72},
            "C": {"min_score": 0.55},
            "D": {"min_score": 0.0},
        },
    }


def _score_quality(
    project: ArrangementProject,
    *,
    validation: dict[str, Any],
    thresholds: dict[str, Any],
    min_rating: str,
    model_trace: dict[str, Any],
) -> dict[str, Any]:
    metrics = _quality_metrics(project, validation=validation, model_trace=model_trace)
    global_thresholds = thresholds.get("global", {})
    score = 1.0
    errors: list[str] = []
    warnings: list[str] = []

    if metrics["validation"]["errors"] > int(global_thresholds.get("max_blocking_errors", 0)):
        score -= 0.45
        errors.append("blocking_validation_errors")
    if metrics["project"]["tracks"] < int(global_thresholds.get("min_tracks", 3)):
        score -= 0.2
        errors.append("too_few_tracks")
    if metrics["project"]["note_events"] < int(global_thresholds.get("min_note_events", 80)):
        score -= 0.2
        warnings.append("low_note_count")
    if metrics["model_trace"]["text2midi_used_in_final"]:
        score -= 0.35
        errors.append("text2midi_sketch_used_in_final")

    for track in metrics["tracks"]:
        if track["note_count"] == 0:
            score -= 0.05
            warnings.append(f"empty_track:{track['track_id']}")
        if track["role"] in {"melody", "horn_response"} and track["breath_rest_count"] < 1:
            score -= 0.03
            warnings.append(f"low_breathing:{track['track_id']}")

    score = max(0.0, min(1.0, score))
    rating = _rating_for_score(score, thresholds.get("ratings", {}))
    if RATING_ORDER[rating] < RATING_ORDER[min_rating]:
        errors.append(f"rating_below_minimum:{rating}<{min_rating}")
    return {
        "schema_version": "0.1.0",
        "status": "pass" if not errors else "fail",
        "score": round(score, 3),
        "rating": rating,
        "min_rating": min_rating,
        "errors": errors,
        "warnings": warnings,
        "metrics": metrics,
    }


def _quality_metrics(
    project: ArrangementProject,
    *,
    validation: dict[str, Any],
    model_trace: dict[str, Any],
) -> dict[str, Any]:
    tracks = [
        _track_quality_metrics(track, project_bars=max(1, project.bar_count))
        for track in project.tracks
    ]
    return {
        "project": {
            "project_id": project.project_id,
            "bars": project.bar_count,
            "tracks": len(project.tracks),
            "note_events": sum(track["note_count"] for track in tracks),
        },
        "validation": {
            "status": validation.get("status"),
            "errors": len(validation.get("errors", [])),
            "warnings": len(validation.get("warnings", [])),
        },
        "tracks": tracks,
        "model_trace": {
            "model_artifact_count": len(model_trace.get("model_artifacts", [])),
            "backends": sorted(
                {
                    str(item.get("backend_id"))
                    for item in model_trace.get("model_artifacts", [])
                    if isinstance(item, dict) and item.get("backend_id")
                }
            ),
            "text2midi_used_in_final": False,
        },
    }


def _track_quality_metrics(track: Track, *, project_bars: int) -> dict[str, Any]:
    notes = [
        event
        for bar in track.bars
        for event in bar.events
        if isinstance(event, NoteEvent)
    ]
    rests = [
        event
        for bar in track.bars
        for event in bar.events
        if isinstance(event, RestEvent)
    ]
    active_bars = {
        bar.number
        for bar in track.bars
        if any(isinstance(event, NoteEvent) for event in bar.events)
    }
    return {
        "track_id": track.id,
        "role": track.role,
        "instrument": track.instrument,
        "note_count": len(notes),
        "active_bar_ratio": round(len(active_bars) / project_bars, 3),
        "breath_rest_count": sum(1 for rest in rests if rest.duration >= 0.5),
    }


def _rating_for_score(score: float, ratings: dict[str, Any]) -> str:
    for rating in ("A", "B", "C", "D"):
        threshold = float((ratings.get(rating) or {}).get("min_score", 0.0))
        if score >= threshold:
            return rating
    return "D"


def _registry_records(registry: Any) -> dict[str, dict[str, Any]]:
    return {
        str(record["id"]): record
        for record in registry.list()
        if isinstance(record, dict) and record.get("id")
    }


def _backend_ready(records: dict[str, dict[str, Any]], backend_id: str) -> bool:
    record = records.get(backend_id)
    return bool(record and record.get("enabled") and record.get("status") == "available")


def _backend_error(records: dict[str, dict[str, Any]], backend_id: str) -> str:
    record = records.get(backend_id)
    if not record:
        return "backend_not_registered"
    return str(record.get("error") or record.get("status") or "backend_unavailable")


def _backend_commercial_use(records: dict[str, dict[str, Any]], backend_id: str) -> str:
    record = records.get(backend_id) or {}
    return str(record.get("commercial_use") or "unknown")


def _track_for_custom_role(project: ArrangementProject, role: str) -> Track | None:
    roles, _bars = CUSTOM_ROLE_TARGETS[role]
    return next((track for track in project.tracks if track.role in roles), None)


def _track_for_midigpt_target(project: ArrangementProject, target: dict[str, Any]) -> Track | None:
    track_id = target.get("track_id")
    if isinstance(track_id, str):
        match = next((track for track in project.tracks if track.id == track_id), None)
        if match is not None:
            return match
    track_role = str(target.get("track_role") or "")
    role_aliases = {
        "piano_comping": {"comping", "piano", "piano_comping"},
        "horn_responses": {"horn_response", "horn_responses"},
        "horn_response": {"horn_response", "horn_responses"},
    }
    roles = role_aliases.get(track_role, {track_role})
    return next((track for track in project.tracks if track.role in roles), None)


def _valid_bars(project: ArrangementProject, bars: list[int]) -> list[int]:
    return sorted({bar for bar in bars if 1 <= bar <= project.bar_count})


def _locked_tracks(project: ArrangementProject, target_track_id: str) -> list[str]:
    return [track.id for track in project.tracks if track.id != target_track_id]


def _song_plan(project: ArrangementProject) -> dict[str, Any] | None:
    song_plan = project.metadata.get("song_plan")
    return song_plan if isinstance(song_plan, dict) else None


def _section_context(song_plan: dict[str, Any] | None, bars: list[int]) -> dict[str, Any] | None:
    return _plan_context(song_plan, bars, key="sections")


def _phrase_context(song_plan: dict[str, Any] | None, bars: list[int]) -> dict[str, Any] | None:
    return _plan_context(song_plan, bars, key="phrases")


def _plan_context(
    song_plan: dict[str, Any] | None,
    bars: list[int],
    *,
    key: str,
) -> dict[str, Any] | None:
    if not song_plan:
        return None
    items = song_plan.get(key)
    if not isinstance(items, list):
        return None
    target = set(bars)
    for item in items:
        if not isinstance(item, dict):
            continue
        start = int(item.get("start_bar", 0))
        end = int(item.get("end_bar", 0))
        if target & set(range(start, end + 1)):
            return item
    return None


def _write_context_midi(project: ArrangementProject, *, output_dir: Path, backend_id: str) -> Path:
    context_path = output_dir / "model_contexts" / f"{backend_id}_{uuid4().hex[:10]}.mid"
    write_full_midi(project, context_path)
    return context_path


def _record_by_type(records: list[ModelArtifactRecord], artifact_type: str) -> ModelArtifactRecord:
    for record in records:
        if record.artifact_type == artifact_type:
            return record
    raise ValueError(f"Generation result did not include {artifact_type!r} artifact")


def _generation_result_with_existing_artifacts(result: Any) -> Any:
    existing = [artifact for artifact in result.artifacts if Path(artifact.path).exists()]
    if not existing or len(existing) == len(result.artifacts):
        return result
    missing = [
        artifact.path
        for artifact in result.artifacts
        if not Path(artifact.path).exists()
    ]
    return result.model_copy(
        update={
            "artifacts": existing,
            "warnings": [
                *result.warnings,
                f"missing_auxiliary_artifacts:{len(missing)}",
            ],
            "raw_metadata": {
                **result.raw_metadata,
                "missing_auxiliary_artifacts": missing,
            },
        }
    )


def _fallback_step(
    step: str,
    *,
    backend_id: str,
    role: str,
    track_id: str,
    bars: list[int],
    attempt: int,
    error: str,
) -> dict[str, Any]:
    return {
        "step": step,
        "status": "fallback",
        "fallback": "rule_based_existing_arrangement",
        "backend_id": backend_id,
        "role": role,
        "track_id": track_id,
        "bars": bars,
        "attempt": attempt,
        "error": error,
    }


def _compact_validation(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "errors": len(report.get("errors", [])),
        "warnings": len(report.get("warnings", [])),
    }


def _result_files(output_dir: Path, export_manifest: dict[str, Any]) -> dict[str, str]:
    names = [
        "arrangement_project.json",
        "full_arrangement.mid",
        "full_score.musicxml",
        "validation_report.json",
        "quality_report.json",
        "model_trace.json",
        "takes_manifest.json",
        "generation_summary.md",
    ]
    files = {name: str(output_dir / name) for name in names if (output_dir / name).exists()}
    if export_manifest:
        files["export_manifest.json"] = str(output_dir / "export_manifest.json")
    return files


def _summary_payload(
    *,
    status: str,
    run_id: str,
    options: ProfessionalGenerationOptions,
    validation: dict[str, Any],
    quality: dict[str, Any],
    export_manifest: dict[str, Any],
    trace: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": status,
        "run_id": run_id,
        "profile": options.profile,
        "prompt": options.prompt,
        "seed": options.seed,
        "validation": _compact_validation(validation),
        "quality": {
            "status": quality.get("status"),
            "rating": quality.get("rating"),
            "score": quality.get("score"),
        },
        "exported": bool(export_manifest),
        "model_artifact_count": len(trace.get("model_artifacts", [])),
        "backends": sorted(
            Counter(
                str(item.get("backend_id"))
                for item in trace.get("model_artifacts", [])
                if isinstance(item, dict) and item.get("backend_id")
            )
        ),
    }


def _summary_markdown(
    *,
    status: str,
    run_id: str,
    validation: dict[str, Any],
    quality: dict[str, Any],
    steps: list[dict[str, Any]],
    output_dir: Path,
) -> str:
    lines = [
        "# Professional Generation Summary",
        "",
        f"Run: `{run_id}`",
        f"Status: `{status}`",
        f"Validation: `{validation.get('status')}`",
        f"Quality: `{quality.get('rating')}` / `{quality.get('score')}`",
        f"Output: `{output_dir}`",
        "",
        "## Steps",
        "",
    ]
    for step in steps:
        label = step.get("step")
        state = step.get("status")
        backend = step.get("backend_id")
        track = step.get("track_id")
        lines.append(f"- `{label}`: `{state}`" + (f" `{backend}` `{track}`" if backend else ""))
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
