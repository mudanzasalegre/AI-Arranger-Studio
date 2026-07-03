from __future__ import annotations

import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from model_backends.artifact import artifact_path, artifact_record
from model_backends.base import ModelCapabilities, ModelGenerationRequest, ModelGenerationResult
from model_backends.errors import (
    ModelBackendUnavailableError,
    ModelGenerationError,
    UnsupportedModelTaskError,
)

ROOT = Path(__file__).resolve().parents[4]
_REQUIRED_MODULES = (
    "torch",
    "transformers",
    "sentencepiece",
    "einops",
    "jsonlines",
    "accelerate",
    "st_moe_pytorch",
)
_ENGINE_MODULES = (
    "text2midi",
    "text2midi.inference",
    "text_to_midi",
    "text_to_midi.inference",
)
_ENGINE_FUNCTIONS = ("generate_text2midi", "generate_midi", "generate")


class Text2MidiBackend:
    backend_id = "text2midi"
    backend_version = "0.3.0"
    unavailable_reason = "Text2MIDI is not installed/configured"
    capabilities = ModelCapabilities(
        symbolic_midi=True,
        multitrack=False,
        bar_infill=False,
        track_generation=False,
        text_prompt=True,
        json_planning=False,
        token_output=False,
        supports_training=True,
        commercial_use="review_required",
    )

    def __init__(
        self,
        *,
        backend_id: str | None = None,
        model_name: str = "text2midi",
        repo_dir: str | Path | None = None,
        checkpoint_dir: str | Path | None = None,
        model_file: str | Path | None = None,
        tokenizer_file: str | Path | None = None,
        flan_tokenizer: str = "google/flan-t5-base",
        wrapper_path: str | Path = "scripts/models/run_text2midi_inference.py",
        output_dir: str | Path = "outputs/model_artifacts/raw",
        install_hint: str | None = None,
        execution_mode: str = "subprocess_or_worker",
        device: str | None = None,
        max_len: int = 2000,
        temperature: float | None = None,
        timeout_seconds: int = 900,
        worker_url: str | None = None,
        **_: Any,
    ) -> None:
        if backend_id is not None:
            self.backend_id = backend_id
        self.model_name = model_name
        self.repo_dir = _resolve_project_path(
            repo_dir or os.environ.get("TEXT2MIDI_REPO_DIR", "models/external_repos/text2midi")
        )
        self.checkpoint_dir = _resolve_project_path(
            checkpoint_dir
            or os.environ.get("TEXT2MIDI_CHECKPOINT_DIR", "models/checkpoints/text2midi")
        )
        self.model_file = str(
            model_file
            or os.environ.get("TEXT2MIDI_MODEL_FILE", "pytorch_model.bin")
        )
        self.tokenizer_file = str(
            tokenizer_file
            or os.environ.get("TEXT2MIDI_TOKENIZER_FILE", "vocab_remi.pkl")
        )
        self.flan_tokenizer = flan_tokenizer
        self.wrapper_path = _resolve_project_path(wrapper_path)
        self.output_dir = _resolve_project_path(output_dir)
        self.install_hint = install_hint or (
            "Clone https://github.com/AMAAI-Lab/text2midi into "
            "models/external_repos/text2midi and download pytorch_model.bin/vocab_remi.pkl"
        )
        self.execution_mode = execution_mode
        self.device = device or os.environ.get("AI_DEVICE", "auto")
        self.max_len = int(max_len)
        self.temperature = temperature
        self.timeout_seconds = int(timeout_seconds)
        self.worker_url = (
            worker_url
            or os.environ.get("TEXT2MIDI_WORKER_URL")
            or os.environ.get("AI_MODEL_WORKER_BASE_URL")
        )

    def is_available(self) -> bool:
        if self._importable_engine_available():
            self.unavailable_reason = ""
            return True
        if self._subprocess_available():
            self.unavailable_reason = ""
            return True
        if self._worker_configured():
            self.unavailable_reason = ""
            return True
        missing = self._missing_requirements()
        self.unavailable_reason = (
            "Text2MIDI is not installed/configured. Missing: "
            + ", ".join(missing)
        )
        return False

    def generate(self, request: ModelGenerationRequest) -> ModelGenerationResult:
        if request.task != "generate_full_sketch":
            raise UnsupportedModelTaskError("Text2MIDI only supports generate_full_sketch")

        output_path = artifact_path(
            self.output_dir,
            backend_id=self.backend_id,
            task=request.task,
            seed=request.seed,
            suffix=".mid",
            request_id=request.request_id,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        prompt = request.prompt or request.instruction or ""
        temperature = request.temperature if self.temperature is None else self.temperature
        max_len = int(request.metadata.get("max_len") or self.max_len)
        device = str(request.metadata.get("device") or self.device)
        errors: list[str] = []

        generation_metadata: dict[str, Any] | None = None
        if self._importable_engine_available():
            try:
                generation_metadata = self._generate_with_importable_engine(
                    prompt=prompt,
                    output_path=output_path,
                    seed=request.seed,
                    temperature=temperature,
                    max_len=max_len,
                    device=device,
                )
            except Exception as exc:
                errors.append(f"importable_engine: {exc}")

        if generation_metadata is None and self._subprocess_enabled():
            missing = self._missing_requirements()
            if not missing:
                try:
                    generation_metadata = self._generate_with_subprocess(
                        prompt=prompt,
                        output_path=output_path,
                        seed=request.seed,
                        temperature=temperature,
                        max_len=max_len,
                        device=device,
                    )
                except Exception as exc:
                    errors.append(f"subprocess: {exc}")
            else:
                errors.append("subprocess unavailable; missing: " + ", ".join(missing))

        if generation_metadata is None and self._worker_configured():
            try:
                generation_metadata = self._generate_with_worker(
                    prompt=prompt,
                    output_path=output_path,
                    seed=request.seed,
                    temperature=temperature,
                    max_len=max_len,
                    device=device,
                )
            except Exception as exc:
                errors.append(f"worker: {exc}")

        if generation_metadata is None:
            reason = "; ".join(errors) or self.unavailable_reason
            if not self.is_available():
                raise ModelBackendUnavailableError(f"{reason}. Install hint: {self.install_hint}")
            raise ModelGenerationError(f"Text2MIDI generation failed. {reason}")

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise ModelGenerationError(
                "Text2MIDI completed without producing a MIDI sketch."
                f" mode={generation_metadata.get('execution_mode')}"
            )

        metadata = {
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "model_name": self.model_name,
            "prompt": prompt,
            "seed": request.seed,
            "temperature": temperature,
            "max_len": max_len,
            "device": device,
            "repo_dir": str(self.repo_dir),
            "checkpoint_dir": str(self.checkpoint_dir),
            "flan_tokenizer": self.flan_tokenizer,
            "sketch_only": True,
            "sketch_status_contract": [
                "sketch_ready",
                "sketch_uncertain",
                "sketch_rejected",
            ],
            **generation_metadata,
        }
        return ModelGenerationResult(
            backend_id=self.backend_id,
            task=request.task,
            artifacts=[artifact_record("midi", output_path, metadata=metadata)],
            confidence=None,
            raw_metadata={
                "backend_version": self.backend_version,
                "model_name": self.model_name,
                "reproducibility": "best_effort",
                "sketch_only": True,
                **generation_metadata,
            },
        )

    def _generate_with_importable_engine(
        self,
        *,
        prompt: str,
        output_path: Path,
        seed: int | None,
        temperature: float,
        max_len: int,
        device: str,
    ) -> dict[str, Any]:
        module, function_name = self._load_importable_engine()
        function = getattr(module, function_name)
        result = _call_supported(
            function,
            prompt=prompt,
            output_path=str(output_path),
            output=str(output_path),
            checkpoint_dir=str(self.checkpoint_dir),
            model_path=str(_resolve_checkpoint_file(self.checkpoint_dir, self.model_file)),
            tokenizer_path=str(_resolve_checkpoint_file(self.checkpoint_dir, self.tokenizer_file)),
            flan_tokenizer=self.flan_tokenizer,
            seed=seed,
            temperature=temperature,
            max_len=max_len,
            device=device,
        )
        _materialize_engine_result(result, output_path)
        return {
            "execution_mode": "importable_engine",
            "engine_module": module.__name__,
            "engine_function": function_name,
        }

    def _generate_with_subprocess(
        self,
        *,
        prompt: str,
        output_path: Path,
        seed: int | None,
        temperature: float,
        max_len: int,
        device: str,
    ) -> dict[str, Any]:
        summary_path = output_path.with_name(f"{output_path.stem}.summary.json")
        cmd = [
            sys.executable,
            str(self.wrapper_path),
            "--repo-dir",
            str(self.repo_dir),
            "--checkpoint-dir",
            str(self.checkpoint_dir),
            "--output",
            str(output_path),
            "--prompt",
            prompt,
            "--temperature",
            str(temperature),
            "--max-len",
            str(max_len),
            "--device",
            device,
            "--flan-tokenizer",
            self.flan_tokenizer,
            "--model-file",
            self.model_file,
            "--tokenizer-file",
            self.tokenizer_file,
            "--summary",
            str(summary_path),
        ]
        if seed is not None:
            cmd.extend(["--seed", str(seed)])

        completed = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
            timeout=self.timeout_seconds,
        )
        wrapper_summary = _load_json(summary_path)
        if completed.returncode != 0:
            raise ModelGenerationError(
                "Text2MIDI subprocess failed"
                f" (exit {completed.returncode})."
                f" summary={_compact_json(wrapper_summary)}"
                f" stdout={_tail(completed.stdout)!r}"
                f" stderr={_tail(completed.stderr)!r}"
            )
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise ModelGenerationError(
                "Text2MIDI subprocess completed without a MIDI sketch."
                f" summary={_compact_json(wrapper_summary)}"
                f" stdout={_tail(completed.stdout)!r}"
                f" stderr={_tail(completed.stderr)!r}"
            )
        return {
            "execution_mode": "subprocess",
            "wrapper_path": str(self.wrapper_path),
            "wrapper_summary_path": str(summary_path),
            "wrapper_summary": wrapper_summary,
            "subprocess_returncode": completed.returncode,
        }

    def _generate_with_worker(
        self,
        *,
        prompt: str,
        output_path: Path,
        seed: int | None,
        temperature: float,
        max_len: int,
        device: str,
    ) -> dict[str, Any]:
        if importlib.util.find_spec("httpx") is None:
            raise ModelBackendUnavailableError("Text2MIDI worker fallback requires httpx")
        import httpx

        base_url = str(self.worker_url).rstrip("/")
        payload = {
            "prompt": prompt,
            "output_path": str(output_path),
            "seed": seed,
            "temperature": temperature,
            "max_len": max_len,
            "device": device,
            "repo_dir": str(self.repo_dir),
            "checkpoint_dir": str(self.checkpoint_dir),
            "model_file": self.model_file,
            "tokenizer_file": self.tokenizer_file,
            "flan_tokenizer": self.flan_tokenizer,
        }
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(f"{base_url}/v1/text2midi/sketch", json=payload)
            response.raise_for_status()
            data = response.json()
        worker_output = Path(str(data.get("output_path") or output_path))
        if worker_output.exists() and worker_output.resolve() != output_path.resolve():
            shutil.copy2(worker_output, output_path)
        return {
            "execution_mode": "worker",
            "worker_url": base_url,
            "worker_response": data,
        }

    def _importable_engine_available(self) -> bool:
        return any(_find_spec(module_name) is not None for module_name in _ENGINE_MODULES)

    def _load_importable_engine(self) -> tuple[Any, str]:
        for module_name in _ENGINE_MODULES:
            if _find_spec(module_name) is None:
                continue
            module = importlib.import_module(module_name)
            for function_name in _ENGINE_FUNCTIONS:
                if callable(getattr(module, function_name, None)):
                    return module, function_name
        raise ModelBackendUnavailableError("No importable Text2MIDI generation function found")

    def _subprocess_enabled(self) -> bool:
        return self.execution_mode in {
            "subprocess",
            "subprocess_or_worker",
            "importable_or_subprocess",
            "importable_subprocess_worker",
        }

    def _subprocess_available(self) -> bool:
        return not self._missing_requirements()

    def _worker_configured(self) -> bool:
        return bool(self.worker_url) and self.execution_mode in {
            "worker",
            "subprocess_or_worker",
            "importable_subprocess_worker",
        }

    def _missing_requirements(self) -> list[str]:
        missing: list[str] = []
        if not self.wrapper_path.exists():
            missing.append(str(self.wrapper_path))
        if not self.repo_dir.exists():
            missing.append(str(self.repo_dir))
        if not (self.repo_dir / "model" / "transformer_model.py").exists():
            missing.append(str(self.repo_dir / "model" / "transformer_model.py"))
        model_path = _resolve_checkpoint_file(self.checkpoint_dir, self.model_file)
        tokenizer_path = _resolve_checkpoint_file(self.checkpoint_dir, self.tokenizer_file)
        if not model_path.exists():
            missing.append(str(model_path))
        if not tokenizer_path.exists():
            missing.append(str(tokenizer_path))
        for module_name in _REQUIRED_MODULES:
            if _find_spec(module_name) is None:
                missing.append(f"python module {module_name}")
        return missing


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def _find_spec(module_name: str) -> Any:
    try:
        return importlib.util.find_spec(module_name)
    except (ImportError, AttributeError, ValueError):
        return None


def _resolve_checkpoint_file(checkpoint_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if path.parent != Path("."):
        return (ROOT / path).resolve()
    return checkpoint_dir / path


def _call_supported(function: Any, **kwargs: Any) -> Any:
    import inspect

    signature = inspect.signature(function)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return function(**kwargs)
    return function(**{key: value for key, value in kwargs.items() if key in signature.parameters})


def _materialize_engine_result(result: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if result is None:
        return
    if isinstance(result, bytes):
        output_path.write_bytes(result)
        return
    if isinstance(result, str | Path):
        result_path = Path(result)
        if result_path.exists() and result_path.resolve() != output_path.resolve():
            shutil.copy2(result_path, output_path)
        return
    if hasattr(result, "dump_midi"):
        result.dump_midi(str(output_path))
        return
    if isinstance(result, dict):
        for key in ("midi_path", "path", "output_path"):
            value = result.get(key)
            if value:
                _materialize_engine_result(value, output_path)
                return
        midi_bytes = result.get("midi_bytes")
        if isinstance(midi_bytes, bytes):
            output_path.write_bytes(midi_bytes)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "fail", "error": f"Invalid JSON summary: {path}"}
    return payload if isinstance(payload, dict) else {}


def _compact_json(value: dict[str, Any], limit: int = 2000) -> str:
    if not value:
        return "{}"
    text = json.dumps(value, ensure_ascii=True, sort_keys=True)
    return text if len(text) <= limit else text[-limit:]


def _tail(value: str | None, limit: int = 4000) -> str:
    if value is None:
        return ""
    return value if len(value) <= limit else value[-limit:]
