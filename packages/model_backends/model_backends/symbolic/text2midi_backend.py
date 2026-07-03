from __future__ import annotations

import importlib.util
import os
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


class Text2MidiBackend:
    backend_id = "text2midi"
    backend_version = "0.2.0"
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

    def is_available(self) -> bool:
        missing = self._missing_requirements()
        if missing:
            self.unavailable_reason = (
                "Text2MIDI is not installed/configured. Missing: "
                + ", ".join(missing)
            )
            return False
        self.unavailable_reason = ""
        return True

    def generate(self, request: ModelGenerationRequest) -> ModelGenerationResult:
        if request.task != "generate_full_sketch":
            raise UnsupportedModelTaskError("Text2MIDI only supports generate_full_sketch")
        self._ensure_available()

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
        temperature = (
            request.temperature
            if self.temperature is None
            else self.temperature
        )
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
            str(self.max_len),
            "--device",
            self.device,
            "--flan-tokenizer",
            self.flan_tokenizer,
            "--model-file",
            self.model_file,
            "--tokenizer-file",
            self.tokenizer_file,
        ]
        if request.seed is not None:
            cmd.extend(["--seed", str(request.seed)])

        completed = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise ModelGenerationError(
                "Text2MIDI subprocess failed"
                f" (exit {completed.returncode})."
                f" stdout={_tail(completed.stdout)!r}"
                f" stderr={_tail(completed.stderr)!r}"
            )
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise ModelGenerationError(
                "Text2MIDI subprocess completed without producing a MIDI sketch."
                f" stdout={_tail(completed.stdout)!r}"
                f" stderr={_tail(completed.stderr)!r}"
            )

        metadata = {
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "model_name": self.model_name,
            "prompt": prompt,
            "seed": request.seed,
            "temperature": temperature,
            "max_len": self.max_len,
            "repo_dir": str(self.repo_dir),
            "checkpoint_dir": str(self.checkpoint_dir),
            "flan_tokenizer": self.flan_tokenizer,
            "execution_mode": "subprocess",
            "sketch_only": True,
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
                "execution_mode": "subprocess",
                "sketch_only": True,
            },
        )

    def _ensure_available(self) -> None:
        if not self.is_available():
            raise ModelBackendUnavailableError(
                f"{self.unavailable_reason}. Install hint: {self.install_hint}"
            )

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
            if importlib.util.find_spec(module_name) is None:
                missing.append(f"python module {module_name}")
        return missing


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def _resolve_checkpoint_file(checkpoint_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if path.parent != Path("."):
        return (ROOT / path).resolve()
    return checkpoint_dir / path


def _tail(value: str, limit: int = 4000) -> str:
    if len(value) <= limit:
        return value
    return value[-limit:]
