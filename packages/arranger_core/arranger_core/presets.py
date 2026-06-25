from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from arranger_core.config_loader import MusicConfigLoader
from arranger_core.schema import GenerationSpec


class PresetModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GenerationPreset(PresetModel):
    id: str
    name: str
    description: str = ""
    prompt: str
    spec: GenerationSpec


class EvaluationPrompt(PresetModel):
    id: str
    preset_id: str
    prompt: str
    seed: int = 0


class PresetLibrary:
    def __init__(
        self,
        presets: dict[str, GenerationPreset],
        evaluation_prompts: list[EvaluationPrompt],
    ) -> None:
        self.presets = presets
        self.evaluation_prompts = evaluation_prompts

    @classmethod
    def load_default(cls, config_root: str | Path | None = None) -> PresetLibrary:
        loader = MusicConfigLoader(config_root)
        preset_data = loader.load_yaml_files("generation_presets/*.yaml")
        presets = {
            data["id"]: GenerationPreset.model_validate(data)
            for data in preset_data.values()
        }
        evaluation_data = loader.load_yaml("evaluation_pack.yaml").get(
            "evaluation_prompts",
            [],
        )
        evaluation_prompts = [
            EvaluationPrompt.model_validate(item) for item in evaluation_data
        ]
        return cls(presets=presets, evaluation_prompts=evaluation_prompts)

    def get(self, preset_id: str) -> GenerationPreset:
        try:
            return self.presets[preset_id]
        except KeyError as exc:
            raise KeyError(f"Unknown generation preset: {preset_id}") from exc

    def list_presets(self) -> list[GenerationPreset]:
        return sorted(self.presets.values(), key=lambda preset: preset.id)

    def evaluation_pack(self) -> list[EvaluationPrompt]:
        return list(self.evaluation_prompts)

    def specs(self) -> dict[str, GenerationSpec]:
        return {preset_id: preset.spec for preset_id, preset in self.presets.items()}

    def model_dump(self) -> dict[str, Any]:
        return {
            "presets": [
                preset.model_dump(mode="json") for preset in self.list_presets()
            ],
            "evaluation_prompts": [
                prompt.model_dump(mode="json") for prompt in self.evaluation_prompts
            ],
        }
