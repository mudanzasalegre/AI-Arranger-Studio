from __future__ import annotations

import json

import pytest
from model_backends import MockSymbolicBackend, ModelGenerationError, ModelGenerationRequest


def test_mock_backend_writes_valid_midi_artifact(tmp_path):
    backend = MockSymbolicBackend(output_dir=tmp_path)

    result = backend.generate(
        ModelGenerationRequest(
            task="infill_bars",
            request_id="valid-midi",
            track_id="alto_sax",
            bars=[17, 18],
            seed=1234,
            song_plan={"song_id": "fixture"},
            section_plan={"id": "section_a"},
            phrase_plan={"id": "phrase_a"},
            groove_map={"feel": "swing"},
            role_intent={"role": "melody"},
        )
    )

    artifact = result.artifacts[0]
    artifact_path = tmp_path / "mock_symbolic_infill_bars_valid-midi.mid"
    assert artifact.artifact_type == "midi"
    assert artifact.path == str(artifact_path)
    assert artifact_path.read_bytes()[:4] == b"MThd"
    assert artifact.metadata["track_id"] == "alto_sax"
    assert artifact.metadata["bars"] == [17, 18]
    assert result.confidence == 0.95


def test_mock_backend_writes_invalid_midi_fixture(tmp_path):
    backend = MockSymbolicBackend(output_dir=tmp_path)

    result = backend.generate(
        ModelGenerationRequest(
            task="generate_track",
            request_id="invalid-midi",
            seed=12,
            metadata={"mock_artifact": "invalid_midi"},
        )
    )

    artifact_path = tmp_path / "mock_symbolic_generate_track_invalid-midi.mid"
    assert result.artifacts[0].path == str(artifact_path)
    assert artifact_path.read_bytes() == b"not a valid midi file"
    assert result.confidence == 0.1
    assert result.warnings == ["mock_invalid_midi"]
    metadata = json.loads(artifact_path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    assert metadata["reason"] == "mock_invalid_midi"


def test_mock_backend_writes_empty_midi_fixture(tmp_path):
    backend = MockSymbolicBackend(output_dir=tmp_path)

    result = backend.generate(
        ModelGenerationRequest(
            task="infill_bars",
            request_id="empty-midi",
            track_id="alto_sax",
            bars=[1],
            seed=13,
            metadata={"mock_artifact": "empty_midi"},
        )
    )

    artifact_path = tmp_path / "mock_symbolic_infill_bars_empty-midi.mid"
    assert result.artifacts[0].path == str(artifact_path)
    assert artifact_path.read_bytes()[:4] == b"MThd"
    assert result.confidence == 0.1
    assert result.warnings == ["mock_empty_midi"]
    assert result.artifacts[0].metadata["empty"] is True


def test_mock_backend_writes_valid_json_plan_accepting_song_plan_contract(tmp_path):
    backend = MockSymbolicBackend(output_dir=tmp_path)

    result = backend.generate(
        ModelGenerationRequest(
            task="plan_song",
            request_id="json-plan",
            style="hard_bop",
            seed=9,
            song_plan={"song_id": "song-a"},
            section_plan={"id": "section-a"},
            phrase_plan={"id": "phrase-a"},
            groove_map={"feel": "swing"},
            role_intent={"instrument": "alto_sax"},
        )
    )

    payload = json.loads((tmp_path / "mock_symbolic_plan_song_json-plan.json").read_text())
    assert result.artifacts[0].artifact_type == "json"
    assert payload["style"] == "hard_bop"
    assert payload["song_plan"] == {"song_id": "song-a"}
    assert payload["section_plan"] == {"id": "section-a"}
    assert payload["phrase_plan"] == {"id": "phrase-a"}
    assert payload["groove_map"] == {"feel": "swing"}
    assert payload["role_intent"] == {"instrument": "alto_sax"}
    assert payload["generation_strategy"]["forbid_audio_models"] is True


def test_mock_backend_can_simulate_generation_error(tmp_path):
    backend = MockSymbolicBackend(output_dir=tmp_path)

    with pytest.raises(ModelGenerationError):
        backend.generate(
            ModelGenerationRequest(
                task="generate_full_sketch",
                metadata={"simulate_error": True},
            )
        )
