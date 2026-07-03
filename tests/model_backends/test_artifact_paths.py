from __future__ import annotations

from model_backends.artifact import MAX_ARTIFACT_STEM_LENGTH, safe_artifact_stem


def test_safe_artifact_stem_preserves_short_names():
    assert safe_artifact_stem("mock_symbolic", "infill_bars", "valid-midi") == (
        "mock_symbolic_infill_bars_valid-midi"
    )


def test_safe_artifact_stem_truncates_long_names_with_stable_hash():
    stem = safe_artifact_stem(
        "custom_jazz_walking_bass_v001",
        "infill_bars",
        "custom_role_model_custom_jazz_walking_bass_v001_88e7c96f8d",
    )
    repeated = safe_artifact_stem(
        "custom_jazz_walking_bass_v001",
        "infill_bars",
        "custom_role_model_custom_jazz_walking_bass_v001_88e7c96f8d",
    )

    assert stem == repeated
    assert len(stem) <= MAX_ARTIFACT_STEM_LENGTH
    assert stem.startswith("custom_jazz_walking_bass_v001_infill_bars")
