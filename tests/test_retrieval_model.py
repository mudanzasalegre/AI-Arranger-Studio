from __future__ import annotations

from arranger_core import (
    GenerationSpec,
    PatternAdapter,
    PatternRetriever,
    RetrievalQuery,
    generate_arrangement,
)


def test_pattern_retriever_ranks_contextual_match_and_adapter_transforms_payload():
    exact = _bass_pattern(
        "exact-hard-bop",
        style="hard_bop",
        quality=3,
        chord_context=["Cm7", "F7"],
        intervals=[0, 7, 10, 0],
    )
    generic = _bass_pattern(
        "generic-swing",
        style="swing",
        quality=5,
        chord_context=["Dbmaj7", "Gb7"],
        intervals=[0, 3, 7, 10],
    )
    query = RetrievalQuery(
        category="walking_bass_cells",
        role="walking_bass",
        style="hard_bop",
        instrument="double_bass",
        density="medium",
        meter="4/4",
        chord_context=["Cm7", "F7", "Bb7"],
        seed=1301,
    )

    matches = PatternRetriever([generic, exact]).search(query)
    adapted = PatternAdapter().adapt(matches[0], query)

    assert matches[0].pattern["id"] == "exact-hard-bop"
    assert "style_exact" in matches[0].reasons
    assert "harmonic_context" in matches[0].reasons
    assert adapted.pattern["payload"]["pitch_intervals"] != exact["payload"]["pitch_intervals"]
    assert adapted.similarity_to_source < 1.0
    assert adapted.transformations_applied
    assert adapted.pattern["context"]["retrieval"]["score"] == matches[0].score
    assert adapted.pattern["context"]["retrieval"]["source_pattern_id"] == "exact-hard-bop"


def test_generators_use_retrieval_model_to_select_and_trace_patterns():
    project = generate_arrangement(
        GenerationSpec(
            ensemble="jazz_trio",
            form="minor_blues_12",
            style="hard_bop",
            seed=1302,
            constraints={
                "pattern_index": {
                    "patterns": [
                        _bass_pattern(
                            "high-quality-wrong-style",
                            style="swing",
                            quality=5,
                            chord_context=["Dbmaj7", "Gb7"],
                            intervals=[0, 3, 7, 10],
                        ),
                        _bass_pattern(
                            "contextual-hard-bop",
                            style="hard_bop",
                            quality=3,
                            chord_context=["Cm7", "F7"],
                            intervals=[0, 7, 10, 0],
                        ),
                    ]
                }
            },
        ),
        project_id="retrieval-model-generation",
    )
    bass = next(track for track in project.tracks if track.id == "double_bass")
    trace = bass.metadata["retrieval_trace"]

    assert bass.metadata["bass_engine_mode"] == "retrieval"
    assert bass.metadata["learned_pattern_id"] == "contextual-hard-bop"
    assert trace["source_pattern_id"] == "contextual-hard-bop"
    assert trace["transformations_applied"]
    assert trace["similarity_to_source"] < 1.0
    assert bass.metadata["bass_validation"]["status"] == "pass"


def _bass_pattern(
    pattern_id: str,
    *,
    style: str,
    quality: int,
    chord_context: list[str],
    intervals: list[int],
) -> dict[str, object]:
    return {
        "id": pattern_id,
        "category": "walking_bass_cells",
        "role": "walking_bass",
        "style": style,
        "quality": quality,
        "weight": 1.0,
        "source_hash": f"hash-{pattern_id}",
        "license": "CC0-1.0",
        "usable_for_training": True,
        "usable_for_pattern_extraction": True,
        "tags": ["bass", "walking_bass"],
        "context": {
            "chord_context": chord_context,
            "instrument_guess": "bass",
            "role_confidence": 0.96,
            "pattern_sensitivity": {
                "level": "low",
                "commercial_training": "allowed",
                "local_learning_only": False,
            },
        },
        "payload": {
            "pitch_intervals": intervals,
            "rhythm": [0.0, 1.0, 2.0, 3.0],
            "density": 4,
        },
        "fingerprint": f"fingerprint-{pattern_id}",
    }
