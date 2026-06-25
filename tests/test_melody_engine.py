from __future__ import annotations

import random

from arranger_core import (
    ChordParser,
    GenerationContext,
    GenerationSpec,
    InstrumentCatalog,
    MelodyEngine,
    NoteEvent,
    RestEvent,
    generate_arrangement,
    generate_harmony_project,
    generate_song_plan,
    note_to_midi,
)


def test_melody_engine_rule_based_tracks_phrases_motifs_and_breathing():
    project = generate_arrangement(
        GenerationSpec(ensemble="jazz_quartet_alto", form="minor_blues_12", seed=801),
        project_id="melody-rule-based",
    )
    melody = _track(project, "alto_sax")
    notes = _notes(melody)
    breath_rests = [
        event
        for bar in melody.bars
        for event in bar.events
        if isinstance(event, RestEvent) and event.annotations.get("breath")
    ]

    assert melody.metadata["melody_engine_version"] == "0.1.0"
    assert melody.metadata["melody_engine_mode"] == "rule_based"
    assert melody.metadata["melody_validation"]["status"] == "pass"
    assert melody.metadata["motif_ledger"]["entries"]
    assert {entry["source"] for entry in melody.metadata["motif_ledger"]["entries"]} == {
        "rule_based"
    }
    assert breath_rests
    melody_engine_notes = [
        event
        for event in notes
        if event.annotations.get("melody_engine") == "MelodyEngine2"
    ]
    assert melody_engine_notes
    assert all("phrase_id" in event.annotations for event in melody_engine_notes)
    assert all(
        note_to_midi("Db3") <= note_to_midi(event.pitch) <= note_to_midi("Ab5")
        for event in notes
    )
    assert project.validate_bar_durations() == []


def test_melody_engine_retrieval_mode_adapts_melodic_motifs():
    project = generate_arrangement(
        GenerationSpec(
            ensemble="jazz_quartet_alto",
            form="minor_blues_12",
            seed=802,
            constraints={
                "melody_mode": "retrieval",
                "pattern_index": {
                    "patterns": [
                        {
                            "id": "motif_fixture_001",
                            "category": "melodic_motifs",
                            "role": "melody",
                            "style": "hard_bop",
                            "quality": 4,
                            "weight": 1.0,
                            "usable_for_pattern_extraction": True,
                            "payload": {
                                "relative_degrees": [0, 2, 3, 7],
                                "rhythm": [0.5, 0.5, 0.5, 0.5],
                            },
                        }
                    ]
                },
            },
        ),
        project_id="melody-retrieval",
    )
    melody = _track(project, "alto_sax")
    retrieved_notes = [
        event
        for event in _notes(melody)
        if event.annotations.get("learned_pattern_id") == "motif_fixture_001"
    ]

    assert melody.metadata["melody_engine_mode"] == "retrieval"
    assert melody.metadata["melody_validation"]["status"] == "pass"
    assert retrieved_notes
    assert all(event.annotations["retrieval_transform"] for event in retrieved_notes)
    assert any(
        entry["source"] == "retrieval"
        for entry in melody.metadata["motif_ledger"]["entries"]
    )
    assert project.validate_bar_durations() == []


def test_melody_engine_ai_infill_accepts_valid_backend_material():
    context = _context(
        GenerationSpec(
            ensemble="jazz_quartet_alto",
            form="minor_blues_12",
            seed=803,
            constraints={"melody_mode": "ai_infill"},
        ),
        project_id="melody-ai-infill",
    )
    engine = MelodyEngine(
        chord_parser=context.chord_parser,
        instrument_catalog=context.instrument_catalog,
        ai_backend=_ValidAiBackend(),
    )

    track = engine.generate_for_instrument(context, "alto_sax")

    assert track.metadata["melody_engine_mode"] == "ai_infill"
    assert track.metadata["ai_infill_status"] == "accepted"
    assert any(event.annotations.get("melody_ai_infill") for event in _notes(track))
    assert any(
        entry["source"] == "ai_infill"
        for entry in track.metadata["motif_ledger"]["entries"]
    )
    assert track.metadata["melody_validation"]["status"] == "pass"


def test_melody_engine_ai_infill_falls_back_when_backend_fails():
    context = _context(
        GenerationSpec(
            ensemble="jazz_quartet_alto",
            form="minor_blues_12",
            seed=804,
            constraints={"melody_mode": "ai_infill"},
        ),
        project_id="melody-ai-fallback",
    )
    engine = MelodyEngine(
        chord_parser=context.chord_parser,
        instrument_catalog=context.instrument_catalog,
        ai_backend=_FailingAiBackend(),
    )

    track = engine.generate_for_instrument(context, "alto_sax")

    assert track.metadata["melody_engine_mode"] == "rule_based"
    assert track.metadata["fallback_reason"].startswith("ai_backend_error")
    assert {entry["source"] for entry in track.metadata["motif_ledger"]["entries"]} == {
        "fallback_rule_based"
    }
    assert track.metadata["melody_validation"]["status"] == "pass"


class _ValidAiBackend:
    def generate_melody_infill(
        self,
        *,
        project,
        base_track,
        instrument_id,
        target_bars,
        context,
    ):
        bars = []
        base_bars = {bar.number: bar for bar in base_track.bars}
        for bar_number in target_bars:
            base_bar = base_bars[bar_number]
            first_note = next(event for event in base_bar.events if isinstance(event, NoteEvent))
            bars.append(
                base_bar.model_copy(
                    update={
                        "events": [
                            NoteEvent(
                                pitch=first_note.pitch,
                                start=0.0,
                                duration=1.0,
                                velocity=86,
                                articulations=["accent"],
                                annotations={
                                    "melodic_role": "cadence",
                                    "source_chord": first_note.annotations["source_chord"],
                                },
                            ),
                            RestEvent(start=1.0, duration=3.0, annotations={"breath": True}),
                        ]
                    },
                    deep=True,
                )
            )
        return base_track.model_copy(update={"bars": bars}, deep=True)


class _FailingAiBackend:
    def generate_melody_infill(self, **_kwargs):
        raise RuntimeError("simulated backend failure")


def _context(spec: GenerationSpec, *, project_id: str) -> GenerationContext:
    chord_parser = ChordParser.load_default()
    instrument_catalog = InstrumentCatalog.load_default()
    project = generate_harmony_project(spec, project_id=project_id)
    song_plan = generate_song_plan(spec, project)
    return GenerationContext(
        spec=spec,
        project=project,
        instrument_ids=spec.instruments or ["drum_kit", "double_bass", "piano", "alto_sax"],
        chord_parser=chord_parser,
        instrument_catalog=instrument_catalog,
        rng=random.Random(spec.seed),
        learned_patterns={},
        song_plan=song_plan,
    )


def _track(project, track_id):
    return next(track for track in project.tracks if track.id == track_id)


def _notes(track):
    return [
        event
        for bar in track.bars
        for event in bar.events
        if isinstance(event, NoteEvent)
    ]
