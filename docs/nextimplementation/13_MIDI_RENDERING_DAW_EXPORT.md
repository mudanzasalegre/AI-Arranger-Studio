# 13 — MIDI rendering y export DAW-ready con IA

## Objetivo

Garantizar que material generado por IA se integra en el mismo estándar de exportación profesional.

## Reglas

- Solo exportar takes accepted.
- Normalizar nombres de pistas.
- Mantener canales MIDI coherentes.
- Mantener program changes.
- Mantener markers.
- Mantener tempo/meter del proyecto salvo sketch aislado.
- No exportar artifacts raw.

## PerformanceMap aplicado a IA

Material importado puede traer:

- velocities extrañas;
- duraciones solapadas;
- microtiming excesivo;
- cuantización irregular.

Se debe aplicar una normalización:

```text
imported_model -> normalized_model -> performance_map final
```

## Export ZIP

Debe incluir:

```text
full_arrangement.mid
midi_tracks/
full_score.musicxml
full_score.pdf
parts_pdf/
validation_report.html
takes_manifest.json
model_trace.json
session_readme.md
```

## model_trace.json

```json
{
  "project_id": "...",
  "active_take_id": "...",
  "model_artifacts": [
    {
      "backend_id": "midigpt",
      "task": "infill_bars",
      "track_id": "alto_sax",
      "bars": [17, 18, 19, 20],
      "seed": 1234,
      "status": "accepted"
    }
  ]
}
```

## Acceptance criteria

- Export ZIP no contiene takes rejected.
- Export ZIP contiene trazabilidad de IA si se usó.
- MIDI abre en DAW con pistas separadas.
- MusicXML abre en MuseScore.
- Particellas se generan desde take aceptada.
