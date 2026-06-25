# 06 — Validación y QA musical

## Filosofía

No basta con que el código funcione: la música generada debe ser legible, tocable y coherente.

## Validadores obligatorios

### BarDurationValidator

- cada compás suma lo correcto
- detecta huecos y sobrecarga

### InstrumentRangeValidator

- error si nota fuera de rango absoluto
- warning si nota fuera de rango cómodo

### TranspositionValidator

- partes transpuestas correctas para Bb/Eb
- full score configurable en concert pitch o written pitch

### HarmonyValidator

- porcentaje de chord tones
- porcentaje de tensions
- cromatismos justificados
- avoid notes como warning

### VoiceLeadingValidator

- cruces de voces
- saltos excesivos
- separación excesiva
- duplicaciones problemáticas

### BreathValidator

- frases demasiado largas para vientos
- densidad excesiva
- falta de silencios

### PianoPlayabilityValidator

- voicings imposibles
- mano demasiado abierta
- duplicación de bajo en grave

### DrumValidator

- canal/pitches de batería correctos
- fills colocados musicalmente
- densidad razonable

### ExportValidator

- se crearon archivos esperados
- MusicXML parseable
- MIDI contiene pistas separadas

## Reporte

```json
{
  "status": "pass_with_warnings",
  "errors": [],
  "warnings": [],
  "metrics": {
    "bars": 32,
    "tracks": 6,
    "avg_note_density": {},
    "range_warnings": 2,
    "harmony_score": 0.82
  }
}
```

## Golden tests

Crear demos estables:

- demo_hard_bop_minor_blues
- demo_bebop_blues
- demo_ballad_aaba
- demo_modal_quintet
- demo_bossa_quartet

Cada demo debe generar:

- JSON project
- MIDI
- MusicXML
- validation report

## Métricas musicales

- densidad por pista
- rango usado por instrumento
- proporción chord tones/tensions
- variedad rítmica
- repetición motívica
- frecuencia de fills
- legibilidad de partitura
- número de warnings por 100 compases
