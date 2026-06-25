# 08 — Especificación de exportación

## Output package

```text
outputs/<project_id>/
  arrangement_project.json
  generation_spec.json
  export_manifest.json
  full_arrangement.mid
  full_score.musicxml
  full_score.pdf
  midi_tracks/
    drums.mid
    double_bass.mid
    piano.mid
    alto_sax.mid
    trumpet.mid
    trombone.mid
  parts_pdf/
    drums.pdf
    double_bass.pdf
    piano.pdf
    alto_sax.pdf
    trumpet.pdf
    trombone.pdf
  audio_preview/
    full_mix.wav
  validation_report.json
  validation_report.html
```

## MIDI

- una pista por instrumento
- nombres claros
- tempo map
- markers de secciones
- canal 10 para batería
- program changes General MIDI razonables
- notas humanizadas en export performance

## MusicXML

Debe incluir:

- partes
- compases
- armadura
- compás
- tempo
- cifrado armónico
- articulaciones básicas
- dinámicas básicas
- marcadores de sección
- partes transpuestas cuando proceda

## PDF

Generado desde MusicXML preferiblemente mediante MuseScore CLI.

## ZIP

El ZIP final debe contener todos los outputs y README breve de uso.
