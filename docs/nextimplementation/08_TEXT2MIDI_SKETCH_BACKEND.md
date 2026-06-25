# 08 — Integración Text2MIDI como sketch backend

## Rol en el sistema

Text2MIDI sirve para crear un **boceto simbólico** desde prompt. No debe generar el arreglo profesional final.

Uso:

```text
prompt -> MIDI sketch -> importer -> ArrangementProject sketch -> validación -> take/sketch
```

## Cuándo usarlo

- Cuando el usuario empieza desde cero y quiere una idea rápida.
- Para comparar contra el motor rule-based.
- Para crear material de inspiración.
- Para generar una melodía/progresión inicial que luego se reescribe.

## Cuándo no usarlo

- No para export final directo.
- No para sustituir `SongPlan`.
- No para sustituir `HarmonyEngine`.
- No para generar MusicXML final sin validación.

## Backend skeleton

```python
class Text2MidiBackend:
    backend_id = "text2midi"
    capabilities = ModelCapabilities(
        symbolic_midi=True,
        multitrack=False,
        bar_infill=False,
        track_generation=False,
        text_prompt=True,
        json_planning=False,
        supports_training=True,
        commercial_use="review_required",
    )

    def generate(self, request: ModelGenerationRequest) -> ModelGenerationResult:
        if request.task != "generate_full_sketch":
            raise ValueError("Text2MIDI solo soporta generate_full_sketch en esta fase")
        # Ejecutar como subproceso/worker, no cargar siempre en API.
```

## Endpoint

```text
POST /v1/ai/text-to-midi-sketch
```

Payload:

```json
{
  "backend": "text2midi",
  "prompt": "Hard bop minor blues in C minor, 132 BPM, jazz sextet with walking bass, piano comping, alto sax lead, trumpet and trombone responses.",
  "seed": 1234
}
```

## Importación

Tras generar el MIDI:

```text
1. Importar MIDI.
2. Detectar tempo/compás.
3. Clasificar pistas.
4. Inferir roles.
5. Inferir forma si es posible.
6. Inferir chord grid si existe o marcar unknown.
7. Ejecutar validadores.
8. Guardar como sketch.
```

## Estados de confianza

```text
sketch_valid
sketch_uncertain
sketch_rejected
```

`sketch_uncertain` se usa cuando:

- no se detectan roles;
- faltan acordes;
- hay pistas ambiguas;
- el MIDI es demasiado denso;
- la estructura no se puede mapear a SongPlan.

## Acceptance criteria

- Funciona con mock aunque Text2MIDI no esté instalado.
- Si backend real no está instalado, error controlado.
- Resultado no se mezcla automáticamente con proyecto profesional.
- Se puede visualizar/importar como sketch.
- Se puede usar como referencia para crear SongPlan nuevo.
