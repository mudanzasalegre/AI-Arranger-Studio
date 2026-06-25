# 07 — Integración MIDI-GPT

## Rol en el sistema

MIDI-GPT es el backend simbólico principal para generación controlada de material MIDI sobre un proyecto existente.

Uso prioritario:

```text
infill_bars
generate_track
continue_section
generate_variation
```

No usar en esta fase para:

```text
generar canción final completa
sustituir HarmonyEngine
sustituir validadores
exportar resultado final directo
```

## Instalación opcional

No incluir como dependencia obligatoria de la API.

```bash
pip install "midigpt[inference]"
```

También puede ejecutarse vía servidor HTTP si se decide aislar modelos en `apps/model_worker`.

## Backend skeleton

```python
class MidiGptBackend:
    backend_id = "midigpt"
    capabilities = ModelCapabilities(
        symbolic_midi=True,
        multitrack=True,
        bar_infill=True,
        track_generation=True,
        text_prompt=False,
        json_planning=False,
        supports_training=True,
        commercial_use="review_required",
    )

    def __init__(self, model_name="yellow", output_dir="outputs/model_artifacts/raw"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self._engine = None

    def _load_engine(self):
        if self._engine is not None:
            return self._engine
        try:
            from midigpt.inference.engine import InferenceEngine
        except ImportError as exc:
            raise RuntimeError(
                "MIDI-GPT no está instalado. Instala el extra: pip install 'midigpt[inference]'"
            ) from exc
        self._engine = InferenceEngine.from_pretrained(self.model_name)
        return self._engine
```

## Flujo `infill_bars`

```text
1. Cargar proyecto/take activa.
2. Exportar contexto temporal a MIDI.
3. Resolver track_id interno -> índice de pista del score.
4. Crear request para compases target.
5. Ejecutar backend.
6. Guardar MIDI raw.
7. Importar artifact.
8. Extraer solo track/bars target.
9. Fusionar en copia temporal.
10. Validar.
11. Crear take pending si pasa.
```

## Context MIDI

El MIDI temporal debe incluir:

- todas las pistas relevantes;
- nombres de pista estables;
- tempo;
- compás;
- markers de secciones;
- material existente en pistas no target;
- silencios o máscara en compases target si el backend lo requiere.

No debe incluir:

- pistas ocultas de debug;
- audio;
- cambios de tempo no soportados sin normalizar;
- información que rompa el importador.

## Endpoint

```text
POST /v1/projects/{project_id}/ai/infill
```

Payload:

```json
{
  "backend": "midigpt",
  "track_id": "alto_sax",
  "bars": [17, 18, 19, 20],
  "instruction": "bebop phrase, medium density, clear resolution into bar 21",
  "density": "medium",
  "temperature": 0.85,
  "seed": 1234
}
```

## Retry policy

```text
attempt 1: temperature original
attempt 2: temperature -0.15 y densidad menor
attempt 3: repair instruction basada en validation report
fallback: generador rule-based/retrieval
```

## Validadores específicos

Para melodía/saxo:

- rango absoluto;
- rango cómodo;
- respiración;
- duración de compás;
- relación nota-acorde;
- resolución en cadencias;
- densidad por PhrasePlan.

Para piano:

- polifonía;
- registro;
- duplicación del bajo;
- distancia entre voicings;
- densidad por GrooveMap.

Para vientos:

- spacing;
- cruce de voces;
- respiración;
- hits en huecos permitidos;
- articulación.

## Acceptance criteria

- Backend no rompe si dependencia falta.
- Con mock o fixture se puede generar take.
- No modifica proyecto activo.
- Solo modifica track/bars target.
- Rechaza material inválido.
- Registra backend/model/seed/instruction.
