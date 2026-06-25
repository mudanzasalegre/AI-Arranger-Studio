# 09 — Contrato para IA futura

## Implementacion actual

- `arranger_core.ModelBackend`, `ModelRequest`, `ModelResponse` y
  `RoleModelGenerator` definen el contrato runtime.
- `arranger_core.AIWalkingBassGenerator` es el primer generador intercambiable.
- `RuleBasedArranger(bass_generator=...)` permite probar un generador de modelo
  sin cambiar API ni frontend.
- `dataset_tools.build_training_examples(...)` crea `train/val/test`,
  `training_examples.jsonl` y `feature_store.json` solo desde patrones con
  permiso de entrenamiento.
- `dataset_tools.PatternTokenizer` deja el placeholder de tokenizacion simbolica.
- `dataset_tools.evaluate_memorization(...)` reporta similitud contra corpus.
- `midi_models.MidiTokBackendAdapter` y `ExternalModelBackendAdapter` existen
  como placeholders explicitos.

Ver tambien `docs/11_FUTURE_AI_TRAINING.md`.

## Objetivo

Que el sistema pueda empezar sin IA y luego admitir modelos simbólicos sin rehacer arquitectura.

## Interfaces

### ModelBackend

```python
class ModelBackend(Protocol):
    name: str
    version: str
    def generate(self, request: ModelRequest) -> ModelResponse: ...
```

### RoleModelGenerator

```python
class RoleModelGenerator(RoleGenerator):
    backend: ModelBackend
```

## Primeros modelos candidatos futuros

- walking bass condicionado por acordes
- piano comping condicionado por acordes/bajo
- melody/head condicionado por forma/acordes
- horn responses condicionado por melodía/huecos
- drums condicionado por estilo/sección

## Tokenización futura

Preparar estructura para:

- MidiTok
- tokens propios `ArrangementProject`
- dataset train/val/test por rol

## Dataset guardrails

- solo entrenar con `usable_for_training=true`
- guardar hash y procedencia
- evitar memorización
- medir similitud contra corpus
- no copiar material protegido

## Reemplazo progresivo

No cambiar API ni UI. Cambiar backend de cada rol:

```text
RuleBasedWalkingBassGenerator
  → PatternRetrievalWalkingBassGenerator
  → AIWalkingBassGenerator
```
