# Objetivo 14 - Contrato para IA futura

## Estado

Completado.

## Implementado

- Interfaces `ModelBackend`, `ModelRequest`, `ModelResponse` y
  `RoleModelGenerator` en `arranger_core`.
- `AIWalkingBassGenerator` compatible con el contrato actual de generadores por
  rol.
- `RuleBasedArranger` acepta `bass_generator=...` para sustituir
  `WalkingBassGenerator` sin tocar API ni web.
- Backend local `DeterministicWalkingBassBackend` para smoke tests y desarrollo
  sin modelo entrenado.
- Dataset train/val/test desde `PatternIndex` con filtrado por licencia,
  calidad y `usable_for_training`.
- `PatternTokenizer` placeholder para patrones y `ArrangementProject`.
- `FeatureStore` JSON para features de ejemplos de entrenamiento.
- Evaluacion de similitud/memorizacion mediante Jaccard de tokens.
- Paquete `midi_models` con adapters placeholder para MidiTok y modelos
  externos.
- Smoke script `scripts/ai_contract_smoke.py`.
- Documentacion de entrenamiento futuro en `docs/11_FUTURE_AI_TRAINING.md`.

## Guardrails

- Los ejemplos de entrenamiento solo se crean desde patrones con
  `usable_for_training=true`.
- Licencias vacias, `unknown`, `proprietary` y `all rights reserved` se bloquean.
- Cada ejemplo conserva procedencia: `source_file_id`, `source_path`,
  `source_hash`, `license` y fingerprint de patron.
- El reporte de memorizacion marca candidatos por encima del umbral configurado.

## Verificacion

Ejecutado:

- `python -m pip install -e ./packages/midi_models` - OK
- `python -m ruff check apps packages scripts tests` - OK
- `python -m pytest -q` - OK, 75 tests
- `npm --prefix apps/web run lint` - OK
- `npm --prefix apps/web run build` - OK
- `python scripts/bootstrap_check.py` - OK
- `python scripts/ai_contract_smoke.py` - OK
- `python scripts/package_smoke.py` - OK

Resultado de `ai_contract_smoke`:

- training examples: 8
- splits: train 6, val 1, test 1
- arranger: `hybrid_rule_model_v0`
- generador de bajo: `AIWalkingBassGenerator`
- backend: `deterministic-walking-bass-placeholder`
- validacion: `pass`
- memorizacion: `pass`
- exportacion: 10 archivos
