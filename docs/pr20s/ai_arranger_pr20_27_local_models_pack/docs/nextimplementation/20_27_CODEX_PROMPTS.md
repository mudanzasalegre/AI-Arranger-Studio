# Prompts para Codex — PR-20 a PR-27

## PR-20

```text
Implementa PR-20 Local Model Runtime usando los archivos del local models pack. No instales modelos reales. Añade configs locales, scripts/models, apps/model_worker health/status, Makefile targets de modelos y variables .env.example. Debe pasar lint/test y `python scripts/models/check_local_model_runtime.py`.
```

## PR-21

```text
Implementa PR-21 MIDI-GPT local. Instala dependencia opcional solo documentada, no obligatoria para tests. Adapta MidiGptBackend a la API real de midigpt: Score.from_midi, InferenceEngine.from_pretrained, GenerationRequest, InferenceConfig, TrackPrompt, engine.session(...).run(), result.to_midi(...). Mantén lazy imports, error controlado si falta midigpt y artifact quarantine. Añade smoke_midigpt.py y tests con mock/monkeypatch.
```

## PR-22

```text
Implementa PR-22 Text2MIDI local como sketch backend. No promuevas sketches a arreglo final. Añade download_text2midi.py, smoke_text2midi.py y wrapper subprocess/worker si el repo externo no funciona como paquete importable. El endpoint /v1/ai/text-to-midi-sketch debe importar, clasificar roles, validar y devolver sketch_validated/sketch_uncertain/sketch_rejected.
```

## PR-23

```text
Implementa PR-23 Local LLM Planner. Añade provider Ollama para LlmPlanner, con JSON estricto, retry una vez y fallback rule-based. El planner no genera notas ni modifica tracks. Añade smoke_ollama_planner.py y endpoint /v1/projects/{id}/ai/plan usando provider si está habilitado en config.
```

## PR-24

```text
Implementa PR-24 MidiTok/training stack. Añade dependencia opcional, tokenizer real importado lazy, tokenización por rol, metadata de licencia por segmento y smoke_miditok.py. No entrenes aún modelos grandes. Mantén el bridge tokenizer actual como fallback.
```

## PR-25

```text
Implementa PR-25 Custom Role Model Bootstrap. Añade loader/backends dummy para modelos propios por rol. Los modelos sin checkpoint/training_manifest/license_report deben aparecer como unavailable. Añade tests de registry y manifest licensing.
```

## PR-26

```text
Implementa PR-26 Local Model Smoke Tests. Crea ai_local_smoke.py que verifica API, registry, planner, infill mock, infill midigpt si enabled, text2midi sketch si enabled, quarantine, takes y export. Debe generar outputs/model_smoke/ai_local_smoke_summary.json.
```

## PR-27

```text
Implementa PR-27 Professional Generation Benchmark. Lee configs/professional_benchmarks.yaml, genera los 5 casos, aplica AI infill si backend está enabled, acepta takes válidas, exporta paquetes y genera summary.json/summary.md. El benchmark debe fallar si hay errores bloqueantes de validación.
```
