# PR-26 — Local Model Smoke Tests

## Objetivo

Crear una batería de pruebas reproducibles que confirme que los modelos locales funcionan sin romper el flujo profesional.

## Tests mínimos

```text
1. models-check
2. mock backend smoke
3. local LLM planner smoke
4. MIDI-GPT load smoke
5. MIDI-GPT infill API smoke
6. Text2MIDI download/checkpoint smoke
7. Text2MIDI sketch API smoke
8. MidiTok tokenization smoke
9. Artifact quarantine smoke
10. Takes accept/reject smoke
11. Export smoke
```

## Target Makefile

Añadir:

```make
ai-local-smoke:
	python scripts/models/ai_local_smoke.py
```

## Script principal

`scripts/models/ai_local_smoke.py` debe:

1. Consultar `/health`.
2. Consultar `/v1/ai/models`.
3. Generar proyecto rule-based.
4. Ejecutar `/v1/projects/{id}/ai/plan`.
5. Ejecutar `/v1/projects/{id}/ai/infill` con `mock_symbolic` siempre.
6. Si `midigpt.enabled=true`, ejecutar infill real.
7. Si `text2midi.enabled=true`, ejecutar sketch real.
8. Comprobar takes.
9. Aceptar una take mock.
10. Exportar ZIP.
11. Escribir `outputs/model_smoke/ai_local_smoke_summary.json`.

## Acceptance criteria

```text
- El smoke completo pasa con mock aunque no haya modelos reales.
- Si midigpt está activado, prueba midigpt real.
- Si text2midi está activado, prueba sketch real.
- Ningún artifact queda sin estado.
- No hay pending takes en export final salvo que el test lo permita explícitamente.
```
