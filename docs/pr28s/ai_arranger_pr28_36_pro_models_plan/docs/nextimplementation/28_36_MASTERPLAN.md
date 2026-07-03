# PR-28 a PR-36 — Masterplan ultra exhaustivo

## Estado actual detectado

El repo ya tiene:

- `model_backends` con contrato común.
- `configs/ai_models.local.example.yaml`.
- `requirements-ai.txt` y `requirements-training-ai.txt`.
- `scripts/models/` con smoke scripts.
- `apps/model_worker`.
- `OllamaPlannerBackend`.
- `MidiTokRealTokenizer`.
- `DummyCustomRoleModelBackend`.
- endpoints de AI infill, Text2MIDI sketch, takes, accept/reject y export.

## Problema pendiente

La fase PR-20 a PR-27 preparó infraestructura, pero todavía no garantiza:

1. instalación automática;
2. activación real por defecto en perfil local/pro;
3. integración MIDI-GPT contra API real completa;
4. wrapper Text2MIDI robusto;
5. planner LLM conectado al endpoint real;
6. MidiTok con datasets reales y splits controlados;
7. modelos propios no dummy;
8. orquestador profesional que reduzca resultados flojos;
9. quality gate que bloquee “MIDI patata”.

## Principio de implementación

La app debe tener tres perfiles:

```text
safe      → mock + rule-based, siempre arranca
local     → modelos descargados pero no necesariamente todos activos
pro       → planner + MIDI-GPT + Text2MIDI + MidiTok + custom/retrieval activados
```

El perfil `pro` no significa “saltarse validación”. Significa:

```text
más generación neural + más validación + más retry + más quality gate
```

## Resultado final esperado

El comando:

```bash
python scripts/models_pro/pro_end_to_end_smoke.py
```

debe:

1. comprobar entorno;
2. comprobar Ollama;
3. comprobar MIDI-GPT;
4. comprobar Text2MIDI;
5. comprobar MidiTok;
6. generar un proyecto hard bop;
7. crear plan con LLM local;
8. generar base rule-based/retrieval;
9. regenerar compases seleccionados con MIDI-GPT;
10. crear sketch Text2MIDI separado;
11. tokenizar fixture con MidiTok;
12. ejecutar custom-role bootstrap;
13. validar musicalmente;
14. exportar MIDI/MusicXML;
15. producir reporte profesional.

## No negociable

- Ningún modelo escribe el proyecto activo directamente.
- Text2MIDI no es resultado final.
- MIDI-GPT solo modifica pista/compases solicitados.
- El planner no genera notas.
- Los modelos propios no entrenan con licencias bloqueadas.
- Export comercial solo con modelos/datasets permitidos.
