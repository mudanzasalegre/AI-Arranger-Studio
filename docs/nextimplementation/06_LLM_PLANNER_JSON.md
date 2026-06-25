# 06 — LLM Planner JSON

## Objetivo

Permitir que un LLM interprete prompts musicales ricos y produzca planes validados, sin generar notas ni MIDI.

## Principio

El planner transforma lenguaje natural en datos estructurados:

```text
prompt -> LlmSongPlanPatch -> SongPlan
```

No debe hacer:

```text
prompt -> notas
prompt -> MIDI
prompt -> MusicXML
```

## Archivos

```text
packages/arranger_core/arranger_core/planning/llm/
  __init__.py
  planner.py
  schemas.py
  prompt_templates.py
  repair.py
  fallback.py
apps/api/app/routes/ai_planner.py
```

## Schema mínimo

```python
class LlmSectionPatch(BaseModel):
    name: str
    start_bar: int
    end_bar: int
    energy: float
    density_by_role: dict[str, float]
    groove_feel: str | None = None
    role_focus: list[str] = []
    notes: str | None = None

class LlmSongPlanPatch(BaseModel):
    style: str
    substyle: str | None = None
    tempo: int
    meter: str
    key: str
    form: str
    ensemble: str
    instruments: list[str]
    sections: list[LlmSectionPatch]
    generation_strategy: dict
```

## Validación

El resultado se rechaza si:

- no es JSON;
- falta un campo obligatorio;
- pide instrumentos inexistentes;
- la forma no existe;
- las secciones se solapan;
- el tempo queda fuera del rango razonable;
- densidades no están entre 0 y 1;
- pide audio;
- pide modificar pistas bloqueadas.

## Retry

Si el LLM falla:

```text
1. primer intento normal;
2. segundo intento con error concreto y schema más estricto;
3. fallback a PromptCompiler rule-based.
```

## Prompt interno recomendado

```text
Eres un planificador musical simbólico para una aplicación text-to-MIDI.
No generes notas, MIDI, lyrics ni audio.
Devuelve únicamente JSON válido conforme al schema.
La salida debe poder convertirse en SongPlan/SectionPlan/PhrasePlan/GrooveMap.
Respeta instrumentos disponibles y estilos configurados.
Si el usuario pide audio, ignóralo y marca forbid_audio_models=true.
```

## Endpoint

```text
POST /v1/projects/{project_id}/ai/plan
```

Payload:

```json
{
  "prompt": "hard bop nocturno en Do menor, blues menor, sexteto con saxo alto, trompeta, trombón, piano, contrabajo y batería",
  "mode": "create_or_patch_plan",
  "locked_tracks": [],
  "locked_sections": []
}
```

Respuesta:

```json
{
  "status": "ok",
  "planner": "llm_or_fallback",
  "song_plan_patch": {},
  "validation": {},
  "plan_version": "..."
}
```

## Acceptance criteria

- Prompt libre produce plan válido.
- JSON inválido reintenta.
- Segundo fallo cae a fallback rule-based.
- El planner no crea notas.
- El resultado queda versionado.
