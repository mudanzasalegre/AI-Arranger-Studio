# 09 — Roadmap del motor musical actualizado

## Objetivo

Continuar el motor musical 2.0, pero ahora con puntos explícitos de integración IA simbólica.

## Estado actual

```text
Epic 16 implementado:
- SongPlan
- SectionPlan
- PhrasePlan
- GrooveMap
```

## Inserción obligatoria antes de Epic 17

```text
Epic 16.5 — AI Model Backend Contract
Epic 16.6 — Artifact Quarantine + Takes
Epic 16.7 — LLM Planner JSON
Epic 16.8 — MIDI-GPT backend
Epic 16.9 — Text2MIDI sketch backend
```

## Epic 17 — GrooveMap y PerformanceMap

Ampliación con IA:

- PerformanceMap debe aplicarse también a material importado de modelos.
- No duplicar humanización si el artifact ya trae microtiming expresivo.
- Añadir `performance_source`: `rule_based | imported_model | normalized_model`.
- Normalizar velocities de modelos antes de exportar.

Acceptance:

- Un artifact IA aceptado suena integrado con el resto.
- No rompe MusicXML.
- No desplaza notas fuera de compás.

## Epic 18 — Melody Engine 2.0

Modos:

```text
rule_based
retrieval
ai_infill
```

IA solo para:

- variaciones;
- compases concretos;
- solos/secciones marcadas;
- respuesta a un motivo existente.

Acceptance:

- MelodyEngine puede pedir a backend simbólico y validar.
- Fallback rule-based si IA falla.
- MotifLedger registra material IA aceptado.

## Epic 19 — Bass Engine 2.0

Prioridad:

```text
rule_based -> retrieval -> statistical -> future custom model
```

No priorizar MIDI-GPT para bajo completo hasta tener validadores fuertes.

Acceptance:

- AI bass solo permitido si supera direction/approach/contour validators.

## Epic 20 — Drums Engine 2.0

Prioridad:

```text
GrooveMap + patterns + retrieval + PerformanceMap
```

IA externa no prioritaria para batería en esta fase.

Acceptance:

- Los fills se coordinan con SongPlan y horn hits.

## Epic 21 — Piano/Comping Engine 2.0

IA útil para:

- variaciones de comping;
- reharmonización controlada;
- compases de puente;
- fills entre frases.

Validadores obligatorios:

- registro;
- polifonía;
- duplicación con bajo;
- voice-leading distance;
- density by section.

## Epic 22 — Dataset profiler / Role classifier

Debe alimentar tanto retrieval como futuros modelos.

Añadir:

- export de segmentos para MidiTok;
- manifest de licencia;
- role confidence;
- pattern sensitivity flag;
- no-memorization fingerprints.

## Epic 23 — Retrieval model

Debe ser el puente entre rule-based e IA neural.

```text
PatternRetriever -> PatternAdapter -> ValidationGate -> Take
```

## Epic 24 — App workflow 2.0

Añadir UI de IA:

- AI Plan;
- AI Infill;
- AI Generate Track;
- AI Sketch;
- Take Manager;
- Validation diff.

## Epic 25 — DAW-ready export

Solo exportar takes accepted.

Añadir a ZIP:

```text
model_trace.json
validation_report.html
takes_manifest.json
```

## Epic 26 — Release quality gate

Añadir gates IA:

- no artifact pending en export release;
- no backend noncommercial en build comercial;
- no rejected artifacts sin reporte;
- no similitud excesiva.
