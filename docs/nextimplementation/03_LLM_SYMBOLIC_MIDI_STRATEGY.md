# 03 — Estrategia LLM para MIDI simbólico profesional

## Tipos de modelos y uso correcto

### LLM Planner

Uso:

```text
prompt libre -> SongPlan/SectionPlan/PhrasePlan/GrooveMap patch
```

No genera notas.

### MIDI-GPT

Uso recomendado:

```text
ArrangementProject existente + track_id + bars -> nuevo material MIDI para esos compases
```

Tareas:

- `infill_bars`;
- `generate_track`;
- `continue_section`;
- `generate_variation` si se encapsula como infill.

No usar como generador final de canción completa en esta fase.

### Text2MIDI

Uso recomendado:

```text
prompt rico -> MIDI sketch -> ArrangementProject sketch
```

Sirve para bocetos, inspiración y comparación. No sustituye al motor profesional.

### MidiTok / modelos propios futuros

Uso futuro:

```text
Dataset curado -> tokens -> modelos por rol
```

Roles recomendados:

- melody;
- walking_bass;
- piano_comping;
- horns;
- drums/fills.

## Flujo recomendado por caso de uso

### Caso A — Crear una canción nueva desde prompt

```text
1. Usuario escribe prompt.
2. LLM Planner o PromptCompiler produce SongPlan.
3. Harmony/Form engine genera progresión.
4. Rule-based engine genera arreglo base.
5. Validadores aceptan base.
6. Opcional: modelos simbólicos mejoran zonas concretas.
7. Usuario acepta takes.
8. Export.
```

### Caso B — Mejorar melodía débil

```text
1. Usuario selecciona track alto_sax y bars 17-24.
2. Se construye RoleIntent para sax melody/solo.
3. MIDI-GPT genera candidato.
4. Artifact quarantine.
5. Melody/Harmony/Breath validators.
6. Si pasa, Take B pending.
7. Usuario acepta.
```

### Caso C — Generar boceto externo

```text
1. Usuario manda prompt a Text2MIDI.
2. Text2MIDI genera MIDI sketch.
3. Importer crea ArrangementProject sketch.
4. TrackRoleClassifier intenta mapear roles.
5. ValidationGate marca confianza.
6. Usuario puede usarlo como referencia, no como proyecto final.
```

### Caso D — Regenerar vientos

```text
1. SongPlan define huecos y hits.
2. RoleIntentBuilder crea restricciones: instruments, bars, density, allowed beats.
3. MIDI-GPT genera voicing/horn responses.
4. Horn validators corrigen/rechazan.
5. ProjectMerger fusiona solo vientos autorizados.
```

## Política de fallback

Si el backend IA falla:

```text
retry 1: bajar temperatura / densidad;
retry 2: instrucción de reparación basada en errores;
retry 3: fallback rule-based/retrieval;
```

Nunca devolver error bruto al usuario como resultado musical final. Debe haber fallback.

## Política de reproducibilidad

Cada request debe registrar:

```text
backend_id
backend_version
model_name
seed
temperature
density
instruction
source_project_take
target_track
target_bars
validation_report_id
```

Si el backend no garantiza reproducibilidad, registrar:

```text
reproducibility: best_effort
```
