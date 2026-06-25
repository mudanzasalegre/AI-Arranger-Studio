# 00 — Contexto y decisiones de integración

## Punto de partida

El proyecto ya ha avanzado hasta una arquitectura con planes musicales:

```text
SongPlan
SectionPlan
PhrasePlan
GrooveMap
```

Eso cambia completamente la forma correcta de integrar modelos. Si no existiera esta capa, tendría sentido usar un modelo text-to-MIDI para generar bocetos completos. Pero al existir `SongPlan`, el modelo debe integrarse **dentro del motor simbólico**, no por encima de él.

## Objetivo de esta fase

Integrar LLM/modelos simbólicos para mejorar la generación profesional de MIDI sin introducir audio, sin perder trazabilidad y sin convertir la app en una caja negra.

La fase debe permitir:

- planificar una canción desde prompt mediante JSON validado;
- regenerar pistas concretas;
- rellenar compases concretos;
- crear bocetos MIDI experimentales desde prompt;
- guardar cada resultado como take;
- validar antes de aceptar;
- mantener exportación MIDI/MusicXML/PDF profesional;
- dejar preparada una capa para modelos propios futuros.

## Fuera de alcance

Queda fuera de esta fase:

- generación de audio;
- voces cantadas;
- stems;
- separación audio-a-MIDI;
- entrenamiento grande desde cero;
- Fine-tuning de modelos externos;
- publicación comercial de modelos con licencias no revisadas.

## Decisiones no negociables

### 1. El formato maestro sigue siendo interno

El modelo puede generar un MIDI o tokens, pero el sistema debe convertirlo siempre a:

```text
ArrangementProject / SongPlan-compatible internal representation
```

Nunca se acepta un `.mid` como verdad final.

### 2. Los modelos son backends sustituibles

Todo modelo externo debe cumplir un contrato común:

```text
MusicModelBackend.generate(ModelGenerationRequest) -> ModelGenerationResult
```

Así se podrá cambiar MIDI-GPT por otro modelo, un servicio HTTP, un modelo local propio o un backend mock sin tocar API/frontend/validadores.

### 3. Artifact quarantine obligatorio

Toda salida de IA entra en:

```text
outputs/model_artifacts/raw
```

Luego pasa por importación, fusión, validación y creación de take. Si falla, va a:

```text
outputs/model_artifacts/rejected
```

Si pasa, va a:

```text
outputs/model_artifacts/validated
```

### 4. Sistema de takes

Los modelos no sobrescriben el proyecto activo. Crean alternativas:

```text
Take A — rule-based actual
Take B — MIDI-GPT sax bars 17-24
Take C — MIDI-GPT sax retry lower temperature
Take D — fallback rule-based repaired
```

El usuario o un gate explícito acepta/rechaza.

### 5. Audio queda desactivado

No añadir backends de audio hasta que la generación MIDI/MusicXML sea sólida.

## Filosofía musical

Un LLM puede proponer material interesante, pero no entiende por sí solo todas las restricciones profesionales:

- tesitura real;
- respiración;
- transposición;
- conducción de voces;
- relación con acordes;
- densidad por sección;
- interacción con batería/bajo/piano;
- exportabilidad MusicXML;
- utilidad en DAW.

Por tanto, la calidad vendrá de:

```text
modelos simbólicos
+ motor rule-based
+ dataset/retrieval
+ validadores musicales
+ revisión por takes
+ export profesional
```
