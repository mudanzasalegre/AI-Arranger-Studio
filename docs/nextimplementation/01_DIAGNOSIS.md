# 01 — Diagnóstico musical y técnico actualizado

## Problema anterior

El sistema era capaz de generar archivos correctos, pero no siempre música convincente. Los síntomas principales eran:

- forma sin narrativa;
- melodía débil;
- bajo sin dirección suficiente;
- piano/comping demasiado estático;
- batería poco real;
- vientos poco idiomáticos;
- MIDI plano;
- validadores centrados en estructura, no en musicalidad.

## Nuevo riesgo al integrar LLM

Los modelos pueden producir resultados más variados, pero también pueden introducir problemas nuevos:

```text
- notas fuera de rango;
- duraciones que no cuadran;
- pistas mal asignadas;
- densidad excesiva;
- pérdida del groove planificado;
- modificaciones fuera de los compases pedidos;
- melodías que no respetan cadencias;
- comping que pisa al bajo;
- vientos imposibles de respirar;
- material parecido a datos de entrenamiento;
- resultados no reproducibles.
```

Por eso la integración LLM no debe ser un generador final, sino una fuente de **candidatos controlados**.

## Diagnóstico de arquitectura

El punto actual es bueno porque ya existe:

```text
SongPlan -> SectionPlan -> PhrasePlan -> GrooveMap
```

Pero falta una capa de entrada/salida para modelos:

```text
ModelBackendContract
ArtifactQuarantine
ProjectMerger
ValidationGate
TakeManager
```

Sin estas piezas, cualquier integración de MIDI-GPT/Text2MIDI acabaría contaminando el proyecto activo.

## Nuevo estándar de calidad con IA

Una generación asistida por IA solo se considera válida si:

- respeta el plan musical;
- toca solo el rango autorizado;
- no rompe compases;
- no modifica pistas bloqueadas;
- supera validadores de rango, armonía, respiración y exportación;
- queda registrada como take;
- es reproducible por `seed` cuando el backend lo permita;
- deja trazabilidad de backend, modelo, prompt, parámetros y validación;
- se puede exportar a MIDI/MusicXML sin edición manual.

## Nueva pregunta de aceptación

Antes la pregunta era:

```text
¿Genera archivos válidos?
```

Ahora debe ser:

```text
¿Genera una take musicalmente útil, validada, trazable y exportable?
```
