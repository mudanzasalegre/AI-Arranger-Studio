# 12 — Workflow frontend para IA simbólica

## Objetivo

Añadir IA sin convertir la interfaz en una caja negra. El usuario debe ver qué se generó, por qué se acepta/rechaza y poder regenerar solo lo malo.

## Pantallas nuevas o ampliadas

### AI Plan Panel

Funciones:

- prompt libre;
- vista JSON del plan generado;
- validación del plan;
- aplicar como nuevo plan o patch.

### AI Regenerate Panel

Campos:

- backend;
- track;
- sección;
- compases;
- instrucción;
- densidad;
- complejidad;
- seed;
- locked tracks.

### Take Manager

Lista:

```text
Take A — accepted — rule-based
Take B — pending — midigpt alto_sax bars 17-24
Take C — rejected — midigpt trumpet bars 25-28
```

Acciones:

- preview;
- diff;
- accept;
- reject;
- duplicate;
- restore.

### Validation Diff

Mostrar:

- errores bloqueantes;
- warnings;
- métricas musicales;
- cambios por track/bars;
- comparación contra take activa.

### Sketch Workspace

Text2MIDI debe ir a un área separada:

```text
Sketches
```

No mezclar directamente con proyectos profesionales.

## UX no negociable

- Ninguna generación IA se acepta silenciosamente.
- Cada resultado tiene `backend`, `seed`, `track`, `bars`, `validation`.
- El usuario puede escuchar/visualizar antes de aceptar.
- Botón claro: `Accept take`.
- Botón claro: `Reject take`.

## Estados UI

```text
idle
generating
importing
validating
pending_review
accepted
rejected
error
fallback_used
```

## Acceptance criteria

- Se puede hacer AI infill desde UI con mock.
- Se ve validation report.
- Se acepta/rechaza take.
- Text2MIDI sketch no aparece como export final por defecto.
