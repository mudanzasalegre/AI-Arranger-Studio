# 01 — Flujo de trabajo para Codex

## Modo de trabajo

Codex debe trabajar por objetivos completos, no por micro-PRs. Cada objetivo tiene contrato, entregables y criterios de aceptación en `docs/04_OBJECTIVES.md`.

## Comando maestro para cada objetivo

```text
Implementa el Objetivo N de docs/04_OBJECTIVES.md completo. Antes de empezar, lee los documentos relacionados. No modifiques el alcance del objetivo. Al acabar, ejecuta tests, lint y un smoke test de generación/exportación si aplica. Documenta lo hecho en docs/progress/OBJ_N.md.
```

## Reglas de implementación

1. Mantener el núcleo musical desacoplado de FastAPI y Next.js.
2. No meter lógica musical compleja dentro de controladores API.
3. No hardcodear conocimiento musical dentro de vistas web.
4. Todo catálogo musical debe vivir en `configs/` o en módulos claros del core.
5. Todo generador debe recibir `seed` y ser determinista si la misma entrada y seed se repiten.
6. Todo output debe generar reporte de validación.
7. Los tests musicales importan tanto como los tests de software.
8. Si una decisión musical es dudosa, preferir warning y configuración antes que comportamiento oculto.

## Definition of Done global

Cada objetivo queda cerrado solo si:

- `make lint` pasa.
- `make test` pasa.
- se actualiza documentación si cambió el contrato.
- se añade al menos un test de integración si el objetivo toca exportación/generación.
- no se rompe el formato `ArrangementProject` sin migración o versión.
- los archivos generados están bajo `outputs/` y no contaminan el repo.

## Política de deuda técnica

- Si algo queda simulado, debe marcarse como `TODO_AI`, `TODO_MUSIC`, `TODO_EXPORT` o `TODO_UI`.
- No dejar funciones vacías que aparenten funcionar.
- No devolver MIDI “fake” sin notas reales.
- No llamar “professional” a una salida que no pase validadores básicos.

## Orden obligatorio

1. Objetivo 0: repo ejecutable.
2. Objetivo 1: modelo simbólico.
3. Objetivo 2: conocimiento musical jazz.
4. Objetivo 3: export MIDI/MusicXML/PDF.
5. Objetivo 4: generadores rule-based.
6. Objetivo 5: validación.
7. Objetivo 6: dataset learning mode.
8. Objetivo 7: API.
9. Objetivo 8: web.
10. Objetivo 9: presets potentes.
11. Objetivo 10: preparación IA.
