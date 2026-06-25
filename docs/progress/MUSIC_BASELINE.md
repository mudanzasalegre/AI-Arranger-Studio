# Music baseline - Epic 15

## Estado

Completado y congelado para PR-01.

## Implementado

- `scripts/golden_generate.py`
- target `make golden-baseline`
- `outputs/golden/<preset>/music_metrics.json`
- `outputs/golden/golden_summary.json`
- `outputs/golden/golden_summary.md`
- tests de regresion en `tests/test_music_baseline.py`

## Comando ejecutado

```powershell
python scripts/golden_generate.py
```

## Resultado global

- presets generados: 8
- total note events: 5.504
- validation: todos `pass`
- quality flags heuristicas: 10
- score heuristico medio: 4.362/5
- estado PR-01: sin cambios en metricas respecto a la ultima foto golden

Importante: el score es estructural y sirve para priorizar regresiones. No es un
veredicto auditivo. La escucha reportada indica que el resultado sigue sonando
incoherente y mecanico; este baseline existe precisamente para empezar a medir y
corregir eso.

## Congelacion Epic 16

El baseline actual ya exporta el plan musical global junto con cada preset:

- `song_plan.json` existe en los 8 paquetes golden.
- `arrangement_project.json` contiene `metadata.song_plan`.
- `metadata.song_plan.sections` guarda `SectionPlan`.
- `metadata.song_plan.phrases` guarda `PhrasePlan`.
- `metadata.song_plan.groove_map` guarda `GrooveMap`.
- Los generadores reciben el plan mediante `GenerationContext.song_plan`.

Prueba automatizada:

- `tests/test_song_planner.py::test_generators_receive_global_song_plan_and_project_metadata_exports_it`

## Artefactos

- `outputs/golden/golden_summary.json`
- `outputs/golden/golden_summary.md`
- `outputs/golden/jazz_ballad_quartet/music_metrics.json`
- `outputs/golden/jazz_bebop_blues_quintet/music_metrics.json`
- `outputs/golden/jazz_bossa_nova_quartet/music_metrics.json`
- `outputs/golden/jazz_funk_straight_eighth_quintet/music_metrics.json`
- `outputs/golden/jazz_hard_bop_minor_blues_sextet/music_metrics.json`
- `outputs/golden/jazz_modal_quintet/music_metrics.json`
- `outputs/golden/jazz_swing_aaba_quartet/music_metrics.json`
- `outputs/golden/jazz_waltz_trio/music_metrics.json`

## Hallazgos por preset

| Preset | Validacion | Flags | Observacion principal |
| --- | --- | ---: | --- |
| `jazz_ballad_quartet` | pass | 1 | Densidad por seccion demasiado plana; falta arco narrativo. |
| `jazz_bebop_blues_quintet` | pass | 1 | Piano con densidad alta; puede saturar. |
| `jazz_bossa_nova_quartet` | pass | 2 | Energia plana y comping denso; el bajo conecta peor hacia siguiente raiz. |
| `jazz_funk_straight_eighth_quintet` | pass | 1 | Comping muy denso para un groove que deberia respirar. |
| `jazz_hard_bop_minor_blues_sextet` | pass | 1 | Piano denso; necesita coordinacion real con vientos/bateria. |
| `jazz_modal_quintet` | pass | 1 | Comping denso; falta desarrollo de tension modal. |
| `jazz_swing_aaba_quartet` | pass | 2 | Energia plana y piano denso; las secciones A/B no contrastan suficiente. |
| `jazz_waltz_trio` | pass | 1 | Energia plana; requiere fraseo de vals y narrativa de seccion. |

## Problemas auditivos esperados

- Los roles se validan por separado, pero no conversan entre si.
- El piano acompana con demasiadas notas por compas en varios presets.
- Las secciones no tienen suficiente contraste de energia.
- Las metricas actuales no capturan suficientemente motivo, groove, swing,
  interpretacion MIDI ni calidad de mezcla.
- La bateria sigue siendo demasiado gramatical/previsible.
- El bajo marca bien beat 1, pero eso no garantiza que la linea sea musical.

## Verificacion

Ejecutado:

- `python -m ruff check scripts/golden_generate.py tests/test_music_baseline.py` - OK tras ajuste
- `python -m pytest tests/test_music_baseline.py -q` - OK
- `python scripts/golden_generate.py` - OK
- `python -m pytest tests/test_song_planner.py tests/test_music_baseline.py -q` - 6 passed
- `python -m ruff check apps packages scripts tests` - OK
- `python -m pytest -q` - 82 passed, 1 warning de `StarletteDeprecationWarning`
- `npm --prefix apps/web run lint` - OK
- `python scripts/golden_generate.py` - OK, 8 presets
