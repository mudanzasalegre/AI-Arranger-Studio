# Entrenamiento simbolico - JAZZVAR + RELEASE2.0

## Estado

Completado.

## Fuentes incluidas

Solo se usaron estas carpetas:

- `midi_databases/JAZZVAR_DATASET`
- `midi_databases/RELEASE2.0_mid_unquant`

El script `scripts/train_symbolic_model.py` valida esas rutas de forma estricta y
aborta si se intenta entrenar con cualquier otra carpeta de `midi_databases`.

## Comando ejecutado

```powershell
python scripts/train_symbolic_model.py
```

## Resultado

- archivos descubiertos: 14.596
- archivos procesados: 14.596
- archivos fallidos: 0
- patrones extraidos raw: 121.543
- patrones unicos: 54.664
- training examples: 54.664
- splits: train 43.731, val 5.466, test 5.467
- patrones guardados en el artefacto compacto: 2.000

Categorias extraidas:

- `piano_voicings`: 51.016
- `melodic_motifs`: 3.648

El extractor actual no encontro `walking_bass_cells` en este corpus. El smoke de
generacion/exportacion paso igualmente porque el backend simbolico conserva un
fallback determinista para roles sin patron entrenado.

## Artefactos

- `outputs/models/jazzvar_release2_symbolic/dataset_manifest.json`
- `outputs/models/jazzvar_release2_symbolic/pattern_index.json`
- `outputs/models/jazzvar_release2_symbolic/training/training_examples.jsonl`
- `outputs/models/jazzvar_release2_symbolic/training/dataset_splits.json`
- `outputs/models/jazzvar_release2_symbolic/training/feature_store.json`
- `outputs/models/jazzvar_release2_symbolic/model/symbolic_pattern_model.json`
- `outputs/models/jazzvar_release2_symbolic/training_run_summary.json`
- `outputs/models/jazzvar_release2_symbolic/smoke_export/`

## Verificacion

Ejecutado:

- `python -m ruff check apps packages scripts tests` - OK
- `python -m pytest -q` - OK, 76 tests
- `npm --prefix apps/web run lint` - OK
- smoke integrado en `python scripts/train_symbolic_model.py` - OK
