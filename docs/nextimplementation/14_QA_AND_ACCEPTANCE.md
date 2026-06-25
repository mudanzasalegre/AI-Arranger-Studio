# 14 — QA y criterios de aceptación con LLM

## Gates técnicos

Por cada epic:

```powershell
python -m ruff check apps packages scripts tests
python -m pytest -q
npm --prefix apps/web run lint
```

Si hay frontend:

```powershell
npm --prefix apps/web run build
```

Si hay generación/exportación:

```powershell
python scripts/golden_generate.py
```

Si hay packaging:

```powershell
python scripts/package_smoke.py
```

## Gates IA

Añadir:

```text
- backend opcional ausente no rompe app;
- artifacts raw no se exportan;
- artifact inválido no modifica proyecto;
- take pending no se exporta como final;
- take rejected queda trazada;
- take accepted pasa validación;
- fallback funciona;
- no se usa backend de audio;
- no se usa backend con licencia incompatible si export comercial está activado.
```

## Métricas musicales mínimas

### Generales

- compases completos;
- rango instrumental;
- chord-tone/tension ratio;
- densidad por sección;
- similitud con fuentes;
- export success;
- no pistas vacías.

### Melodía

- motivo recurrente;
- respiración;
- rango cómodo;
- resolución en cadencia;
- variación sin caos.

### Bajo

- dirección entre acordes;
- approach notes;
- contour smoothness;
- relación con kick.

### Piano

- voice-leading distance;
- registro;
- duplicación grave;
- densidad;
- espacio con melodía.

### Vientos

- spacing;
- cruce de voces;
- rango;
- breath score;
- hits en huecos reales.

## Tests específicos IA

```text
test_model_registry_lists_mock_backend
test_missing_midigpt_dependency_is_controlled
test_ai_artifact_saved_to_raw
test_invalid_artifact_goes_to_rejected
test_valid_artifact_creates_pending_take
test_take_accept_changes_active_take
test_take_reject_preserves_active_take
test_ai_infill_only_modifies_target_bars
test_llm_planner_invalid_json_fallback
test_text2midi_sketch_not_exported_as_final
```

## Golden demos

Cada golden preset debe probar:

- generación rule-based;
- AI infill mock;
- validation report;
- export MIDI/MusicXML;
- take manifest.

## Release gate

No marcar release si:

- hay errors bloqueantes;
- hay artifacts pending;
- hay rejected artifacts sin reporte;
- export usa backend no comercial en modo comercial;
- alguna demo golden baja de score 3/5;
- MIDI/MusicXML no se abre/importa.
