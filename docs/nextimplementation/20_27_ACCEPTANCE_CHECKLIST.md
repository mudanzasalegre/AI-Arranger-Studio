# Checklist de aceptación — PR-20 a PR-27

## PR-20

- [ ] `configs/local_model_runtime.yaml` creado localmente.
- [ ] `configs/ai_models.local.yaml` creado localmente.
- [ ] `.env` contiene `AI_MODELS_ROOT`, `HF_HOME`, `HF_HUB_CACHE`, `AI_MODELS_CONFIG`.
- [ ] `python scripts/models/ensure_local_model_dirs.py` crea carpetas.
- [ ] `python scripts/models/check_local_model_runtime.py` pasa.
- [ ] API funciona sin modelos reales.

## PR-21

- [ ] `pip install "midigpt[inference]"` documentado/ejecutado en entorno local.
- [ ] `python scripts/models/download_midigpt.py --model-name yellow` pasa.
- [ ] `python scripts/models/smoke_midigpt.py` genera MIDI.
- [ ] Backend usa API real de MIDI-GPT.
- [ ] `/ai/infill` con midigpt crea take pendiente.
- [ ] Proyecto activo no cambia hasta accept.

## PR-22

- [ ] Repo Text2MIDI clonado en `models/external_repos/text2midi`.
- [ ] Checkpoints en `models/checkpoints/text2midi`.
- [ ] `smoke_text2midi.py` genera MIDI.
- [ ] Sketch importado como `sketch_validated` o `sketch_uncertain`.
- [ ] Sketch no se exporta automáticamente como final.

## PR-23

- [ ] Ollama instalado.
- [ ] Modelo planner descargado.
- [ ] `smoke_ollama_planner.py` devuelve JSON válido.
- [ ] `/ai/plan` usa provider cuando está habilitado.
- [ ] Fallback rule-based funciona.

## PR-24

- [ ] `miditok` instalado opcionalmente.
- [ ] Tokenización real por rol implementada.
- [ ] Segmentos tienen metadata de licencia.
- [ ] Datasets no permitidos quedan fuera de train.

## PR-25

- [ ] Custom role backends aparecen en registry.
- [ ] Modelos sin manifest/licencia están unavailable.
- [ ] Dummy backend para tests funciona.

## PR-26

- [ ] `ai_local_smoke.py` pasa con mock.
- [ ] Si midigpt enabled, lo prueba.
- [ ] Si text2midi enabled, lo prueba.
- [ ] Artifacts tienen estado final.
- [ ] Takes se pueden aceptar/rechazar.

## PR-27

- [ ] 5 benchmarks corren.
- [ ] 0 errores bloqueantes.
- [ ] MIDI y MusicXML exportados.
- [ ] model_trace generado si hubo IA.
- [ ] ZIP final no contiene pending takes.
