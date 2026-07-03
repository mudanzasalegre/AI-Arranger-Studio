# scripts/models

Scripts auxiliares para PR-20 a PR-27.

Orden recomendado:

```bash
python scripts/models/ensure_local_model_dirs.py
python scripts/models/check_local_model_runtime.py --config configs/local_model_runtime.yaml
python scripts/models/download_midigpt.py --model-name yellow
python scripts/models/smoke_midigpt.py
python scripts/models/download_text2midi.py
python scripts/models/smoke_text2midi.py
python scripts/models/smoke_ollama_planner.py --model qwen3:8b
python scripts/models/ai_local_smoke.py
python scripts/models/professional_generation_benchmark.py --config configs/professional_benchmarks.yaml
```

Todos los scripts deben fallar con errores claros si falta una dependencia opcional.
