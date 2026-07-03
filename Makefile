.PHONY: setup setup-web test lint api web demo-jazz export-demo validate-demo zip-demo package-smoke ai-contract-smoke tokenization-smoke statistical-smoke custom-role-model-smoke train-symbolic-model golden-baseline demo-check build-web models-bootstrap models-check midigpt-download midigpt-smoke text2midi-download text2midi-smoke ollama-planner-smoke miditok-smoke custom-role-models-smoke ai-local-smoke professional-benchmark pro-readiness-audit pro-activate-profile pro-install-models pro-verify-models pro-midigpt-infill-smoke pro-text2midi-sketch-smoke pro-ollama-planner-endpoint-smoke pro-miditok-manifest-smoke pro-train-custom-role-ngram-models pro-smoke-custom-role-ngram-models pro-generate-professional-midi pro-quality-gate pro-professional-benchmark-gate model-worker

setup:
	python -m pip install -U pip
	python -m pip install -r requirements.txt

setup-web:
	npm --prefix apps/web install

test:
	pytest -q

lint:
	ruff check apps packages scripts tests
	npm --prefix apps/web run lint

api:
	python -m uvicorn app.main:app --app-dir apps/api --reload --port 8000

web:
	npm --prefix apps/web run dev

demo-jazz:
	python scripts/demo_jazz.py

export-demo:
	python scripts/demo_jazz.py

validate-demo:
	python scripts/validate_demo.py

zip-demo:
	python scripts/zip_demo.py

package-smoke:
	python scripts/package_smoke.py

ai-contract-smoke:
	python scripts/ai_contract_smoke.py

tokenization-smoke:
	python scripts/tokenization_dataset_smoke.py

statistical-smoke:
	python scripts/statistical_baselines_smoke.py

custom-role-model-smoke:
	python scripts/custom_role_model_smoke.py

train-symbolic-model:
	python scripts/train_symbolic_model.py

golden-baseline:
	python scripts/golden_generate.py

demo-check: package-smoke

build-web:
	npm --prefix apps/web run build

models-bootstrap:
	python scripts/models/ensure_local_model_dirs.py

models-check:
	python scripts/models/check_local_model_runtime.py --config configs/local_model_runtime.yaml

midigpt-download:
	python scripts/models/download_midigpt.py

midigpt-smoke:
	python scripts/models/smoke_midigpt.py

text2midi-download:
	python scripts/models/download_text2midi.py --checkpoint-dir models/checkpoints/text2midi

text2midi-smoke:
	python scripts/models/smoke_text2midi.py --allow-check-only

ollama-planner-smoke:
	python scripts/models/smoke_ollama_planner.py

miditok-smoke:
	python scripts/models/smoke_miditok.py

custom-role-models-smoke:
	python scripts/models/smoke_custom_role_models.py

ai-local-smoke:
	python scripts/models/ai_local_smoke.py

professional-benchmark:
	python scripts/models/professional_generation_benchmark.py --config configs/professional_benchmarks.yaml

pro-readiness-audit:
	python scripts/models_pro/pro_readiness_audit.py

pro-activate-profile:
	python scripts/models_pro/activate_pro_profile.py

pro-install-models:
	python scripts/models_pro/install_all_local_models.py --profile pro --planner-model qwen3:8b

pro-verify-models:
	python scripts/models_pro/verify_all_models.py

pro-midigpt-infill-smoke:
	python scripts/models_pro/midigpt_project_infill_smoke.py

pro-text2midi-sketch-smoke:
	python scripts/models_pro/text2midi_sketch_import_smoke.py

pro-ollama-planner-endpoint-smoke:
	python scripts/models_pro/ollama_planner_endpoint_smoke.py

pro-miditok-manifest-smoke:
	python scripts/models_pro/miditok_dataset_from_manifest_smoke.py

pro-train-custom-role-ngram-models:
	python scripts/models_pro/train_custom_role_ngram_models.py

pro-smoke-custom-role-ngram-models:
	python scripts/models_pro/smoke_custom_role_ngram_models.py

pro-generate-professional-midi:
	python scripts/models_pro/generate_professional_midi.py --profile pro

pro-quality-gate:
	python scripts/models_pro/pro_quality_gate.py --output-dir outputs/pro_benchmarks/pr35_acceptance

pro-professional-benchmark-gate:
	python scripts/models_pro/professional_benchmark_gate.py

model-worker:
	python -m uvicorn app.main:app --app-dir apps/model_worker --reload --port 8010
