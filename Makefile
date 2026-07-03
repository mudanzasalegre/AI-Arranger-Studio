.PHONY: setup setup-web test lint api web demo-jazz export-demo validate-demo zip-demo package-smoke ai-contract-smoke tokenization-smoke statistical-smoke custom-role-model-smoke train-symbolic-model golden-baseline demo-check build-web models-bootstrap models-check midigpt-download midigpt-smoke text2midi-download text2midi-smoke ollama-planner-smoke miditok-smoke custom-role-models-smoke ai-local-smoke professional-benchmark model-worker

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

model-worker:
	python -m uvicorn app.main:app --app-dir apps/model_worker --reload --port 8010
