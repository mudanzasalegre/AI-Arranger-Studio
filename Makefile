.PHONY: setup setup-web test lint api web demo-jazz export-demo validate-demo zip-demo package-smoke ai-contract-smoke tokenization-smoke statistical-smoke custom-role-model-smoke train-symbolic-model golden-baseline demo-check build-web

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
