# AI Arranger Studio

AI Arranger Studio is a symbolic music arrangement system for generating editable,
DAW-ready jazz arrangements from text prompts. It focuses on structured musical
output first: MIDI, MusicXML, validation reports, takes, and model traces. It does
not generate final audio directly.

The project combines deterministic arranging engines, dataset-derived pattern
retrieval, local symbolic model adapters, and a FastAPI/Next.js application shell.

## What It Does

- Compiles natural-language prompts into structured `GenerationSpec` objects.
- Generates multitrack jazz arrangements with form, harmony, song plan, groove,
  bass, piano, drums, melody, and horn-response roles.
- Exports full MIDI, per-track MIDI, full-score MusicXML, optional MuseScore PDFs,
  validation reports, session readmes, take manifests, and model traces.
- Supports manual accept/reject workflows for AI-generated takes before final export.
- Quarantines local model artifacts through raw, imported, validated, and rejected
  states.
- Provides local AI integrations for MIDI-GPT, Text2MIDI sketches, Ollama planning,
  MidiTok training workflows, and custom role-model bootstrap checkpoints.
- Ships reproducible professional benchmark cases for jazz generation quality.

## Current Scope

The system is intentionally symbolic-first:

```text
Prompt
  -> GenerationSpec
  -> ArrangementProject
  -> rule-based / retrieval arrangement
  -> optional local symbolic model infill
  -> artifact import and validation
  -> pending take
  -> accept or reject
  -> DAW-ready export package
```

Models do not write directly into final exports. Every model result is imported,
validated, traced, and accepted through the take system.

## Repository Layout

```text
apps/api/               FastAPI service and project/take/export endpoints
apps/web/               Next.js UI shell with score rendering support
apps/model_worker/      Optional local model worker scaffold
packages/arranger_core/ Core music schema, generation, validation, export, takes
packages/dataset_tools/ Dataset import, profiling, and pattern extraction
packages/midi_models/   Symbolic baseline model interfaces
packages/model_backends/Local model backend registry and adapters
packages/training/      Training/tokenization helpers, including MidiTok
scripts/                Demo, packaging, validation, and training scripts
scripts/models/         Local model bootstrap, smoke tests, and benchmarks
configs/                Public config templates and safe defaults
docs/                   Architecture, implementation plan, QA, and roadmap docs
```

Generated outputs, local configs, datasets, model checkpoints, and private MIDI
files are intentionally ignored by Git.

## Requirements

Base runtime:

- Python 3.12
- Node.js 18+ for the web app
- MuseScore CLI is optional, only needed for PDF score export
- Ollama is optional, only needed for the local LLM planner

Optional AI dependencies are installed separately because they can pull large
packages and model runtimes.

## Quick Start

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
npm --prefix apps/web install
```

Run core checks:

```powershell
python -m ruff check apps packages scripts tests
python -m pytest -q
npm --prefix apps/web run lint
```

Start the API and web app in separate terminals:

```powershell
python -m uvicorn app.main:app --app-dir apps/api --reload --port 8000
npm --prefix apps/web run dev
```

The API runs at `http://127.0.0.1:8000`. The web app runs at
`http://127.0.0.1:3000`.

If GNU Make is available, equivalent targets exist:

```bash
make setup
make setup-web
make lint
make test
make api
make web
```

## Generate a Project

Compile a prompt:

```powershell
python scripts/compile_prompt.py --prompt "hard bop nocturno en Do menor, 132 bpm, blues menor, sexteto con saxo alto, trompeta, trombon, piano, contrabajo y bateria" --seed 1234
```

Generate/export through the API:

```powershell
$body = @{
  project_id = "demo_hard_bop"
  prompt = "hard bop nocturno en Do menor, 132 bpm, blues menor, sexteto con saxo alto, trompeta, trombon, piano, contrabajo y bateria"
  seed = 1234
  options = @{
    export = $true
    validate = $true
    include_pdf = $false
  }
} | ConvertTo-Json -Depth 8

Invoke-RestMethod -Uri http://127.0.0.1:8000/v1/projects/generate -Method Post -ContentType application/json -Body $body
```

Download a ZIP package:

```powershell
Invoke-WebRequest -Uri http://127.0.0.1:8000/v1/projects/demo_hard_bop/zip -OutFile outputs/demo_hard_bop.zip
```

## Main API Surface

- `GET /health`
- `GET /v1/ai/models`
- `POST /v1/prompts/compile`
- `POST /v1/projects/generate`
- `GET /v1/projects/{project_id}`
- `POST /v1/projects/{project_id}/export`
- `GET /v1/projects/{project_id}/zip`
- `GET /v1/projects/{project_id}/validation`
- `POST /v1/projects/{project_id}/ai/plan`
- `POST /v1/projects/{project_id}/ai/infill`
- `GET /v1/projects/{project_id}/takes`
- `POST /v1/projects/{project_id}/takes/{take_id}/accept`
- `POST /v1/projects/{project_id}/takes/{take_id}/reject`
- `POST /v1/ai/text-to-midi-sketch`
- `POST /v1/datasets/import`
- `GET /v1/patterns/search`

## Local Model Runtime

Local model support is configured with ignored, machine-local files. Create them
from the examples:

```powershell
Copy-Item configs/local_model_runtime.example.yaml configs/local_model_runtime.yaml
Copy-Item configs/ai_models.local.example.yaml configs/ai_models.local.yaml
Copy-Item configs/model_registry.example.yaml configs/model_registry.yaml
```

Merge `.env.local-models.example` into your local `.env`. Do not commit `.env` or
the copied local YAML files.

Bootstrap and verify local directories:

```powershell
python scripts/models/ensure_local_model_dirs.py
python scripts/models/check_local_model_runtime.py --config configs/local_model_runtime.yaml
```

This prepares:

```text
models/hf_cache/hub
models/external_repos
models/checkpoints/text2midi
models/checkpoints/custom/...
outputs/model_artifacts/raw
outputs/model_artifacts/imported
outputs/model_artifacts/rejected
outputs/model_artifacts/validated
```

Install optional AI dependencies only when needed:

```powershell
python -m pip install -r requirements-ai.txt
python -m pip install -r requirements-training-ai.txt
```

Useful local model commands:

```bash
make models-bootstrap
make models-check
make midigpt-download
make midigpt-smoke
make text2midi-download
make text2midi-smoke
make ollama-planner-smoke
make miditok-smoke
make custom-role-models-smoke
make ai-local-smoke
```

## Professional Benchmark

PR-27 adds a reproducible benchmark suite for local professional-generation QA:

```powershell
python scripts/models/professional_generation_benchmark.py --config configs/professional_benchmarks.yaml --api http://127.0.0.1:8000
```

The benchmark writes:

```text
outputs/professional_benchmark/
  summary.json
  summary.md
  <case_id>/
    arrangement_project.json
    full_arrangement.mid
    full_score.musicxml
    validation_report.json
    model_trace.json
    takes_manifest.json
    music_metrics.json
    package.zip
```

The quality gates check validation errors, required tracks, non-empty MIDI,
MusicXML export, model trace requirements, artifact final states, no pending
takes in export, and package completeness.

## Demo and Packaging

Generate the built-in jazz demo and validate the export package:

```bash
make package-smoke
```

Separate commands:

```bash
make demo-jazz
make export-demo
make validate-demo
make zip-demo
make demo-check
```

The golden baseline metrics runner is:

```bash
make golden-baseline
```

## Validation Philosophy

The internal object model is `ArrangementProject`. Exports are derived from that
single source of truth. Validation covers:

- bar duration and voice timing consistency
- instrument range and transposition
- harmonic support and voice leading
- breathing/rest checks for wind instruments
- piano voicing checks
- drum pattern sanity
- export manifest/package integrity
- release-gate checks for takes and model traces

See `docs/06_VALIDATION_AND_QA.md`, `docs/08_EXPORT_SPEC.md`, and
`docs/nextimplementation/27_PROFESSIONAL_GENERATION_BENCHMARK.md`.

## Data and License Safety

This repository does not version private datasets, generated outputs, model
weights, checkpoints, caches, or local secrets. Keep these paths local:

```text
.env
configs/ai_models.local.yaml
configs/local_model_runtime.yaml
configs/model_registry.yaml
models/
outputs/
data/raw/
data/private/
data/processed/
midi_databases/
*.mid
*.midi
*.musicxml
*.mxl
*.pt
*.pth
*.bin
*.safetensors
*.ckpt
*.pkl
```

Review `docs/nextimplementation/LOCAL_MODEL_SECURITY_AND_LICENSE.md` before
adding new datasets or model checkpoints.

## Documentation Map

Core docs:

- `docs/00_MASTER_RECIPE.md`
- `docs/01_CODEX_WORKFLOW.md`
- `docs/02_ARCHITECTURE.md`
- `docs/03_MUSICAL_SPEC_JAZZ.md`
- `docs/04_OBJECTIVES.md`
- `docs/05_DATASET_LEARNING_MODE.md`
- `docs/06_VALIDATION_AND_QA.md`
- `docs/07_API_SPEC.md`
- `docs/08_EXPORT_SPEC.md`
- `docs/09_AI_EXTENSION_CONTRACT.md`
- `docs/10_PACKAGING_AND_SMOKE.md`
- `docs/11_FUTURE_AI_TRAINING.md`

Local-model implementation docs:

- `docs/nextimplementation/20_27_EXECUTION_MASTERPLAN.md`
- `docs/nextimplementation/20_LOCAL_MODEL_RUNTIME.md`
- `docs/nextimplementation/21_MIDIGPT_LOCAL.md`
- `docs/nextimplementation/22_TEXT2MIDI_LOCAL.md`
- `docs/nextimplementation/23_LOCAL_LLM_PLANNER.md`
- `docs/nextimplementation/24_MIDITOK_TRAINING_STACK.md`
- `docs/nextimplementation/25_CUSTOM_ROLE_MODEL_BOOTSTRAP.md`
- `docs/nextimplementation/26_LOCAL_MODEL_SMOKE_TESTS.md`
- `docs/nextimplementation/27_PROFESSIONAL_GENERATION_BENCHMARK.md`

## Development Notes

- Prefer deterministic, seed-controlled generation for reproducibility.
- Keep model-backed generation behind the artifact importer and take manager.
- Keep generated files out of Git unless they are small examples or docs.
- Run focused tests before broad test suites when changing model adapters or
  export behavior.
- Use `make professional-benchmark` after changes that affect generation,
  validation, takes, model traces, or export packaging.
