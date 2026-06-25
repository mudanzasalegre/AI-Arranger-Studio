# Packaging and smoke tests

Objective 13 adds a repeatable demo package flow for a new developer or evaluator.
The canonical package is generated from the `jazz_hard_bop_minor_blues_sextet`
preset and writes to `outputs/obj13_demo_jazz`.

## Install

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
npm --prefix apps/web install
```

MuseScore CLI is optional, but when it is installed the smoke test also creates
`full_score.pdf` and PDF parts.

## Fast verification

```powershell
python -m ruff check apps packages scripts tests
python -m pytest -q
npm --prefix apps/web run lint
```

With GNU Make:

```bash
make lint
make test
```

## Demo package flow

```bash
make demo-jazz
make validate-demo
make zip-demo
```

Or run the whole packaging smoke test:

```bash
make package-smoke
```

Equivalent PowerShell commands:

```powershell
python scripts/demo_jazz.py
python scripts/validate_demo.py
python scripts/zip_demo.py
python scripts/package_smoke.py
```

Expected outputs:

```text
outputs/obj13_demo_jazz/
  arrangement_project.json
  generation_spec.json
  full_arrangement.mid
  midi_tracks/
  full_score.musicxml
  validation_report.json
  validation_report.html
  export_manifest.json
  demo_jazz_summary.json
  validate_demo_summary.json
  zip_demo_summary.json
  package_smoke_summary.json
outputs/obj13_demo_jazz.zip
```

On a machine with Python, Node dependencies and MuseScore already installed,
`make package-smoke` is intended to complete well under 10 minutes.
