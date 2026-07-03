# AI Arranger Studio model worker

Optional worker for heavy local models.

The API can run with direct backends during development, but torch/MIDI-GPT/Text2MIDI should eventually run here to avoid loading heavy dependencies inside `apps/api`.

## Start

```bash
python -m uvicorn app.main:app --app-dir apps/model_worker --reload --port 8010
```

## Endpoints

```text
GET /health
GET /v1/models/status
```

Generation endpoints should be added only after PR-21/PR-22 are stable.
