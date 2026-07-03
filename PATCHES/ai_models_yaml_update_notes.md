# Notes for updating configs/ai_models.yaml

Keep `configs/ai_models.yaml` safe and mock-first.

Create `configs/ai_models.local.yaml` from `configs/ai_models.local.example.yaml` and point `.env` to it:

```env
AI_MODELS_CONFIG=./configs/ai_models.local.yaml
```

Only enable one real backend after its smoke test passes.

Recommended activation order:

```yaml
mock_symbolic.enabled: true
local_llm_planner.enabled: true
midigpt.enabled: true
text2midi.enabled: true
```

Never enable audio backends in this phase.
