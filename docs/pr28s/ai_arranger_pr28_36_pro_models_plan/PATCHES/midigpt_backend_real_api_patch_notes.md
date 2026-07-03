# Patch notes — MIDI-GPT backend real API

Apply in `packages/model_backends/model_backends/symbolic/midigpt_backend.py`.

Replace `_run_engine` with logic equivalent to:

```python
from midigpt import Score
from midigpt.inference import GenerationRequest, InferenceConfig, TrackPrompt

score = Score.from_midi(context_midi_path)
target_idx = self._resolve_track_index(score, request.track_id, request.project)
target_bars = [bar - 1 for bar in request.bars]

prompts = []
for index, _track in enumerate(score.tracks):
    if index == target_idx:
        prompts.append(
            TrackPrompt(
                id=index,
                bars=target_bars,
                autoregressive=request.task == "generate_track",
                attributes=self._attributes_for_request(request),
            )
        )
    else:
        prompts.append(TrackPrompt(id=index, bars=[], ignore=True))

generation_request = GenerationRequest(
    tracks=prompts,
    config=InferenceConfig(
        model_dim=self._model_dim_for_bars(target_bars),
        temperature=request.temperature,
        top_p=request.metadata.get("top_p", 0.95),
        mask_mode=request.metadata.get("mask_mode", "attention"),
        polyphony_hard_limit=self._polyphony_limit(request),
    )
)

result = engine.session(score, generation_request).run()
result.to_midi(str(output_path))
return output_path
```

Do not leave the existing generic `engine.generate(...)` fallback as the primary path.
