# MIDI-GPT backend real API notes

The current backend is intentionally defensive. In PR-21, replace generic method probing with the real API flow:

```python
from midigpt import Score
from midigpt.inference import InferenceEngine, GenerationRequest, InferenceConfig, TrackPrompt

engine = InferenceEngine.from_pretrained(model_name)
score = Score.from_midi(str(context_midi_path))
track_index = resolve_track_index(score, request.track_id)
bar_indices = [bar - 1 for bar in request.bars]
generation_request = GenerationRequest(
    tracks=[TrackPrompt(id=track_index, bars=bar_indices)],
    config=InferenceConfig(
        temperature=request.temperature,
        top_p=0.95,
        model_dim=8,
        mask_mode="attention",
    ),
)
result = engine.session(score, generation_request).run()
result.to_midi(str(output_path))
```

Do not bypass artifact quarantine.
