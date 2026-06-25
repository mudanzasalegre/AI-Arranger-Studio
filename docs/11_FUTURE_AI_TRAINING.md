# Future AI training contract

Objective 14 does not train a neural model. It defines the interfaces and file
contracts needed to train or plug one in later without changing API or web code.

## Runtime contract

The core model interfaces live in `arranger_core.model_contract` and are
exported from `arranger_core`:

- `ModelBackend`: backend protocol with `name`, `version` and `generate(request)`.
- `ModelRequest`: symbolic request containing role, style, meter, tempo, form,
  seed, chord context, previous tokens and controls.
- `ModelResponse`: symbolic response containing target tokens and backend
  metadata.
- `RoleModelGenerator`: role generator protocol carrying a `ModelBackend`.
- `AIWalkingBassGenerator`: first swappable role generator.

`RuleBasedArranger` still defaults to `WalkingBassGenerator`, but accepts
`bass_generator=AIWalkingBassGenerator(...)`.

## Dataset guardrails

Training examples are built from `dataset_tools.PatternIndex` with:

```python
from dataset_tools import build_training_examples

summary = build_training_examples(pattern_index, "data/training/v0", seed=140)
```

Only patterns that satisfy all conditions are exported:

- `usable_for_training=true`
- quality is at least the requested minimum
- license is present and not blocked as unknown/proprietary/all-rights-reserved

The output directory contains:

- `training_examples.jsonl`
- `dataset_splits.json`
- `feature_store.json`
- `training_summary.json`

Splits are deterministic train/val/test assignments derived from stable hashes
and a seed.

## Tokenization placeholder

`PatternTokenizer` is intentionally simple and symbolic. It can encode extracted
dataset patterns and `ArrangementProject` note streams by role. The token format
is not final; it exists so future MidiTok or custom tokenizers can be swapped in
behind the same training-example shape.

## Similarity and memorization

Use `evaluate_memorization(candidates, references, threshold=...)` before
accepting generated model outputs. The current implementation is a conservative
token Jaccard check. Future versions can replace it while preserving the report
shape.

## Placeholder adapters

`midi_models` exposes empty adapters:

- `MidiTokBackendAdapter`
- `ExternalModelBackendAdapter`

They raise `NotImplementedError` until a real tokenizer/model endpoint is
configured.

## Local symbolic training command

The current trainable backend is `midi_models.SymbolicPatternModelBackend`.
To train it on the approved local jazz folders only:

```powershell
python scripts/train_symbolic_model.py
```

This command is hard-restricted to:

- `midi_databases/JAZZVAR_DATASET`
- `midi_databases/RELEASE2.0_mid_unquant`

It writes the compact model to:

```text
outputs/models/jazzvar_release2_symbolic/model/symbolic_pattern_model.json
```

The Make target is:

```bash
make train-symbolic-model
```
