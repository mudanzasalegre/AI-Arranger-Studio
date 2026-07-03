# Patch notes — StatisticalCustomRoleBackend

`DummyCustomRoleModelBackend` currently emits fixed token templates. Keep it as fallback, but add:

```text
packages/model_backends/model_backends/custom_role/statistical_backend.py
```

The backend should:

1. load `model.json`;
2. read token n-gram tables per role;
3. condition on `role_intent`, `bars`, `density`, `style`, `chord_context`;
4. produce `.tokens.json`;
5. optionally render a small role MIDI artifact;
6. include `training_manifest.yaml`, `license_report.json`, `metrics.json`.

Do not call this “professional” until its output passes:

```text
ArtifactImporter
ProjectMerger
ValidationGate
QualityGate
```
