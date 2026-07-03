# PR-25 — Custom Role Model Bootstrap

## Objetivo

Preparar la infraestructura para modelos propios por rol, aunque todavía sean dummy/baseline.

## Modelos previstos

```text
custom_jazz_melody_v001
custom_jazz_walking_bass_v001
custom_jazz_piano_comping_v001
custom_jazz_horn_responses_v001
custom_jazz_drums_v001
```

## Estructura

```text
models/checkpoints/custom/
  melody/jazz_melody_v001/
    model.safetensors
    tokenizer.json
    config.yaml
    training_manifest.yaml
    metrics.json
    license_report.json
  bass/jazz_walking_bass_v001/
  piano_comping/jazz_piano_comping_v001/
  horns/jazz_horn_responses_v001/
  drums/jazz_drums_v001/
```

## Backend

Crear/validar:

```text
packages/model_backends/model_backends/custom_role/
  __init__.py
  base.py
  dummy_backend.py
  loader.py
```

Interfaz mínima:

```python
class CustomRoleModelBackend:
    backend_id: str
    role: str
    capabilities: ModelCapabilities
    def generate(self, request: ModelGenerationRequest) -> ModelGenerationResult: ...
```

## Reglas

- El modelo propio genera solo un rol.
- El rol debe coincidir con `role_intent.role`.
- Si el checkpoint no existe, backend `unavailable`.
- Si el modelo no tiene `training_manifest.yaml`, backend `unavailable`.
- Si el manifest contiene datasets no permitidos, backend no se puede usar en `export_mode=commercial`.

## Acceptance criteria

```text
- Backends custom aparecen en registry como disabled/unavailable hasta tener checkpoint.
- Dummy backend puede crear artifact válido para tests.
- Loader rechaza modelos sin manifest/licencia.
- API puede listar custom models sin cargarlos en memoria.
```
