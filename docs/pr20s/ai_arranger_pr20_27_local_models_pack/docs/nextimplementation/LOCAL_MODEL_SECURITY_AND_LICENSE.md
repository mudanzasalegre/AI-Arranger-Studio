# Seguridad, privacidad y licencia de modelos locales

## Local no significa riesgo cero

Ejecutar localmente reduce exposición de prompts, proyectos y datos privados, pero no elimina:

- Riesgo de licencia de pesos.
- Riesgo de dependencias pesadas o abandonadas.
- Riesgo de supply chain al clonar repos externos.
- Riesgo de reproducibilidad por versiones.
- Riesgo de memorizar datasets privados.

## Regla de licencias

Cada modelo debe tener entrada en `configs/model_registry.yaml` con:

```yaml
license: review_required
commercial_use: true|false|review_required|depends_on_training_manifest
source: huggingface|github|local_training|ollama
```

## Datasets

No entrenar modelos comerciales con:

```text
- archivos sin licencia;
- Real Book MIDI de webs random;
- outputs de modelos externos sin permiso;
- datos `research_only`;
- datos `non_commercial` si el resultado se monetiza.
```

Sí permitido para uso local privado:

```text
- private_user_library: true
- not_redistributable: true
- local_learning_only: true
```

## Anti-memorization

Antes de exportar modelos propios:

- Comparar similitud contra training set.
- Bloquear resultados demasiado similares.
- Guardar `license_report.json` y `training_manifest.yaml`.

## Ollama

Mantener Ollama escuchando en localhost. No exponer `11434` a internet.

## Hugging Face

Configurar:

```env
HF_HOME=./models/hf_cache
HF_HUB_CACHE=./models/hf_cache/hub
HF_HUB_DISABLE_TELEMETRY=1
```

Si el entorno debe trabajar offline después de descargar:

```env
HF_HUB_OFFLINE=1
```
