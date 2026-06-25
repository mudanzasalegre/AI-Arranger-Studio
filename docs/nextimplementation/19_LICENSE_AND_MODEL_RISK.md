# 19 — Licencias y riesgo de modelos/datasets

## Regla general

Toda integración de modelo/dataset debe tener manifest de licencia y uso.

Campos mínimos:

```yaml
id: string
kind: model|dataset|code
source: string
license: string
license_confidence: high|medium|low
commercial_use: allowed|forbidden|review_required|unknown
notes: string
```

## MIDI-GPT

Tratar como:

```yaml
commercial_use: review_required
```

Aunque el código pueda tener licencia permisiva, revisar siempre por separado:

- código;
- pesos;
- dataset de entrenamiento;
- model card;
- condiciones de Hugging Face;
- uso comercial.

## Text2MIDI

Tratar como:

```yaml
commercial_use: review_required
```

Aunque el repo indique MIT para el código, revisar pesos/model card/dataset.

## Datasets tipo Real Book

Separar:

```text
- uso privado local;
- research;
- training comercial;
- redistribución;
- modelo derivado.
```

No meter en training comercial si:

- procedencia desconocida;
- transcripción de standard con copyright;
- chart de iReal/Real Book sin permiso;
- scraping sin licencia clara.

## Export comercial

Si `EXPORT_MODE=commercial`, bloquear:

- backends marcados `non_commercial`;
- datasets `research_only`;
- artifacts con source license `unknown`;
- patterns con similarity demasiado alta a fuente protegida.

## Acceptance criteria

- `configs/ai_models.yaml` contiene campo `commercial_use`.
- Dataset manifests contienen licencia.
- Release gate revisa compatibilidad.
- `model_trace.json` permite auditoría.
