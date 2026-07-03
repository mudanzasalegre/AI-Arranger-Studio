# AI Arranger Studio — PR-28 a PR-36: instalación local automática y modo profesional

Este paquete se copia sobre la raíz del repo después de haber completado PR-27.

## Objetivo

Cerrar la última milla entre “infraestructura lista” y “generación MIDI profesional usable”:

```text
instalador local reproducible
→ modelos descargados/cacheados en models/
→ backends reales activados en perfil local/pro
→ planner local JSON activo
→ MIDI-GPT funcionando con la API real
→ Text2MIDI funcionando como sketch backend
→ MidiTok generando datasets reales por rol
→ modelos propios bootstrap entrenables
→ generación profesional con quality gate
```

## Regla principal

No se activa nada globalmente en `configs/ai_models.yaml`.

El perfil real se activa mediante:

```text
configs/ai_models.pro.yaml
configs/local_model_runtime.pro.yaml
.env
```

El core seguro sigue siendo:

```text
modelo → artifact raw → importación → fusión controlada → validación → take pendiente → aceptación → export
```

## Orden estricto

```text
PR-28 — Repo health, environment lock y baseline ejecutable
PR-29 — Instalador automático de modelos locales
PR-30 — MIDI-GPT real adapter + infill profesional
PR-31 — Text2MIDI real sketch backend + wrapper robusto
PR-32 — Local LLM Planner activo por defecto en perfil pro
PR-33 — MidiTok training pipeline hardening
PR-34 — Modelos propios por rol: bootstrap entrenable y no dummy
PR-35 — Professional generation orchestrator
PR-36 — Pro quality gate, benchmark y release candidate
```

## Primer comando tras copiar

```bash
python scripts/models_pro/pro_readiness_audit.py
```

Después sigue:

```bash
python scripts/models_pro/install_all_local_models.py --profile pro --planner-model qwen3:8b
python scripts/models_pro/activate_pro_profile.py --write-env
python scripts/models_pro/verify_all_models.py
python scripts/models_pro/pro_end_to_end_smoke.py
```

## Qué se versiona

Sí:

```text
docs/
configs/*.example.yaml
scripts/
patches/
```

No:

```text
models/
outputs/
data/raw/
data/private/
data/processed/
```
