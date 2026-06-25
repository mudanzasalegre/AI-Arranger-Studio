# 05 — Artifact Quarantine y sistema de takes

## Problema que resuelve

Los modelos pueden generar material útil o basura. No se puede escribir su resultado directamente sobre el proyecto activo.

## Carpetas

```text
outputs/model_artifacts/
  raw/
  imported/
  rejected/
  validated/
```

## Estados

```text
raw       -> artifact recién generado por backend
imported  -> convertido a representación interna parcial
validated -> candidato que superó validación
rejected  -> candidato rechazado por errores graves
```

## Flujo obligatorio

```text
1. Backend genera artifact.
2. ArtifactStore guarda en raw.
3. ArtifactImporter intenta convertirlo.
4. ProjectMerger fusiona en copia temporal.
5. ValidationGate valida.
6. TakeManager crea pending/accepted/rejected.
7. Solo una take accepted puede convertirse en proyecto activo.
```

## Entidades

### ModelArtifactRecord

```python
class ModelArtifactRecord(BaseModel):
    artifact_id: str
    project_id: str | None
    backend_id: str
    task: str
    raw_path: str
    imported_path: str | None = None
    status: str
    created_at: str
    metadata: dict = {}
```

### ArrangementTake

```python
class ArrangementTake(BaseModel):
    take_id: str
    project_id: str
    parent_take_id: str | None = None
    source: Literal["rule_based", "model", "manual", "retrieval"]
    backend_id: str | None = None
    task: str | None = None
    track_id: str | None = None
    bars: list[int] = []
    instruction: str | None = None
    seed: int | None = None
    status: Literal["pending", "accepted", "rejected"] = "pending"
    validation_report_id: str | None = None
    artifact_ids: list[str] = []
```

## ProjectMerger

Debe fusionar de forma quirúrgica.

Casos:

```text
- target track + bars;
- target section + roles;
- new track;
- full sketch separado;
```

Reglas:

- No tocar pistas bloqueadas.
- No tocar compases fuera del rango pedido.
- No eliminar metadata del proyecto.
- No sobrescribir GrooveMap ni SongPlan salvo endpoint de plan explícito.
- Si el artifact trae tempo/compás distinto, ignorar salvo sketch aislado.

## ValidationGate

Bloquea si:

- compases incompletos;
- notas fuera de rango absoluto;
- pista vacía cuando se esperaba material;
- pista modificada fuera de target;
- errores de transposición;
- MusicXML export falla;
- MIDI corrupto;
- similitud excesiva con patrón sensible;
- violación grave de RoleIntent.

Warnings no bloqueantes:

- registro incómodo;
- exceso de densidad;
- frase de viento larga;
- chord-tone ratio bajo;
- comping demasiado denso;
- bajo con demasiados saltos.

## Acceptance criteria

- Artifact válido con mock genera take pending.
- Artifact inválido va a rejected.
- Proyecto activo no cambia hasta aceptar.
- Se puede listar takes.
- Se puede aceptar/rechazar take.
- Se puede restaurar take anterior.
