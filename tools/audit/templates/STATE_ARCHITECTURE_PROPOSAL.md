# Proposed Authoritative World State — DESIGN ONLY

## Goals

- Physics state independent of camera and page UI.
- Stable IDs for all materialized scientific entities.
- No silent promotion of illustrative data into simulated measurements.
- Versioned checkpoints containing solver, cache, coupled state, agent parameters and RNG state.
- Explicit local clocks; distinct universes need no fictitious global common time.

## Suggested contracts (proposed, not implemented)

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

class Provenance(str, Enum):
    SIMULATED = 'simulated'
    ANALYTIC = 'analytic'
    RECONSTRUCTED = 'reconstructed'
    ILLUSTRATIVE = 'illustrative'
    SPECULATIVE = 'speculative'

@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str
    provenance: Provenance
    model_id: str
    uncertainty: float | None = None

@dataclass
class Entity:
    id: str
    parent_id: str | None
    universe_id: str
    kind: str
    fields: dict[str, Quantity]
    model_version: str
    updated_at_local_time: float

@dataclass
class UniverseClock:
    universe_id: str
    integration_time_s: float
    model_epoch: str
    scale_factor: float | None

@dataclass
class CheckpointManifest:
    schema_version: int
    code_sha: str
    model_versions: dict[str, str]
    simulation_clocks: dict[str, Any]
    rng_states: dict[str, Any]
    state_blob_checksums: dict[str, str]
    environment_fingerprint: dict[str, str]
```

## Proposed coupling interface

`ModelAdapter.advance(input_state, dt, boundary_conditions) -> (output_state, exchanges, uncertainty, provenance)`

Track model-specific mass/energy/momentum/species ledgers and clear independent-clock semantics. A model must not duplicate matter inventories when a habitat agent consumes resources. Reconstructed details are represented as such; camera movement must not advance physical time. Specify API versioning, memory budgets and deterministic replay limits.

## Open design questions

- What currently owns the authoritative clock?
- Are halo/stars client-side procedural, backend-derived, or both?
- Where are habitat IDs currently created and mutated?
- What checkpoint/RNG parts are missing?
- What are actual units across interfaces?
- Can the viewer render directly from backend snapshots?

## Evidence required before implementation

Document live source and exact branch/SHA, all existing state formats, and a migration path that preserves old checkpoints.
