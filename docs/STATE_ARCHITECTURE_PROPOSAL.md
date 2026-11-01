# Proposed Authoritative World State — Universe Engine Sprint 14

**DESIGN ONLY.** Not implemented. No source code changed.  
**Audited SHA:** `7f4f09b86a44bafd5f92bfe84ec04f2385e7ff1d`  
**Audit branch:** `audit/project-genesis-evidence`

## Goals

- Physics state independent of camera and page UI; camera/navigation never advance simulation time.
- Stable IDs for all materialized scientific entities (halo, star, planet, habitat, organism); identity persists independently of camera position.
- No silent promotion of illustrative data into simulated measurements; every quantity carries provenance (`simulated`, `analytic`, `reconstructed`, `illustrative`, `speculative`) and units.
- Versioned checkpoints containing solver state, cache, coupled state, agent parameters, and **RNG state** (Python `random`, `numpy.random.Generator`, and any other stateful library).
- Explicit local clocks per universe; distinct universes need no fictitious global common time.
- Time controls independent of navigation.

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
    kind: str                 # halo | star | planet | habitat | organism | ...
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
    rng_states: dict[str, Any]          # python, numpy, per-subsystem streams
    state_blob_checksums: dict[str, str]
    environment_fingerprint: dict[str, str]  # OS, versions, hardware, threads
```

## Proposed coupling interface

`ModelAdapter.advance(input_state, dt, boundary_conditions) -> (output_state, exchanges, uncertainty, provenance)`

Track model-specific mass/energy/momentum/species ledgers; a model must not duplicate matter inventories when a habitat agent consumes resources. Reconstructed details represented as such; camera movement must not advance physical time. Specify API versioning, memory budgets, and deterministic-replay limits.

## Open design questions (to be answered from live source before implementation)

- What currently owns the authoritative clock?
- Are halos/stars client-side procedural, backend-derived, or both?
- Where are habitat IDs currently created and mutated?
- Which checkpoint/RNG parts are missing (controller RNG stream position confirmed missing)?
- What are the actual units across interfaces?
- Can the viewer render directly from backend snapshots without fabricating state?

## Evidence required before implementation

Document live source and exact branch/SHA, all existing state formats, and a migration path that preserves old checkpoints. Do not merge major architectures until the audit and design are reviewed.
