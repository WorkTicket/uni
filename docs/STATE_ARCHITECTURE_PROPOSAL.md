# Universe Engine — State Architecture Proposal

**Status:** DESIGN ONLY. Not implemented. This is the Stage 2 contract; do not implement during Stage 0 audit.  
**Audited base:** `ef10c4c` (branch `audit/project-genesis-evidence`, based on `feature/project-genesis-research` `7f4f09b`)  
**Machine/versions:** Windows 11, Python 3.14.5

This proposal covers persistent identities, units, clocks, provenance, cross-scale coupling, and deterministic checkpoints. It is informed by the existing `world_state.py` (PASS 4 world-state spine) which is an adapter/registry layer and does not replace existing models.

## 1. Goals

- Scientific state exists independently of rendering. The viewer asks for state and never invents it.
- Persistent, deterministic save / close / reload / resume that matches an uninterrupted run (hash-verified).
- Every value carries provenance (Simulated / Analytic / Reconstructed / Illustrative / Speculative) and explicit units.
- Stable object IDs across the chain universe → halo → stellar system → planet → habitat → organism.
- Independent simulation clocks per universe and per local process; camera/zoom never move a clock.

## 2. Versioned world-state schema

```
world-state/v2
{
  "schema": "world-state/v2",
  "code_sha": "<git sha of build that produced it>",
  "model_version": "0.2.0",
  "universe_id": "<stable uuid>",
  "created_at": "<ISO-8601>",
  "universe": { ... },          // UniverseEntity
  "entities": { "<id>": WorldEntity, ... },
  "clocks": { "<universe_id>": WorldClock, ... },
  "provenance_ledger": [ ... ], // promotions/demotions of reconstructed detail
  "rng_states": { "<rng_id>": {bit_generator, state}, ... },
  "state_hash": "<sha256 of canonical serialization>",
  "checks": { "mass_drift": ..., "momentum_drift": ..., "energy_drift": ... }
}
```

### WorldEntity
```
{
  "id": "<stable id>",
  "parent_id": "<id | null>",
  "universe_id": "<id>",
  "kind": "universe|halo|stellar_system|star|planet|habitat|organism",
  "fields": { "<name>": Quantity, ... },
  "model_version": "<str>",
  "updated_at_local_time": <float seconds>,
  "provenance": "<Simulated|Analytic|Reconstructed|Illustrative|Speculative>",
  "generator_version": "<str | null>"   // for reconstructed detail
}
```

### Quantity
```
{
  "value": <float>,
  "unit": "<explicit unit string, e.g. 'm', 'kg', 's', 'Gyr', 'AU', 'K'>,
  "provenance": "<tier>",
  "model_id": "<str>",
  "uncertainty": <float | null>
}
```

Units are explicit everywhere. No unitless scientific quantity. Comoving vs proper is a field, not implicit.

### WorldClock (one per universe, independent)
```
{
  "universe_id": "<id>",
  "integration_time_s": <float>,
  "model_epoch": "<str>",
  "scale_factor": <float | null>,
  "paused": <bool>,
  "rate": <float>              // sim seconds per wall second (presentation, not physics)
}
```
- Each universe owns its clock. Distinct universes need no common time.
- Camera movement, zooming, and mode changes never modify `integration_time_s`, `scale_factor`, or `rate`.
- `paused: true` freezes all solvers for that universe.

## 3. Stable identities

IDs are assigned at object creation and never regenerated:
- `universe-<uuid>`, `halo-<index>-<universe>`, `star-<id>`, `planet-<id>`, `habitat-<id>`, `organism-<id>`.
- Position is a pure function of `(object ID, simulation time, parent state at t)` — never of camera position, zoom, or frame count (fixes Bug C).
- A single physical object is never regenerated with a different history.

## 4. Provenance tiers and the reconstruction service

Detail below simulated resolution is a deterministic pure function:
`reconstruct(object_id, parent_state_at_t, level, region, generator_version) → Reconstructed | Illustrative`

- Cache keyed by `(object ID, parent state hash, level, region)`.
- Every reconstructed/illustrative value carries `generator_version` and `provenance`.
- Reconstructed detail is promoted into persistent state only when selected or needed; promotion is recorded in the `provenance_ledger` (what was promoted, when, by whom, generator version).
- The viewer renders a tier mix; reconstructed detail announces itself via toast and fidelity strip (Stage 5).
- Where the simulation does not produce a level, the viewer shows "Not modeled", never an invented substitute.

## 5. Cross-scale coupling and conservation

Couplings are explicit and labeled (numerical / reduced-order / reconstructed / illustrative / speculative):

- Cosmology → Halo: reduced-order (density-peak candidates).
- Halo → Star: illustrative / statistical.
- Star → Planet: illustrative (probabilistic samples).
- Planet → Habitat: numerical (environment → chemistry each step).
- Habitat → Organism: numerical (local patch sensing, action on chemistry).

Conservation tests (Stage 2):
- Reconstructed stellar mass sums to parent stellar mass (stated tolerance).
- Reconstructed biomass matches habitat cell (stated tolerance).
- Mass/momentum/energy drift tracked in `checks`; drift above threshold halts the run (Mission Control / Bug banner).

## 6. Deterministic checkpoints

- Checkpoint captured only at step boundaries (pause/save/load bitwise on step boundary, as in `session.py`).
- Canonical serialization before hashing: deterministic key order, float normalization, arrays as contiguous binary.
- `state_hash = sha256(canonical serialization)`.
- Restore verifies `state_hash` match; on mismatch reports named mismatching fields.
- All RNG states captured: Python `random`, NumPy `bit_generator.state`, any other library RNG.
- Thread counts fixed and recorded; results scoped to OS/library versions/hardware.

## 7. Save / load / resume contract

- `save(id)` → checkpoint file + `state_hash`.
- `load(id)` → validate schema, code_sha compatibility, state_hash; restore entities, clocks, RNG states; rebuild habitat from checkpoint; reconnect organism by stable ID.
- Resume continues from restored `integration_time_s`.
- Acceptance test: run A (uninterrupted) vs run B (save, reload, continue) over many steps; full state compared by canonical serialization and hash; any mismatching field named. Results hold only for the machine/library versions tested.

## 8. Relationship to existing code

- `world_state.py` (`WorldState`, `WorldEntity`, `WorldClock`, `Quantity`, `Provenance`, `connect_chain`) is the partial existing implementation of this proposal (PASS 4). It is an adapter/registry; it does not replace `Planet`, `Habitat`, `PersistentOrganism`.
- This proposal extends it to: full chain including halo/stellar system, independent per-universe clocks, `state_hash`, RNG state capture, provenance ledger, reconstruction service, and the save/resume acceptance contract.

## 9. What this is NOT

- Not a global quantum wavefunction (by design).
- Not a consciousness model; functional measures only.
- Not end-to-end numerical cosmology→organism (several links are illustrative/reduced-order).
- Not implemented yet — design only for Stage 2.
