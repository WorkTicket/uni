# T00 audit — original `big-bang-universe` project (2026-10-30)

Audited independently and **not modified or merged**. It remains a separate repository.

## Inventory (from commands, not memory)
- 48 files excluding caches; 853 lines of Python in `src/universe/` (18 modules); 23 documents (`docs/00`–`22`).
- Tests: **36 collected, 36 passed** (`python -m pytest -q tests`, 2.5 s) in its own environment (separate from this package's venv).
- Tests per file: cosmology 4, design decisions 14, rulebook 10, stars/planets 8.
- Optional dependencies for its data layer: `periodictable`, `radioactivedecay`, `pyscf` (tests skip PySCF cases if absent).

## Module map against this package
| Original module | What it does | Package counterpart | Notes |
|---|---|---|---|
| `cosmology.py`, `timeline.py` | Friedmann clock from radiation era; epoch table | `physics.py` | Overlap. **Cross-check: original age at a=0.02 is 0.0478 Gyr; package elapsed 13.7429 Gyr; sum 13.7907 vs original total 13.7909 (0.16 Myr, 1.2e-5 relative).** Two independent implementations agree. |
| `stars.py` | Kroupa IMF sampling, piecewise mass-luminosity, lifetimes, fates, approximate yields | `physical_models.stellar_track` | Original is richer (package uses a single L~M^3.5 for all masses, no fates). |
| `planets.py` | habitable zone, snow line, retention, habitability checks | catalog sampling in `evolution.py` | Partial overlap; original has explicit rules. |
| `nuclear_data.py`, `nuclear.py` | real atomic masses/Q-values/half-lives; semi-empirical formula as fallback | none | Original measured the formula's RMS error (~7.2 MeV vs 2,711 nuclides; gets triple-alpha Q wrong by 3x). |
| `chemistry.py` | real PySCF HF/DFT calculations | `prebiotic_network` (toy kinetics) | Different purpose. |
| `quantum.py`, `qsim.py` | periodic table from quantum numbers; exact small state-vector simulator | none | |
| `adapters.py` | config with hashing, validation, ensemble sampling of unknowns | config sha256 only | Strong candidate for R15. |
| `provenance.py`, `fidelity.py`, `rules.py`, `relativity.py`, `constants.py`, `rng.py` | provenance, method chooser, rule registry, black-hole basics, CODATA, seeded RNG | none | R15 / R9 candidates. |

## Not compared
I did not read `MASTER_BLUEPRINT.md` or the original's `docs/` line by line against each other; the module map above is from code and test inspection only.

## Smallest low-risk integration (proposed, not performed)
1. Keep both repos separate. No file overwritten in either.
2. Port **by phase, with their tests**, as new modules: R8 `stars.py` + `nuclear_data.py` (extras `[data]`); R10/R11 `planets.py`; R15 `adapters.py`, `provenance.py`, `fidelity.py`; R14 `qsim.py`.
3. Avoid the package-name collision (`universe` vs `universe_engine`) by importing under `universe_engine.<subpackage>`; no top-level `universe` package in the starter.
4. Port the cosmology cross-check as a permanent regression test when the original is vendored.
5. Carry over the original's "adapter for every unknown" convention (`docs/20_design_decisions.md` in that repo).
