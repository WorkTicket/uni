# Safe Branch Integration Plan — Universe Engine Sprint 14

**DESIGN ONLY / NO MERGES.** No deletion, force pushes, history rewriting, hard reset, or branch replacement.  
**Audited SHA:** `7f4f09b86a44bafd5f92bfe84ec04f2385e7ff1d` (`feature/project-genesis-research`)  
**Audit branch:** `audit/project-genesis-evidence` @ `7f4f09b`  
**Worktree:** `C:\Users\Slay3r\Downloads\uni-audit`

## Existing refs (from `git for-each-ref`, captured in `audit_runs/branch_inventory.json`)

| Ref | SHA | Merge-base with research | Unique modules | Potential conflicts |
|---|---|---|---|---|
| `refs/heads/feature/project-genesis-research` (also `origin/...`) | `7f4f09b` | self | Sprint 13 memory + stats + checkpoint + organism API | audit base; do not alter |
| `refs/heads/audit/project-genesis-evidence` | `7f4f09b` | `7f4f09b` | audit tools + reports only | none |
| `refs/heads/feature/fundamental-cosmology` | `93713c4` | older than research | WebGL2 chem-field shader fixes | UI shader namespace |
| `refs/heads/feature/multiverse-overhaul` | `dcb0e95` | older than research | black-hole engineering + package exports | package export collisions |
| `refs/heads/feature/unified-planetary-life` | `f8a2a89` | older than research | unified planetary life + read-only observatory API | habitat API overlap |
| `refs/heads/feature/release-b-calibrated-cosmology` | `80b5d5e` | older than research | observatory repair docs | docs only |
| `refs/heads/main` / `origin/main` | `021a093` | older than research | initial foundation | stale; not evidence |
| `refs/heads/observatory-repair-pass1` | `dc2fd34` | older than research | Release A PM3D foundation | superseded by research |

## Safeguards

- No deletion, force pushes, history rewriting, hard reset, or branch replacement.
- Preserve the local working tree with `git status` and worktrees. The primary checkout remains dirty (`ui.js`, `test_quantum.py` modified; untracked debug/node_modules files) — these were NOT touched, staged, or discarded.
- Preserve branch snapshots with unique tags or the SHA manifest in `audit_runs/branch_inventory.json`.
- Create a dedicated future integration branch only after user approval of this audit and plan.
- Resolve dependency and semantic API conflicts before any actual merge.
- Re-run all tests and essential conservation/cosmology/persistence benchmarks after each proposed change.

## Sequence after user approval

1. Branch `fix/trainable-memory-baseline` from `7f4f09b` (isolated feature branch; preserves audited baseline).
2. Implement the trainable recurrent baseline fix (Stage 5) and RNG-state continuation (Stage 7) on that branch; run tests.
3. User reviews results; approves merge into `feature/project-genesis-research`.
4. Create integration branch from `7f4f09b`, merge approved feature branch, run full pytest + key benchmarks, then fast-forward `feature/project-genesis-research` (no force-push; normal push only).
5. Never merge the `audit/project-genesis-evidence` branch into research (it holds audit-only tooling/reports).

## Rollback design

- The integration branch can be removed only after audit backup verification; never delete original refs (`7f4f09b` and all branch tips preserved).
- All pre-merge SHAs recorded here; rollback = reset integration branch to `7f4f09b` (never force-push shared refs).
