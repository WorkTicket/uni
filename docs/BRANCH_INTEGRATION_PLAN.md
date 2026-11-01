# Universe Engine — Branch Integration Plan

**Audited base:** `feature/project-genesis-research` = `7f4f09b` (Sprint 13)  
**Audit branch:** `audit/project-genesis-evidence` = `ef10c4c`  
**Machine/versions:** Windows 11, Python 3.14.5  
**Rule:** DO NOT merge any branch during this audit. This plan is design-only; merges require explicit user approval after audit review.

## 1. Branch inventory and ancestry

From `git branch -avv` and `git log --all` (captured in `audit_runs/branch_inventory.json`):

| Branch | Commit | Description | Relationship to research head `7f4f09b` |
|---|---|---|---|
| `feature/project-genesis-research` | `7f4f09b` | Sprint 13: validated memory, paired statistics, checkpoint/replay, observatory organism | **research head (integration target)** |
| `audit/project-genesis-evidence` | `ef10c4c` | Audit: Sprint 14 scientific audit reports and tooling | Branched from `7f4f09b` |
| `fix/trainable-memory-baseline` | `1f49241` | PASS 5: embodied learning harness (research, experimental) | Descendant line with PASS 1-5 work |
| `main` | `021a093` | Initial commit: verified universe engine foundation | NOT research-authoritative; ancestor |
| `observatory-repair-pass1` | `dc2fd34` | Release A foundation: comoving 3D PM gravity, coordinate transforms, Scientific Simulation mode | Parallel release line |
| `feature/fundamental-cosmology` | `93713c4` | Fix WebGL2 chem-field shader constructor errors | Parallel |
| `feature/multiverse-overhaul` | `dcb0e95` | Extend civilizations with black-hole engineering | Parallel (has stash `009392a`) |
| `feature/release-b-calibrated-cosmology` | `80b5d5e` | docs: record observatory repair status and test counts | Parallel |
| `feature/unified-planetary-life` | `f8a2a89` | Add unified planetary life simulation with read-only observatory API | Parallel |

Ancestry verified: `17839a1` (Sprint 11) and `cb82d86`→`05b64f8` (Sprint 12) are ancestors of `7f4f09b`. `05b64f8 IS an ancestor of 7f4f09b` (merge-base check passed).

`main` (`021a093`) is NOT authoritative for the research program; it is the original foundation commit. Do not treat `main` as the integration base.

## 2. Backup status (pre-integration safety)

Non-moving backup tags exist for every branch tip (created 2026-11-01, before any integration work):

- `audit-backup-main-021a093`
- `audit-backup-feature-project-genesis-research-7f4f09b`
- `audit-backup-fix-trainable-memory-baseline-1f49241`
- `audit-backup-observatory-repair-pass1-dc2fd34`
- `audit-backup-feature-fundamental-cosmology-93713c4`
- `audit-backup-feature-multiverse-overhaul-dcb0e95`
- `audit-backup-feature-release-b-calibrated-cosmology-80b5d5e`
- `audit-backup-feature-unified-planetary-life-f8a2a89`
- `audit-backup-audit-project-genesis-evidence-ef10c4c`

No force-push, no rebase, no branch deletion. A stash `refs/stash` (`009392a`, "WIP on feature/multiverse-overhaul") exists on `feature/multiverse-overhaul` and must be preserved.

## 3. Proposed integration order (dependency-ready, one at a time)

Per the master plan, stages build on the Stage 2 state contract. Proposed merge order into `feature/project-genesis-research` (research head), each preceded by a dedicated integration branch and full test run:

1. **Stage 2 authoritative state** (new work on `audit/project-genesis-evidence` → integration branch) — stable IDs, clocks, provenance, checkpoint hash. This is the contract all later stages depend on.
2. **Stage 1 viewer fixes** (Bugs A/B/C) — viewer commits; depends on nothing, but Bug C is the first piece of the state contract.
3. **Stage 3-7 UI/state/features** — each on its own branch, merged only after its stage gate passes.

Cross-branch merges (e.g. `feature/unified-planetary-life`, `observatory-repair-pass1`) are OUT OF SCOPE for this audit. They are parallel feature lines with potential conflicts (PM3D, observatory, multiverse). Do not merge until the user reviews the audit and explicitly approves.

## 4. Conflicts and risks

- `observatory-repair-pass1` (`dc2fd34`) and `feature/fundamental-cosmology` (`93713c4`, chem-field shader) both touch the observatory viewer. Merging them into the research line risks re-introducing the WebGL2 shader constructor errors or conflicting with Stage 1 Bug A (black box) fixes.
- `feature/multiverse-overhaul` has an uncommitted stash (`009392a`) that must be resolved before any merge of that branch.
- `main` is not an ancestor of the research head in a meaningful way for research work; merging `main` into research would regress Sprint 1-13 work.

## 5. Rollback plan

Every merge is performed on a fresh integration branch from `7f4f09b` (or the current research head). If a stage gate fails:
- The integration branch is abandoned (not deleted, kept for inspection).
- `feature/project-genesis-research` is never moved except by a fast-forward of a fully-gated integration branch.
- Backup tags allow restoring any branch tip to its pre-audit state.

## 6. Replay / verification tests after each merge

After each merge into the research line, run:
- `python -m pytest -q -ra` (full suite; baseline 527 pass / 2 skip / 0 fail / 0 hang at `ef10c4c`)
- `universe benchmark hydro|gravity|nbody`
- Stage 2 acceptance test: save → close → reload → resume == uninterrupted run (hash, named mismatches)

Record exact command, exit code, pass/fail/skip/hang counts, git SHA, machine and library versions in `docs/IMPLEMENTATION_STATUS.json` regenerated from run output (never hand-edited).

## 7. Gate to proceed to integration

The user must review this audit report and explicitly approve implementation before any branch is merged. This document is design-only.
