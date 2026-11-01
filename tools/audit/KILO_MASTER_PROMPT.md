# KILO — UNIVERSE ENGINE FULL SCIENTIFIC / ARCHITECTURE AUDIT

Repository: https://github.com/WorkTicket/uni
Priority branch: `feature/project-genesis-research`
Sprint 11: `17839a158d6980fca301630ecece6593f6969ec2`
Sprint 12: `cb82d86644df49249ebf8b5783a789377797ddb0`

## MISSION AND HARD RULES

Complete an **evidence-backed, read-only audit** of the *actual* repository and its branches. Do not write or change scientific implementation source, fix model defects, change pass thresholds, merge, rebase, cherry-pick or force-push. You may add audit-only scripts and reports to a dedicated audit branch. Do not represent a test as executed unless you actually executed it and saved its output. Never silently turn a failing result into a successful one. Failed experiments are not necessarily software bugs.

Long-term Universe Engine goal: one persistent scientific platform linking cosmology, halos, galaxies, stars, planets, climate, chemistry, evolution, embodied minds, quantum experiments, and separately labeled speculative multiverse/brane hypotheses. The immediate work is to determine what is genuinely working; **do not implement new features during this audit**.

Do not use stale public `main` as evidence of the newest research branch. Back up all relevant refs before work; never force-push, delete branches, reset or overwrite uncommitted changes. If checkout is dirty, create a separate worktree instead of altering it. Never embed credentials in the reports or logs.

## APPROACH: FIVE AUDIT STAGES, NO SKIPPING

### Stage 0 — Safety + repository inventory

1. At repo root run `git status --porcelain=v1`, `git remote -v`, `git branch -avv`, `git log --all --decorate --oneline -n 40`, `git rev-parse HEAD`, `git fetch --all --prune` (fetch does not modify checked-out source; do not prune local branches), record stdout and errors.
2. Tag existing local branch tips with nonmoving unique audit snapshot tags, without modifying branches; or record SHA hashes if tags are prohibited. Do not overwrite existing tags. Record remote-tracking branches separately.
3. Create a dedicated `audit/project-genesis-evidence` branch from the CURRENT validated research commit using a separate Git worktree, **not from `main`**. If worktree is unsupported, stop and state the blocker; never risk user changes.
4. Copy this pack into audit branch `tools/audit/` or run from external path. Run `scripts/branch_inventory.py` to produce `audit_runs/branch_inventory.json`. Audit output under `audit_runs` should normally be gitignored; reports in `docs/` are committed.
5. Run `scripts/module_inventory.py` against the exact checked-out commit to produce `audit_runs/module_inventory.json`. Classify implementation only after reading relevant source and checking data flow. Files existing do not imply functionality.
6. Stop and create an initial staged audit summary with exact branch hashes. Commit report-only changes.

### Stage 1 — Deterministic test execution and environment capture

1. Record `python --version`, `python -m pip freeze`, OS, CPU, Python path, `git rev-parse HEAD`, and timestamp. For JS capture Node version, npm version, lockfile hash. Redact secrets.
2. Inspect `pyproject.toml`, `pytest.ini`, `requirements*.txt`, `package.json`, Windows build scripts and repo instructions to discover actual supported commands. DO NOT guess test paths.
3. Run `scripts/run_tests.py --timeout 180 -- python -m pytest -q -ra`, with output files. If this times out, first run `python -m pytest --collect-only -q` with timeout and then isolated discovered test modules (60-180 sec each). Record pass/fail/skip/error/timeout for EACH command. A hang is NOT a pass.
4. Run syntax checks and any supplied research reproduction commands in a bounded process. Use independent reference formulas where relevant; do not accept test `all_passed=true` alone.
5. For each reported module: identify test commands; verify tests were executed against this SHA; record evidence log file, conditions and limitations. Do NOT correct scientific code during this stage.
6. Commit intermediate evidence summary and updated `docs/AUDIT_REPORT.md`.

### Stage 2 — Scientific claim audit and statistical preregistration

Read source, tests, status files, and all scientific reports from actual checked-out branch.

**Original World Model Failure:** Preserve every seed's original results, including reactive +0.48/+0.49/+0.42 vs original recurrent -0.77/-1.16/-2.44. Clearly distinguish original from post-bugfix reruns. Fully observable GridWorld being a memory-irrelevant task is H1 *hypothesis*, not a proven sole root cause. Recurrent model failure to imitate a hand-written reactive policy can imply learning, reward optimization, or implementation defects.

**Comparison fairness:** Identify which baseline is hand-coded and which learned; action access, observation access, train/eval split, episode budget, optimizer and compute, reward shaping, trainable parameter count, recurrent state resets, reward/next-observation alignment, RNG state, hidden state mutated during action candidate selection, transfer of learned weights, whether agents actually trained across episodes, and whether the world model uses action-conditioned dynamics. Check that memory tasks' action interface (e.g. `direction` vs `dx`) is consistent. Audit and record each issue; don't fix here.

**Stats:** BEFORE running any new comparison, write `docs/STATISTICAL_ANALYSIS_PLAN.md` from provided template, specifying exactly one primary comparison, endpoints, planned 20+ independent training seeds when feasible, independent held-out environment seeds, matching episodes and access, stopping rule, confidence interval on *paired per-seed differences*, handling of training-seed vs eval-seed hierarchy, effect sizes and a Holm correction for multiple secondary comparisons. Predeclare how to handle timeouts/failed runs, and report exploratory results separately. The 20 seed minimum is a starting point; justify sample size or state limitations. Avoid pseudoreplication: many episodes from one trained agent do not constitute many independent trained agents. A CI for model mean alone is NOT a CI for model-minus-baseline; individual CIs overlapping/including zero do NOT determine pairwise significance. If tests cannot run with resources available, write plan and NOT TESTED — do not quietly reduce quality gates.

Use `scripts/paired_stats.py` on a clean CSV of **paired evaluation outcomes** (same task and evaluation seed for both agents, matched training replicate or matched predefined pairing). If no legitimate pairs exist, report them as unpaired and do not misuse the tool. Preserve raw CSV, manifest, and logs; data changes must be traceable.

**Sprint 12:** Verify `persistent_organism.py`, reported habitat mutation, energy/resource ledger, private/local sensory interface, memory and controller policy, true training versus static controller, checkpoint restore, and `inspector_state()`. Mark the frontend Mind Inspector as NOT IMPLEMENTED if it remains unwired. Do not equate a Python lifecycle demo with a desktop/browser E2E demonstration.

**Quantum:** Verify density matrix/CHSH math, local classical bound <=2, quantum expected 2√2, uncertainty interval, no-signalling marginals and independent numerical reference. Do not equate agreement with toy CHSH with a full quantum universe.

**Multiscale:** Conservation tests and cost claims must document compared schemes and center-of-mass/accuracy errors.

**Multiverse:** Label numerical hypothesis, reduced order, procedural illustration and missing; independently inspect actual equations, state evolution and origin-event rules. Do not interpret a UI link to a laboratory as an integrated physical multiverse.

Record all scientific claims in `docs/SCIENTIFIC_CLAIMS_MATRIX.md` with path, test command, log and evidence labels.

### Stage 3 — Persistence and integration audit

Trace actual object/data flow across cosmology -> halo -> stellar system -> planet -> habitat -> organism. For every edge, record source output, destination input, quantity, physical units, whether coupling is truly dynamic, IDs, model assumptions, update cadence and proven test. Mark each `numerical`, `reduced-order`, `reconstructed`, `illustrative`, `speculative` or `missing`. Distinguish independent seeded catalogs from numerically derived structure. Do not claim a complete causal chain simply because modules exist.

For checkpoint/replay, inspect EVERY mutable state component: stable object IDs, timers, body, chemistry, local patch, controller weights, recurrent hidden vectors, memory, workspace, self-model, evolution, events, Python RNG, NumPy RNG, other libraries, relevant caches and environmental RNG. Compare uninterrupted evolution and checkpoint/restore evolution from same initial state. Freeze seed, relevant threading and environment; canonicalize serializable state before hashing. Use `scripts/checkpoint_compare.py` to diff **pre-generated snapshots** captured from the actual system, and create adapter tests using the actual repo APIs. Do not compare only a few selected fields. Declare exact reproducibility scoped to OS, versions, hardware and thread settings. Report all mismatched fields explicitly. Don't make a universal cross-platform bit-exact claim.

Audit viewer source: is the renderer an authoritative state consumer, or does it create its own scientific facts based on seed/time/camera? Does camera selection change physics time? Does navigating away and back change IDs or habitat? Does FPS/playback move the state without backend integration? Test in actual browser only if browser automation environment exists.

Windows: If NOT on Windows, mark Windows executable, installer, WebGL2 runtime and clean-machine acceptance as **NOT TESTED**; do not claim verified. Create `docs/WINDOWS_ACCEPTANCE_CHECKLIST.md` for user. If Windows is available, run genuine build/install/run tests and record logs, hardware and screenshot evidence.

### Stage 4 — Final evidence and safe integration plan

Generate/finish exactly these required documents:

- `docs/AUDIT_REPORT.md` — evidence-backed module matrix, precise commands, results, environment, blocker, highest risks.
- `docs/SCIENTIFIC_CLAIMS_MATRIX.md` — claim, mathematical validity, empirical validation, evidence path/command/log, and whether on-screen scientific representation is authoritative or illustrative.
- `docs/BRANCH_INTEGRATION_PLAN.md` — exact source/target SHA for each branch, ancestry, conflicts, backups, safe merge ordering, replay tests, rollback plan. DO NOT merge in audit.
- `docs/STATE_ARCHITECTURE_PROPOSAL.md` — interfaces and schema (IDs, units, clocks, provenance, checkpoint RNG, semantic model boundaries, cross-scale adapter and visualization contract). DESIGN ONLY.
- `docs/STATISTICAL_ANALYSIS_PLAN.md` — preregistered experimental plan and status.
- `docs/WINDOWS_ACCEPTANCE_CHECKLIST.md` — actual tested/not tested status.

For EACH module/claim mark: VERIFIED / IMPLEMENTED_UNVERIFIED / PROTOTYPE / ILLUSTRATIVE / STUB / FAILED / NOT_TESTED / NOT_APPLICABLE. `VERIFIED` requires concrete executed acceptance evidence; negative *scientific* result is not automatic software failure.

Repair typos and stale metadata only in **new audit documents** and note defects in original status documents; do not change source or rewrite historical findings during audit. Example known anomalies: `chss.py` vs `chsh.py`, stale `head_commit` recorded during a commit, original per-seed failure details softened/omitted in later status summary, and CI interpretation errors. Version stamps should describe the SHA being audited, not pretend to anticipate the SHA of a future commit. Prefer external audit manifests generated at runtime.

Commit reports and safe audit tools to dedicated audit branch only. Push normally; NEVER force-push. Keep raw logs/CSV available as artifacts and reference them by stable relative paths and SHA256; do not accidentally commit enormous files or secrets. Provide exact remote link and commit SHA, tests run, skipped, failures, timeouts, not-tested items and three recommended next engineering actions.

## HARD STOP RULES

1. Do not modify any code under audit, alter tests to pass, or relax science standards.
2. Do not merge branches until the user reviews the audit and explicitly approves implementation.
3. If an audit stage is blocked, record `BLOCKED` and continue independent audit tasks; never invent evidence.
4. Keep results consistent with underlying evidence; every 'VERIFIED' row needs a command, output file, SHA and independent reference or invariant.
5. Stop after completing docs and audit tools: NO new physics, no UI rewrite, no consciousness model improvements in this pass.
6. Report after each stage, with commit when appropriate; do not defer all findings to a single end summary.

## EXECUTION — COPY EXACTLY ON A SHELL FROM REPO ROOT

PowerShell (Windows):
```powershell
$ErrorActionPreference = 'Stop'
git status --porcelain=v1
git remote -v
git fetch --all --prune
git branch -avv
git log --all --decorate --oneline -n 40
git rev-parse HEAD
python --version
```

Bash (Linux/macOS):
```bash
set -euo pipefail
git status --porcelain=v1
git remote -v
git fetch --all --prune
git branch -avv
git log --all --decorate --oneline -n 40
git rev-parse HEAD
python --version
```

Then create worktree, e.g. from repo root (**only if** path does not exist and target branch is correct):
```bash
git worktree add -b audit/project-genesis-evidence ../uni-audit origin/feature/project-genesis-research
```
If that branch name or worktree already exists, inspect and use a unique safe name rather than deleting/replacing.

Run audit scripts with Python from worktree:
```bash
python tools/audit/scripts/branch_inventory.py --out audit_runs/branch_inventory.json
python tools/audit/scripts/module_inventory.py --root . --out audit_runs/module_inventory.json
python tools/audit/scripts/run_tests.py --timeout 180 --outdir audit_runs/pytest_full -- python -m pytest -q -ra
```
Do not assume `tools/audit` exists until the attached audit pack is copied there. See README.
