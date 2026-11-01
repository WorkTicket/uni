# Universe Engine scientific audit pack

This bundle is for a *read-only source audit*, not a code repair sprint. Give `KILO_MASTER_PROMPT.md` to Kilo and copy the entire package to `tools/audit/` on a dedicated audit worktree. Files under `scripts/` are stdlib Python 3.10+ (with optional installed numpy for external repo adapters). They inspect Git metadata / Python source, run bounded test commands, calculate paired statistics from CSV, and compare JSON checkpoints.

**Important:** scripts are infrastructure, not proof of scientific correctness. Kilo must inspect source and run actual repository-specific experiments and record outputs. Repo paths, available tests, and checkpoint APIs must be discovered rather than assumed.

Suggested first steps:

```bash
python tools/audit/scripts/branch_inventory.py --out audit_runs/branch_inventory.json
python tools/audit/scripts/module_inventory.py --root . --out audit_runs/module_inventory.json
python tools/audit/scripts/run_tests.py --timeout 180 --outdir audit_runs/pytest_full -- python -m pytest -q -ra
```

Windows PowerShell equivalent:

```powershell
python tools/audit/scripts/branch_inventory.py --out audit_runs/branch_inventory.json
python tools/audit/scripts/module_inventory.py --root . --out audit_runs/module_inventory.json
python tools/audit/scripts/run_tests.py --timeout 180 --outdir audit_runs/pytest_full -- python -m pytest -q -ra
```

Scripts produce output directories automatically. They do not edit scientific source or existing historical status files.
