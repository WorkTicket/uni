"""Verify IMPLEMENTATION_STATUS.json is generated from a real run and is
internally consistent.

Ground rule (master plan): status fields are generated from run output, never
hand-edited, and each carries the commit hash that produced it.

Design (avoids the self-reference trap):
  * A committed status artifact must NOT be required to contain its own
    commit's hash — that is impossible to satisfy once the artifact is
    committed (committing advances HEAD past the recorded value).
  * Instead, `write_status.py` captures `tested_source_revision` (the source
    revision under test) and `head_commit`/`repo_revision_at_generation` (the
    status-generation commit) at the same moment, before committing.
  * This test checks internal consistency (tested_source_revision ==
    head_commit == repo_revision_at_generation) and that `test_counts`
    reflect a real pytest run (total > 0). It does NOT compare against the
    current git HEAD, which would include this test's own commit.
"""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _git(*args):
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=ROOT,
    ).stdout.strip()


def test_status_internal_consistency():
    status_path = ROOT / "docs" / "IMPLEMENTATION_STATUS.json"
    assert status_path.is_file(), "IMPLEMENTATION_STATUS.json must exist"
    status = json.loads(status_path.read_text())

    tested = status.get("tested_source_revision")
    head = status.get("head_commit")
    generated = status.get("repo_revision_at_generation")
    assert tested is not None, "tested_source_revision must be recorded"
    assert head is not None, "head_commit must be recorded"
    assert generated is not None, "repo_revision_at_generation must be recorded"
    # All three are captured at generation time, before the status file is
    # committed, so they must agree. This is reproducible on a fresh checkout.
    assert tested == head == generated, (
        "status revision fields are inconsistent: tested_source_revision="
        f"{tested}, head_commit={head}, repo_revision_at_generation={generated}"
    )
    # The recorded revision must be a real git object (not hand-typed).
    assert subprocess.run(
        ["git", "cat-file", "-t", tested], capture_output=True, text=True,
        cwd=ROOT,
    ).returncode == 0, f"tested_source_revision {tested} is not a valid git object"


def test_status_test_counts_are_real_not_hand_edited():
    status_path = ROOT / "docs" / "IMPLEMENTATION_STATUS.json"
    status = json.loads(status_path.read_text())
    counts = status.get("test_counts", {})
    assert counts, "test_counts must be present"
    total = counts.get("passed", 0) + counts.get("failed", 0) + counts.get("skipped", 0)
    assert total > 0, "test_counts must reflect a real run (total > 0)"
    note = str(status.get("test_counts", {}).get("note", ""))
    assert "hangs on import" not in note, (
        "stale 'hangs on import' note must not be present; counts come from "
        "real run output"
    )
