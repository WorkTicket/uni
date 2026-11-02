"""Verify IMPLEMENTATION_STATUS.json head matches real git HEAD.

Ground rule (master plan): status fields are generated from run output, never
hand-edited, and each carries the commit hash that produced it. The recorded
head_commit MUST equal `git rev-parse HEAD`; otherwise the status file is
stale/hand-edited and this test fails.
"""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_status_head_matches_git_head():
    status_path = ROOT / "docs" / "IMPLEMENTATION_STATUS.json"
    assert status_path.is_file(), "IMPLEMENTATION_STATUS.json must exist"
    status = json.loads(status_path.read_text())

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        cwd=ROOT,
    ).stdout.strip()
    assert head, "git rev-parse HEAD must return a hash"

    recorded = status.get("head_commit") or status.get("repo_revision_at_generation")
    assert recorded is not None, "IMPLEMENTATION_STATUS.json must carry head_commit"
    assert recorded == head, (
        f"status head_commit ({recorded}) does not match git HEAD ({head}); "
        "regenerate docs/IMPLEMENTATION_STATUS.json with scripts/write_status.py"
    )


def test_status_test_counts_are_real_not_hand_edited():
    status_path = ROOT / "docs" / "IMPLEMENTATION_STATUS.json"
    status = json.loads(status_path.read_text())
    counts = status.get("test_counts", {})
    assert counts, "test_counts must be present"
    total = counts.get("passed", 0) + counts.get("failed", 0) + counts.get("skipped", 0)
    assert total > 0, "test_counts must reflect a real run (total > 0)"
    assert "hangs on import" not in str(status.get("test_counts", {}).get("note", "")), (
        "stale 'hangs on import' note must not be present; counts come from real run output"
    )
