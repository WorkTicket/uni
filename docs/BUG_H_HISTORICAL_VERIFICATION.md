# Bug H Historical Verification Evidence

## Scope
Stage 1 historical verification: prove the existing Bug H regression test fails on the pre-fix implementation and passes on the post-fix implementation.

## Tested revisions
- Pre-fix: 54b7fc1a77e10177bc0884ad89d1a35d050197b2
- Post-fix: 8a4e7f188b5ec6b0a437805ec4ecdbef9960ca38

## Test logic (unchanged between runs)
`tests/observatory/playwright_viewer.js` — Bug H regression test:
- Opens real Chromium (headless) against a live FastAPI/uvicorn observatory server.
- Asserts generation reaches "Ready" (not stuck on "Preparing...") and the Begin button is enabled.
- Also asserts Bug B orbit-rate behaviour after generation.

## Environment
- Browser / renderer: Chromium 130.0.6723.31 (headless, via Playwright 1.48.2)
- Server: FastAPI 0.162.0 + uvicorn 0.54.0
- Python: .buildenv (universe_engine_starter 0.1.0 editable install)
- Server port: 8001 (OBSERVATORY_URL=http://127.0.0.1:8001)
- Each run launched the server from the revision's own `src/` via PYTHONPATH so the tested revision's code was exercised (not the editable main-tree source).

## Results

### Pre-fix 54b7fc1 — FAILS (expected Bug H)
- Command: `node tests/observatory/playwright_viewer.js` with OBSERVATORY_URL=http://127.0.0.1:8001
- Exit code: 2 (FATAL)
- Failing assertion: `waitForReady` — `page.waitForFunction` timed out after 30000ms waiting for `#prog-label` to include "Ready" (playwright_viewer.js:19, line 69 call).
- Detail: app stayed stuck on "Preparing..."; generation never reached Ready; Begin button remained disabled. This is the original Bug H (unhandled exception in `bindUI()`/`main.js` boot, no staged loader).
- Classification: Expected failure caused by the original Bug H (not test-infrastructure failure, not unexpected source failure).

### Post-fix 8a4e7f1 — PASSES
- Command: `node tests/observatory/playwright_viewer.js` with OBSERVATORY_URL=http://127.0.0.1:8001
- Exit code: 0
- Passing assertions (all 6):
  1. Bug H: generation reaches Ready (not stuck on Preparing)
  2. Bug H: Begin button enabled after generation — prog label "Ready — 22,000 halos, seed 436586338"
  3-6. Bug B: reached a star system / system has planets / rate = innermost period / 30s / innermost orbit >= 30 wall-seconds
- PASS: 6, FAIL: 0

## Verdict
VERIFIED: Pre-fix fails for the expected Bug H reason; post-fix passes. Same test logic run against real browser/server behavior for both revisions.

## Notes
- Pre-fix revision does not contain the Playwright test file; the SAME test logic from the post-fix/main tree was executed against both implementations via an isolated runner (no source modification to either revision).
- The /health endpoint exists only in post-fix (added by the Bug H fix); pre-fix returns 404 for /health.
- No source code in either revision was modified. No scratch worktrees or generated dependencies committed.

## Related (documented, not re-run)
- 0b.5b recurrent-learning diagnostics (frozen recurrent weights, hidden state across episode boundaries) — prior completed work, not re-run.
- Status-file validation committed at b9946b5 — needs later independent check that run artifacts correspond to the reported tested revision (not part of this task).
