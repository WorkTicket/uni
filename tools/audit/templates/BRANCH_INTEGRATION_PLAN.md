# Safe Branch Integration Plan — DESIGN ONLY / NO MERGES

## Existing refs (run git for-each-ref)

| Ref | SHA | Merge-base with research | Unique modules | Potential conflicts |
|---|---|---|---|---|
| | | | | |

## Safeguards

- No deletion, force pushes, history rewriting, hard reset or branch replacement.
- Preserve local working tree with `git status` and worktrees.
- Preserve branch snapshots with unique tags or SHA manifest.
- Create a dedicated future integration branch after user approval.
- Resolve dependency and semantic API conflicts before actual merge.
- Re-run all tests and essential conservation/cosmology/persistence benchmarks after each proposed change.

## Sequence after user approval

1. ... (record exact source and target SHAs)
2. ... (describe merging and integration test gates)
3. ...

## Rollback design

Describe new integration branch removal only after audit backup verification, never delete original refs.
