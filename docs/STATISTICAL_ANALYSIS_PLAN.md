# Preregistered Statistical Analysis Plan — Universe Engine Sprint 14

**Version/date:** v1.0, `2026-11-01T20:45:00Z`  
**Audited SHA:** `7f4f09b86a44bafd5f92bfe84ec04f2385e7ff1d` (`feature/project-genesis-research`, Sprint 13)  
**Audit branch:** `audit/project-genesis-evidence`  
**Status:** PREREGISTERED — written BEFORE any new training runs. No outcomes have been seen under this plan.

## Research question / confirmatory hypothesis

**Primary outcome:** held-out episodic return on ONE prespecified memory-required environment (delayed-cue).  
**Primary comparison:** learned recurrent/world-model controller vs. **learned feedforward reactive baseline** on matched held-out episodes.  
**Estimand:** paired per-training-seed difference `d_i = reward_trainable_memory_i − reward_learned_reactive_i`.  
**Separate fully observable GridWorld** as a negative-control benchmark (memory should not help when current observations suffice).  
**Do not** substitute a hand-coded baseline (oracle or hand-tuned reactive) for the learned reactive baseline in the primary comparison.  
**Programmed memory oracle** is a separate positive-control reference, compared against but NOT part of the primary confirmatory claim.

## Independent replication unit

Independent training run/seed; average held-out evaluations within each run.  
Target: **≥20 independent training seeds** where computationally feasible; otherwise a justified smaller number + status `INCONCLUSIVE` (never quietly reduce the quality gate).  
Re-use matched evaluation environments and seeds within each replicate.  
**No overlap** between training, hyperparameter-tuning, and final-evaluation seeds.  
Avoid pseudoreplication: many episodes from one trained agent do NOT constitute many independent trained agents. Aggregate per training seed BEFORE paired analysis.

## Budget and matched baselines (to be fixed before running)

Specify, before running: training episodes/steps, compute time, policy capacity (hidden dim, parameter count), action space, observations, hyperparameter selection, reset rules, optimizer, learning rate, and reward shaping.  
Matched conditions: same architecture capacity where comparable, same episode budget, same evaluation seeds, same action/observation interface (`direction` vs `dx`/`dy` must be consistent across agents), same reward/next-observation alignment, same RNG handling (no hidden-state reset between episodes unless predeclared), same candidate-action evaluation protocol (must not corrupt hidden state).

## Primary analysis

For each independent training replicate `i`, compute `mean(candidate_return − reactive_return)` over matched held-out episodes.  
Calculate **95% CI for the mean paired difference** (bootstrap at the training-replicate level; pre-specify iterations and RNG seed).  
Effect size: paired standardized difference (mean difference / SD of differences).  
For strongly non-normal/high-variance distributions, report a robust sensitivity analysis.  
**Do not** equate a CI for each policy mean with a CI for the difference; individual CIs overlapping/including zero do not determine pairwise significance.

## Secondary comparisons / multiplicity

List in advance: (a) trainable vs. programmed memory oracle (separate reference); (b) recurrent vs. world-model vs. gated-recurrent; (c) occluded-target task; (d) changed-dynamics task; (e) per-component causal ablations (recurrence, workspace, self-model, world model, planning, episodic memory).  
Apply **Holm step-down correction** to secondary p-values; report unadjusted and adjusted p-values.  
No post-hoc subgroup selection for confirmatory claims.

## Failure / missingness handling

Crashes and timeouts are logged as failures; do not exclude outcomes without a predeclared rule. Provide all seeds and raw outputs. Never re-run only unfavorable seeds. Timeouts recorded as `TIMEOUT`, never as pass.

## Stop rule

Fixed number of independent training runs and evaluations defined **before** seeing outcomes; no optional stopping for favorable results.

## Environment / reproducibility

Commit SHA (`7f4f09b`), Python/library versions, hardware, RNG settings, CPU threads, benchmark config, and raw JSON/CSV data recorded. Canonical serialization; field-by-field difference reporting.

## Report wording

A nonsignificant result is insufficient evidence of a positive advantage; it does **not** prove equivalence or "no benefit." Report the effect estimate and its uncertainty, never "proved null." If any reported positive result fails to replicate, keep the negative result.
