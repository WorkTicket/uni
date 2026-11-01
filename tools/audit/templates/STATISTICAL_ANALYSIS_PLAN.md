# Preregistered Statistical Analysis Plan — WRITE BEFORE NEW RUNS

**Version/date and SHA:** ...

## Research question / confirmatory hypothesis

Primary outcome: held-out episodic return (predefine reward and direction). Primary comparison: learned recurrent/world-model controller vs learned feedforward reactive baseline on ONE prespecified memory-required environment. Separate fully observable GridWorld as a negative-control benchmark. Do not substitute a hand-coded baseline for learned baseline in the primary comparison.

## Independent replication unit

Independent training run/seed; average held-out evaluations within each run. Use >=20 independent training seeds if resources permit, else justified number + INCONCLUSIVE. Re-use matched evaluation environments and seeds within each replicate. No overlap between training, hyperparameter tuning, and final evaluation seeds.

## Budget and matched baselines

Specify training episodes/steps, compute time, policy capacity, action space, observations, hyperparameter selection and reset rules BEFORE running.

## Primary analysis

For each independent training replicate, compute mean(candidate return - learned reactive return) over matched held-out episodes. Calculate 95% CI for mean **paired difference** (bootstrap at training replicate level, pre-specify iterations/seed). Effect size: paired standardized difference. For strongly nonnormal/high-variance distributions report robust sensitivity analysis. Do not equate CI for each policy mean with CI for difference.

## Secondary comparisons / multiplicity

List all secondary hypotheses in advance. Apply Holm step-down correction to secondary p-values; report unadjusted and adjusted p-values. Do not use post hoc subgroup selection for confirmatory claims.

## Failure/missingness handling

Crashes and timeouts are logged as failures; do not exclude outcomes without predeclared rule. Provide all seeds and raw outputs. Never re-run only unfavorable seeds.

## Stop rule

Fixed number of independent training runs and evaluations defined before seeing outcomes; no optional stopping for favorable results.

## Environment / reproducibility

Commit SHA, Python/library versions, hardware, RNG settings, CPU threads, benchmark config and JSON/CSV raw data.

## Report wording

A nonsignificant result is insufficient evidence of positive advantage; it does not prove equivalence or no benefit. Report effect estimate and uncertainty, not 'proved null.'
