#!/usr/bin/env python3
"""Paired seed-level analysis, with CI on paired difference (no pseudoreplication)."""
import argparse, csv, json, math, pathlib, random, statistics

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--csv',required=True); ap.add_argument('--out',default='audit_runs/paired_result.json'); ap.add_argument('--baseline-col',default='baseline'); ap.add_argument('--candidate-col',default='candidate'); ap.add_argument('--seed-col',default='training_seed'); ap.add_argument('--bootstrap',type=int,default=10000); args=ap.parse_args()
    rows=list(csv.DictReader(open(args.csv,newline='',encoding='utf-8')))
    if not rows: ap.error('Empty CSV')
    seeds=set(); diffs=[]
    for row in rows:
        seed=row[args.seed_col]
        if seed in seeds: ap.error('Multiple rows per independent training seed. Aggregate evaluation episodes per training seed BEFORE paired analysis.')
        seeds.add(seed); diffs.append(float(row[args.candidate_col])-float(row[args.baseline_col]))
    if len(diffs)<2: ap.error('Need >=2 independent pairs')
    rng=random.Random(83121); n=len(diffs)
    boots=sorted(sum(diffs[rng.randrange(n)] for _ in range(n))/n for _ in range(args.bootstrap))
    mean=statistics.mean(diffs); sd=statistics.stdev(diffs)
    result={'n_independent_pairs':n,'paired_difference_mean':mean,'paired_difference_sd':sd,'paired_cohens_dz':mean/sd if sd else None,'bootstrap_percentile_95_CI':[boots[int(.025*len(boots))],boots[min(len(boots)-1,int(.975*len(boots)))]] ,'individual_differences':{row[args.seed_col]:float(row[args.candidate_col])-float(row[args.baseline_col]) for row in rows},'method':'seed-level paired bootstrap percentile interval; descriptive/exploratory unless analysis preregistered; no multiplicity correction automatically','warning':'For complex nested evaluation, cluster by independent training replicate. Confirm legitimate pairing and use pre-specified Holm correction for secondary hypotheses.'}
    path=pathlib.Path(args.out); path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='individual_differences'},indent=2))
if __name__=='__main__': main()
