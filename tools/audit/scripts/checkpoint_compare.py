#!/usr/bin/env python3
"""Compare two JSON-serializable snapshots exported by the actual simulation. Does not generate snapshots."""
import argparse, hashlib, json, math, pathlib

def canonical(x): return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)

def walk(a,b,path='$',tol=0.0,errors=None):
    errors=[] if errors is None else errors
    if isinstance(a,dict) and isinstance(b,dict):
        for k in sorted(set(a)|set(b)):
            if k not in a or k not in b: errors.append({'path':f'{path}.{k}','left':a.get(k,'<MISSING>'),'right':b.get(k,'<MISSING>')})
            else: walk(a[k],b[k],f'{path}.{k}',tol,errors)
    elif isinstance(a,list) and isinstance(b,list):
        if len(a)!=len(b): errors.append({'path':path+'.length','left':len(a),'right':len(b)})
        for i,(x,y) in enumerate(zip(a,b)): walk(x,y,f'{path}[{i}]',tol,errors)
    elif isinstance(a,(float,int)) and isinstance(b,(float,int)) and not isinstance(a,bool) and not isinstance(b,bool):
        if not math.isclose(a,b,rel_tol=tol,abs_tol=tol): errors.append({'path':path,'left':a,'right':b})
    elif a!=b: errors.append({'path':path,'left':a,'right':b})
    return errors

def main():
    ap=argparse.ArgumentParser();ap.add_argument('uninterrupted');ap.add_argument('resumed');ap.add_argument('--tol',type=float,default=0.0);ap.add_argument('--out',default='audit_runs/checkpoint_diff.json');args=ap.parse_args()
    a=json.loads(pathlib.Path(args.uninterrupted).read_text(encoding='utf-8'));b=json.loads(pathlib.Path(args.resumed).read_text(encoding='utf-8')); errors=walk(a,b,tol=args.tol);res={'equal':not errors,'tolerance':args.tol,'left_sha256':hashlib.sha256(canonical(a).encode()).hexdigest(),'right_sha256':hashlib.sha256(canonical(b).encode()).hexdigest(),'mismatch_count':len(errors),'mismatches':errors[:500]}
    p=pathlib.Path(args.out);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res,indent=2),encoding='utf-8');print(f"{'MATCH' if not errors else 'MISMATCH'}: {len(errors)} mismatched paths; details in {p}")
    if errors: raise SystemExit(1)
if __name__=='__main__': main()
