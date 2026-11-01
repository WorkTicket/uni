#!/usr/bin/env python3
"""Read-only git branch/SHA/worktree inventory. No checkout, merging or editing."""
import argparse, datetime, json, pathlib, subprocess, sys

def git(*args):
    p = subprocess.run(['git', *args], text=True, capture_output=True)
    return {'ok': p.returncode == 0, 'code': p.returncode, 'out': p.stdout.strip(), 'err': p.stderr.strip()}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out', default='audit_runs/branch_inventory.json'); args=ap.parse_args()
    commands={'top':('rev-parse','--show-toplevel'), 'head':('rev-parse','HEAD'), 'status':('status','--porcelain=v1'), 'branches':('for-each-ref','--format=%(refname) %(objectname) %(upstream)'), 'worktrees':('worktree','list','--porcelain'), 'log':('log','--all','--decorate','--oneline','-n','50')}
    result={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commands':{name:git(*cmd) for name,cmd in commands.items()}}
    dest=pathlib.Path(args.out); dest.parent.mkdir(parents=True,exist_ok=True); dest.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(f'Inventory recorded: {dest}'); print(f'HEAD: {result["commands"]["head"]["out"]}')
    if not result['commands']['top']['ok']: sys.exit('Not inside a git repository')
if __name__=='__main__': main()
