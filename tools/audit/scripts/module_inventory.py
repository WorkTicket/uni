#!/usr/bin/env python3
"""Read-only Python AST inventory; DOES NOT assert implementation quality."""
import argparse, ast, datetime, hashlib, json, pathlib, subprocess

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='.'); ap.add_argument('--out',default='audit_runs/module_inventory.json'); args=ap.parse_args()
    root=pathlib.Path(args.root).resolve()
    git=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True)
    output={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'repo':str(root),'sha':git.stdout.strip() if git.returncode==0 else None,'files':[],'note':'AST presence inventory only; does not validate behavior.'}
    for p in sorted(root.rglob('*.py')):
        if any(x in p.parts for x in ('.git','.venv','venv','node_modules','__pycache__','build','dist')): continue
        rel=str(p.relative_to(root)); data=p.read_bytes(); entry={'path':rel,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
        try:
            tree=ast.parse(data, filename=rel); entry['classes']=[n.name for n in tree.body if isinstance(n,ast.ClassDef)]; entry['functions']=[n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]; entry['parse']='ok'
        except (SyntaxError,UnicodeError) as e: entry['parse']='error'; entry['error']=str(e)
        output['files'].append(entry)
    dest=pathlib.Path(args.out); dest.parent.mkdir(parents=True,exist_ok=True); dest.write_text(json.dumps(output,indent=2),encoding='utf-8'); print(f"{len(output['files'])} Python files inventoried at {dest}")
if __name__=='__main__': main()
