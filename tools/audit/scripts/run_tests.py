#!/usr/bin/env python3
"""Bounded process runner writing exact stdout/stderr and JSON manifest; no test editing."""
import argparse, datetime, hashlib, json, os, pathlib, platform, re, subprocess, sys, time

def main():
    ap=argparse.ArgumentParser(description='Run command with timeout and durable logs.'); ap.add_argument('--timeout',type=float,default=180); ap.add_argument('--outdir',required=True); ap.add_argument('command',nargs=argparse.REMAINDER); args=ap.parse_args()
    cmd=args.command[1:] if args.command[:1]==['--'] else args.command
    if not cmd: ap.error('Use -- command arg1 arg2')
    dest=pathlib.Path(args.outdir); dest.mkdir(parents=True,exist_ok=True)
    meta={'command':cmd,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'timeout_seconds':args.timeout,'python':sys.version,'platform':platform.platform(),'cwd':os.getcwd()}
    try:
        g=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,timeout=10); meta['git_sha']=g.stdout.strip() if g.returncode==0 else None
    except Exception: meta['git_sha']=None
    start=time.monotonic(); rc=None; timed_out=False
    try:
        with (dest/'stdout.log').open('w',encoding='utf-8') as out, (dest/'stderr.log').open('w',encoding='utf-8') as err:
            p=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=(os.name!='nt'))
            try: rc=p.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out=True
                if os.name=='nt': subprocess.run(['taskkill','/F','/T','/PID',str(p.pid)],capture_output=True)
                else:
                    import signal
                    try: os.killpg(p.pid,signal.SIGKILL)
                    except ProcessLookupError: pass
                p.wait(timeout=10)
    except OSError as exc: meta['spawn_error']=str(exc)
    meta.update({'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'duration_s':round(time.monotonic()-start,3),'exit_code':rc,'timed_out':timed_out,'status':'TIMEOUT' if timed_out else ('SPAWN_ERROR' if rc is None else ('PASS_EXIT_0' if rc==0 else 'FAIL_EXIT_NONZERO'))})
    for name in ['stdout.log','stderr.log']:
        file=dest/name
        if file.exists(): meta[name+'_sha256']=hashlib.sha256(file.read_bytes()).hexdigest()
    (dest/'run.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps({'status':meta['status'],'exit_code':rc,'duration_s':meta['duration_s'],'outdir':str(dest)},indent=2))
    sys.exit(124 if timed_out else (rc if rc is not None else 127))
if __name__=='__main__': main()
