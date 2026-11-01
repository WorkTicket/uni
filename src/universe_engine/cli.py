import argparse, hashlib, json
from pathlib import Path
import yaml
import numpy as np
from .physics import solve
from .field import gaussian_field

def main():
    p=argparse.ArgumentParser(description='FLRW + toy linear perturbations (not a full universe)')
    p.add_argument('command',choices=['run','validate','benchmark','doctor'])
    p.add_argument('target',nargs='?',default=None,help='benchmark target (hydro)')
    p.add_argument('--output-root',default=None,help='benchmark output root (default runs/benchmarks/<target>)')
    p.add_argument('--config',default='config/reference.yaml')
    a=p.parse_args()
    if a.command=='doctor':
        from .doctor import main as doctor_main
        raise SystemExit(0 if doctor_main()['ok'] else 1)
    if a.command=='benchmark':
        if a.target not in ('hydro','gravity','nbody'): p.error('benchmark target must be one of: hydro, gravity, nbody')
        import importlib
        run_all=importlib.import_module(f'.benchmarks.{a.target}',__package__).run_all
        metrics,out=run_all(a.output_root or f'runs/benchmarks/{a.target}',f'universe benchmark {a.target}')
        for b in metrics['benchmarks']:
            print(f"{'PASS' if b['passed'] else 'FAIL'}  {b['name']}  ({sum(c['passed'] for c in b['checks'])}/{len(b['checks'])} checks)")
        print(f"{metrics['checks_passed']}/{metrics['checks_total']} checks passed; results in {out}")
        raise SystemExit(0 if metrics['all_passed'] else 1)
    path=Path(a.config)
    raw=path.read_bytes()
    config=yaml.safe_load(raw)
    cosm=config['cosmology']
    solution=solve(cosm)
    fieldcfg=config['field']
    n=int(fieldcfg['grid_size'])
    if fieldcfg['box_size_Mpc'] <= 0: raise ValueError('box_size_Mpc must be positive')
    field=gaussian_field(n,int(config['run']['seed']),float(fieldcfg['initial_rms']),float(fieldcfg['spectral_index']))
    if a.command=='validate':
        print('PASS: configuration, flat FLRW integration and toy density realization')
        return
    output=Path(config['run']['output']); output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'background.npz', **solution)
    np.save(output/'initial_overdensity.npy',field)
    # Save latest scaled linear overdensity for inspection: no nonlinear collapse.
    growth_ratio=solution['D'][-1]/solution['D'][0]
    np.save(output/'final_linear_overdensity.npy',field*growth_ratio)
    metadata={'status':'illustrative linear-theory demonstration, NOT a physical end-to-end universe',
      'config_sha256':hashlib.sha256(raw).hexdigest(), 'seed':int(config['run']['seed']),
      'box_size_Mpc':fieldcfg['box_size_Mpc'], 'n':n,
      'a_start':float(solution['a'][0]), 'a_end':float(solution['a'][-1]),
      'elapsed_Gyr':float(solution['elapsed_Gyr'][-1]),
      'H0_km_s_Mpc':float(solution['H'][-1]),
      'initial_rms':float(field.std()),'final_linear_rms':float(field.std()*growth_ratio),
      'warning':'final density contrast may exceed linear regime; no transfer function, N-body, baryons, stars, or chemistry'}
    (output/'manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(metadata,indent=2))

if __name__=='__main__': main()
