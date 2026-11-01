"""Independent CLI so existing reference CLI remains fully backward compatible."""
import argparse,json
from pathlib import Path
import numpy as np
from .evolution import pm_evolve,halo_candidates,galaxy_population,stellar_and_planetary_catalog,evolutionary_ecology

def main():
    p=argparse.ArgumentParser(description='Integrated reduced-order experimental universe')
    p.add_argument('--grid',type=int,default=16);p.add_argument('--steps',type=int,default=140)
    p.add_argument('--seed',type=int,default=4242);p.add_argument('--amplitude',type=float,default=.09)
    p.add_argument('--output',default='runs/integrated-toy');p.add_argument('--generations',type=int,default=200)
    a=p.parse_args()
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    pos,vel,density,history=pm_evolve(n=a.grid,steps=a.steps,seed=a.seed,amplitude=a.amplitude)
    halos=halo_candidates(density)
    galaxies=galaxy_population(halos,seed=a.seed+1)
    systems=stellar_and_planetary_catalog(galaxies,seed=a.seed+2)
    evolution=evolutionary_ecology(generations=a.generations,seed=a.seed+3)
    np.savez_compressed(output/'nbody.npz',positions=pos,velocities=vel,density=density)
    results={'scientific_status':'INTEGRATED TOY MODELS: nonlinear periodic particle-mesh gravity; local density peaks not bound halos; parameterized galaxies/stars/planets; separately explicitly seeded evolutionary artificial life; no ab-initio life or galaxy formation',
    'seed':a.seed,'grid':a.grid,'steps':a.steps,'amplitude':a.amplitude,'halo_candidates':len(halos),'galaxies_proxy':len(galaxies),'sampled_stellar_systems':len(systems),'sampled_planets':sum(len(s['planets']) for s in systems),'ecology_generations':a.generations,'history':history,'halos':halos,'galaxies':galaxies,'stellar_systems':systems,'evolution':evolution}
    (output/'integrated_results.json').write_text(json.dumps(results,indent=2))
    print(json.dumps({k:v for k,v in results.items() if k not in ['history','halos','galaxies','stellar_systems','evolution']},indent=2))
if __name__=='__main__':main()
