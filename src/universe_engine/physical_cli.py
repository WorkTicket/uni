"""CLI exercising executable physical submodels, with scientific honesty."""
import argparse,json
from pathlib import Path
import numpy as np
from .physical_models import gas_evolve,stellar_track,disk_accretion,prebiotic_network

def main():
    parser=argparse.ArgumentParser(description='Multiphysics component demo (approximate models)')
    parser.add_argument('--output',default='runs/physics-demo')
    parser.add_argument('--gas-grid',type=int,default=128)
    parser.add_argument('--gas-steps',type=int,default=300)
    args=parser.parse_args()
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    gas,g=gas_evolve(n=args.gas_grid,steps=args.gas_steps)
    disk=disk_accretion()
    chemistry=prebiotic_network()
    np.savez_compressed(output/'physical_fields.npz',gas_density=gas.rho,gas_momentum=gas.momentum,gas_energy=gas.energy,metals=gas.metals,stellar_mass_density=gas.stars,disk_radii=disk['radii_AU'],disk_dust=disk['dust_earth_masses'],disk_planetesimals=disk['planetesimals_earth_masses'],chemistry_time=chemistry['time'],polymer=chemistry['polymer_P'])
    result={'scientific_status':'Executable coarse 1D Euler gas, analytic main sequence estimates, annular coagulation, driven chemical kinetics; NOT validated cosmological hydrodynamics, stellar evolution, planet formation or abiogenesis', 'gas':g, 'stellar_track_1_solar_mass':stellar_track(1.0), 'disk_total_mass_earth':disk['total_solids_earth_masses'], 'disk_planetesimal_mass_earth':float(disk['planetesimals_earth_masses'].sum()), 'prebiotic_last_concentrations':chemistry['last_concentrations']}
    (output/'physical_results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
