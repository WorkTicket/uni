"""Executable reduced-resolution physical modules with explicit limits.

The gas solver is one-dimensional Euler hydrodynamics; stellar model uses
analytic fits; disk model is a conservative annular coagulation approximation;
prebiotic model is a chemically explicit mass-action network. None proves
end-to-end emergent galaxies, planets, or life.
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class GasState:
    rho: np.ndarray
    momentum: np.ndarray
    energy: np.ndarray
    metals: np.ndarray
    stars: np.ndarray


def gas_pressure(state, gamma=5/3):
    kinetic=.5*state.momentum**2/np.maximum(state.rho,1e-30)
    return np.maximum((gamma-1)*(state.energy-kinetic),1e-10)


def _flux(U, gamma):
    rho=np.maximum(U[0],1e-12)
    v=U[1]/rho
    p=np.maximum((gamma-1)*(U[2]-.5*rho*v*v),1e-10)
    return np.stack((U[1],rho*v*v+p,(U[2]+p)*v,U[3]*v,U[4]*v)),np.abs(v)+np.sqrt(gamma*p/rho)


def gas_step(state, dx, dt, cooling=0.003, star_efficiency=0.0, gravity=None, gamma=5/3):
    """Periodic Rusanov finite-volume Euler + cooling + local star formation.

    Metals and stellar masses are densities. Gravity is optional acceleration
    array. Cooling removes gas energy (radiation escapes); gas-to-star transfer
    removes mass and specific kinetic/internal energy. Only coarse 1D physics.
    """
    U=np.stack((state.rho,state.momentum,state.energy,state.metals,state.stars))
    F,s=_flux(U,gamma)
    maxspeed=float(np.max(s))
    if dt<=0 or dx<=0 or dt*maxspeed/dx>0.45: raise ValueError('CFL violation')
    Fr=.5*(F+np.roll(F,-1,axis=1))-.5*np.maximum(s,np.roll(s,-1))[None,:]*(np.roll(U,-1,axis=1)-U)
    V=U-dt/dx*(Fr-np.roll(Fr,1,axis=1))
    if gravity is not None:
        accel=np.broadcast_to(np.asarray(gravity,float),V[0].shape)
        v_before=V[1]/np.maximum(V[0],1e-12)
        V[1]+=dt*V[0]*accel
        V[2]+=dt*V[0]*v_before*accel+.5*V[0]*(dt*accel)**2
    if np.any(V[0]<=0) or np.any(V[2]-.5*V[1]**2/V[0]<=0):
        raise FloatingPointError('nonphysical gas state; shorten timestep')
    v=V[1]/V[0]
    thermal=V[2]-.5*V[0]*v*v
    # Explicit cooling, thermal energy converted into escaping radiation.
    lost=np.minimum(thermal*.3,dt*cooling*V[0]**2*np.sqrt(np.maximum(thermal/V[0],0)))
    V[2]-=lost
    # Temperature/overdensity proxy for cold star-forming gas.
    cold=V[0]>1.2
    formed=np.where(cold,np.minimum(.05*V[0],dt*star_efficiency*V[0]),0.)
    fraction=formed/V[0]
    V[4]+=formed
    V[0]-=formed
    V[1]*=(1-fraction)
    V[2]*=(1-fraction)
    V[3]*=(1-fraction)
    return GasState(*V),{'radiated_energy':float(np.sum(lost)*dx),'new_stellar_mass':float(np.sum(formed)*dx)}


def gas_evolve(n=128, steps=300, cooling=.003, star_efficiency=.015):
    """Periodic sinusoidal 1D gas with mild perturbation and self-contained run."""
    x=(np.arange(n)+.5)/n
    rho=1+.38*np.cos(2*np.pi*x)
    v=np.zeros(n)
    pressure=np.ones(n)*.15
    state=GasState(rho,rho*v,pressure/(5/3-1)+.5*rho*v*v,rho*.001,np.zeros(n))
    diagnostics={'radiated_energy':0.,'new_stellar_mass':0.}
    for _ in range(steps):
        speeds=np.abs(state.momentum/state.rho)+np.sqrt((5/3)*gas_pressure(state)/state.rho)
        dt=.28/(n*np.max(speeds))
        state,d=gas_step(state,1/n,dt,cooling,star_efficiency)
        for key,value in d.items():diagnostics[key]+=value
    diagnostics['gas_mass']=float(state.rho.mean())
    diagnostics['stellar_mass']=float(state.stars.mean())
    diagnostics['mass_error']=float(abs(state.rho.mean()+state.stars.mean()-1))
    return state,diagnostics


def stellar_track(mass_solar, metallicity=.0142, age_gyr=0.0):
    """Bounded analytic main-sequence estimates; NOT stellar structure solver.

    Representative approximation L~M^3.5, R~M^0.8; mass loss and red giants
    are outside this model. Use detailed stellar tracks for physical science.
    """
    m=float(mass_solar)
    if not .08<=m<=30 or not 0<=metallicity<=.05 or age_gyr<0:raise ValueError('out-of-range stellar parameters')
    luminosity=m**3.5
    radius=m**.8
    lifetime=10*m/luminosity
    teff=5772*(luminosity/radius**2)**.25
    return {'mass_solar':m,'luminosity_solar_approx':luminosity,'radius_solar_approx':radius,'temperature_K_approx':teff,'lifetime_gyr_approx':lifetime,'on_main_sequence_approx':bool(age_gyr<lifetime),'metallicity':metallicity}


def disk_accretion(bins=64,steps=300,dt=.002,collision_rate=1.0):
    """Annular dust-to-planetesimal conversion conserving solid mass.

    This is local two-bin coagulation, not a resolved N-body protoplanetary disk.
    """
    if bins<4 or not 0<dt*collision_rate<=.1:raise ValueError('invalid disk settings')
    radii=np.geomspace(.1,30,bins)
    dust=np.exp(-radii/8)/radii
    dust=dust/dust.sum()*100.0 # total Earth masses of solids
    bodies=np.zeros(bins)
    for _ in range(steps):
        rate=collision_rate*dust*dust/(1+dust)
        converted=np.minimum(dust,dt*rate)
        dust-=converted; bodies+=converted
    return {'radii_AU':radii,'dust_earth_masses':dust,'planetesimals_earth_masses':bodies,'total_solids_earth_masses':float(dust.sum()+bodies.sum())}


def prebiotic_network(steps=2000,dt=.01,feed=.08,decay=.025):
    """Well-mixed driven A+B -> P; P+A -> 2P; P -> A; with bounded feed.

    A nutrient, B precursor, P autocatalytic polymer proxy. *Not life*.
    Explicit mass-action fluxes and an external resource reservoir.
    """
    if steps<1 or not 0<dt<=.05:raise ValueError('invalid reaction integration')
    a,b,p=1.,.7,.001
    samples=[]
    for i in range(steps):
        synth=.12*a*b
        copy=.08*p*a
        breakdown=.04*p
        # external feed and leakage produce an open, driven chemistry system
        da=feed*(1-a)-synth-copy+breakdown
        db=feed*(.7-b)-synth
        dp=synth+copy-breakdown-decay*p
        a,b,p=a+dt*da,b+dt*db,p+dt*dp
        if min(a,b,p)<-1e-9:raise FloatingPointError('negative concentration')
        if i%(max(1,steps//40))==0:samples.append((i*dt,a,b,p))
    return {'time':np.asarray(samples)[:,0], 'nutrient_A':np.asarray(samples)[:,1], 'precursor_B':np.asarray(samples)[:,2],'polymer_P':np.asarray(samples)[:,3],'last_concentrations':{'A':a,'B':b,'P':p}}
