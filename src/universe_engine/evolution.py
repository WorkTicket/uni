"""Integrated *reduced-order* research demonstrator.

Cosmological comoving particle-mesh nonlinear gravity (periodic box); coarse
halo candidate finder; phenomenological galaxy/stellar/planet populations;
explicitly seeded toy evolutionary ecology. No claim of ab initio life.
Distances in box-fraction coordinates; time variable s=ln(a).
"""
from dataclasses import dataclass
import numpy as np
from .physics import e2
from .field import gaussian_field


def cic_density(pos, n):
    """Cloud-in-cell deposition, normalized so uniform density ~1."""
    q=(pos%1)*n
    low=np.floor(q).astype(int); f=q-low
    rho=np.zeros((n,n,n),dtype=float)
    for x in (0,1):
        for y in (0,1):
            for z in (0,1):
                w=(f[:,0] if x else 1-f[:,0])*(f[:,1] if y else 1-f[:,1])*(f[:,2] if z else 1-f[:,2])
                ix=(low+np.array([x,y,z]))%n
                np.add.at(rho,(ix[:,0],ix[:,1],ix[:,2]),w)
    rho *= (n**3 / len(pos))
    return rho


def sample_vector(grid,pos):
    n=grid.shape[0]; q=(pos%1)*n; low=np.floor(q).astype(int); f=q-low
    out=np.zeros((len(pos),3))
    for x in (0,1):
        for y in (0,1):
            for z in (0,1):
                w=(f[:,0] if x else 1-f[:,0])*(f[:,1] if y else 1-f[:,1])*(f[:,2] if z else 1-f[:,2])
                ix=(low+np.array([x,y,z]))%n
                out+=w[:,None]*grid[ix[:,0],ix[:,1],ix[:,2]]
    return out


def poisson_gradient(delta):
    """FFT derivative of inverse Laplacian: grad Δ^-1 δ, periodic unit box."""
    n=delta.shape[0]
    k=2*np.pi*np.fft.fftfreq(n,d=1/n)
    K=np.meshgrid(k,k,k,indexing='ij')
    k2=sum(ki**2 for ki in K)
    inv=np.zeros_like(k2); inv[k2>0]=1/k2[k2>0]
    d=np.fft.fftn(delta-delta.mean())
    return np.stack([np.fft.ifftn(-1j*ki*inv*d).real for ki in K],axis=-1)


def initial_particles(n,seed,amplitude=0.035):
    """Perturbed lattice using synthetic spectrum; NOT Boltzmann-derived ICs."""
    grid=np.stack(np.meshgrid(*[np.arange(n)/n+0.5/n]*3,indexing='ij'),-1).reshape(-1,3)
    field=gaussian_field(n,seed,0.025,0.965)
    grad=poisson_gradient(field)
    displacement=-sample_vector(grad,grid)* (amplitude/0.025)
    return (grid+displacement)%1, displacement


def pm_evolve(n=16,steps=100,seed=42,a_start=0.08,a_end=1.0,amplitude=0.035,omega_m=0.315,omega_lambda=0.684908,omega_r=0.000092):
    if not (8<=n<=48 and steps>=4 and steps<=2000 and 0<a_start<a_end<=1 and 0<amplitude<=0.3):
        raise ValueError('unsafe parameters: n 8..48; steps 4..2000; 0<a_start<a_end<=1; amplitude 0..0.3')
    if abs(omega_m+omega_lambda+omega_r-1)>1e-6: raise ValueError('requires flat background')
    p,u=initial_particles(n,seed,amplitude)
    s0,s1=np.log(a_start),np.log(a_end); h=(s1-s0)/steps
    history=[]
    def force(s,positions):
        a=np.exp(s)
        delta=cic_density(positions,n)-1
        om=omega_m/a**3/(omega_r/a**4+omega_m/a**3+omega_lambda)
        return -1.5*om*sample_vector(poisson_gradient(delta),positions),delta
    for i in range(steps):
        s=s0+i*h; a=np.exp(s)
        e2a=omega_r/a**4+omega_m/a**3+omega_lambda
        drag=2+(-4*omega_r/a**4-3*omega_m/a**3)/(2*e2a)
        acc,delta=force(s,p)
        # Kick-drift-kick, using local cosmological drag approx (small h).
        uhalf=u+0.5*h*(acc-drag*u)
        p=(p+h*uhalf)%1
        an=np.exp(s+h); e2n=omega_r/an**4+omega_m/an**3+omega_lambda
        dragn=2+(-4*omega_r/an**4-3*omega_m/an**3)/(2*e2n)
        acc2,delta=force(s+h,p)
        u=uhalf+0.5*h*(acc2-dragn*uhalf)
        if i % max(1,steps//10)==0 or i==steps-1:
            history.append({'a':float(an),'rms':float(np.std(delta)),'max_density':float(np.max(delta+1))})
        if not np.isfinite(u).all() or not np.isfinite(p).all(): raise FloatingPointError('unstable N-body integration')
    return p,u,cic_density(p,n),history


def halo_candidates(density,threshold=2.5):
    """Local peaks in CIC density. NOT gravitationally bound halo finder."""
    mask=density>=threshold
    for ax in range(3):
        mask &= density>=np.roll(density,1,axis=ax)
        mask &= density>=np.roll(density,-1,axis=ax)
    coords=np.argwhere(mask)
    return sorted([{'cell':[int(x) for x in c],'overdensity':float(density[tuple(c)]-1)} for c in coords],key=lambda v:-v['overdensity'])


def galaxy_population(halos,seed=42):
    """Phenomenological population tied to resolved density peaks, not hydrodynamics."""
    rng=np.random.default_rng(seed)
    out=[]
    for i,h in enumerate(halos):
        strength=h['overdensity']; gas= max(0.,strength*2e8)
        efficiency=float(np.clip(0.02+0.025*np.log1p(strength),0.01,0.14))
        stellar=gas*efficiency; metals=float(np.clip(0.0003+0.0008*np.log1p(stellar/1e7),0,0.02))
        star_count=max(0,int(stellar/0.55))
        # A *synthetic aggregate* population; not individual stars.
        systems=int(rng.poisson(min(20,stellar/5e7)))
        out.append({'id':i,'cell':h['cell'],'halo_proxy':strength,'gas_mass_proxy_solar':gas,'stellar_mass_proxy_solar':stellar,'metallicity_proxy':metals,'estimated_star_count':star_count,'sampled_star_systems':systems})
    return out


def stellar_and_planetary_catalog(galaxies,seed=43):
    """Synthetic individual sample systems. Not ab-initio disk/planet formation."""
    rng=np.random.default_rng(seed); systems=[]
    for g in galaxies:
        for j in range(g['sampled_star_systems']):
            # Illustrative IMF proxy, bounded 0.08..30 solar masses
            mass=float(np.clip(rng.lognormal(np.log(0.5),0.8),0.08,30))
            lifetime=float(10*mass**-2.5) # coarse Gyr main-sequence scaling
            k=int(rng.poisson(2.0 if g['metallicity_proxy']>.001 else .7))
            planets=[]
            for t in range(k):
                radius=float(10**rng.uniform(-1,1.7)) # Earth radii, illustrative
                orbit=float(10**rng.uniform(-1.2,1.6)) # AU
                eqtemp=float(278*mass**0.85 / np.sqrt(orbit))
                planets.append({'radius_earth_proxy':radius,'semi_major_axis_AU':orbit,'equilibrium_temperature_K_proxy':eqtemp,'temperate_candidate':bool(230<eqtemp<340 and radius<2.5)})
            systems.append({'galaxy_id':g['id'],'star_mass_solar':mass,'main_sequence_lifetime_Gyr_proxy':lifetime,'planets':planets})
    return systems


def evolutionary_ecology(generations=200,seed=44,founders=100,mutation_probability=0.015):
    """Artificial life after EXPLICIT SEEDING: selection, heredity, mutation and resources.

    This does not implement abiogenesis, genes, metabolism or actual organisms.
    Trait in [0,1] adjusts energy acquisition vs metabolic cost.
    """
    if not 1<=generations<=100000 or not 2<=founders<=10000 or not 0<=mutation_probability<=1: raise ValueError('invalid ecology configuration')
    rng=np.random.default_rng(seed); traits=rng.uniform(.05,.95,size=founders); records=[]
    for gen in range(generations):
        resource=1/(1+len(traits)/350)
        fitness=(0.25+1.3*traits*resource-0.48*traits**2)
        weights=fitness/fitness.sum()
        count=int(np.clip(rng.poisson(len(traits)*(0.91+0.15*resource)),2,1000))
        kids=traits[rng.choice(len(traits),size=count,p=weights)].copy()
        mutated=rng.random(count)<mutation_probability
        kids[mutated]+=rng.normal(0,.08,mutated.sum())
        traits=np.clip(kids,0,1)
        if gen%max(1,generations//20)==0 or gen==generations-1:
            records.append({'generation':gen,'population':int(len(traits)),'mean_trait':float(traits.mean()),'trait_variance':float(traits.var())})
    return records
