"""Flat FLRW reference background and scale-independent linear-growth ODE.

Not a Boltzmann solver, early-universe solver, or nonlinear gravity solver.
H(a)^2 / H0^2 = Omega_r/a^4 + Omega_m/a^3 + Omega_Lambda.
D''(ln a) + [2 + dlnH/dln a] D'(ln a) - 3 Omega_m(a)/2 D = 0.
The latter assumes pressureless matter, smooth dark energy, and no scale
 dependence; thus illustrative for sufficiently late linear matter growth.
"""
from math import sqrt
import numpy as np

MPC_KM = 3.0856775814913673e19
SECONDS_PER_GYR = 365.25 * 86400 * 1e9


def validate_cosmology(c):
    for k in ('H0_km_s_Mpc','Omega_m','Omega_r','Omega_Lambda','a_start','a_end','steps'):
        if k not in c: raise ValueError(f'missing cosmology.{k}')
    if c['H0_km_s_Mpc'] <= 0 or c['Omega_m'] <= 0 or c['Omega_r'] < 0 or c['Omega_Lambda'] < 0:
        raise ValueError('invalid cosmological parameters')
    if abs(c['Omega_m']+c['Omega_r']+c['Omega_Lambda']-1) > 1e-7:
        raise ValueError('starter requires flat cosmology with sum Omega = 1')
    if not (0 < c['a_start'] < c['a_end'] <= 1):
        raise ValueError('require 0 < a_start < a_end <= 1')
    if int(c['steps']) < 8 or int(c['steps']) > 200000:
        raise ValueError('steps outside supported limits')


def e2(a,c):
    return c['Omega_r']/a**4 + c['Omega_m']/a**3 + c['Omega_Lambda']


def hubble(a,c):
    return c['H0_km_s_Mpc']*sqrt(e2(a,c))


def dlnh_dlna(a,c):
    return (-4*c['Omega_r']/a**4-3*c['Omega_m']/a**3)/(2*e2(a,c))


def dydx(x,y,c):
    a=np.exp(x)
    D, V, age = y
    omega_m_a = c['Omega_m']/a**3 / e2(a,c)
    h_s = hubble(a,c)/MPC_KM
    return np.array([V,-(2+dlnh_dlna(a,c))*V+1.5*omega_m_a*D, 1/(h_s*SECONDS_PER_GYR)])


def solve(c):
    validate_cosmology(c)
    x=np.linspace(np.log(c['a_start']),np.log(c['a_end']),int(c['steps'])+1)
    # D(a_start) = a_start, D' = a_start is an approximate growing mode;
    # inaccurate when radiation is dynamically significant.
    y=np.empty((len(x),3),dtype=float)
    y[0] = [c['a_start'],c['a_start'],0.]
    for i in range(len(x)-1):
        h=x[i+1]-x[i]; v=y[i]
        k1=dydx(x[i],v,c)
        k2=dydx(x[i]+h/2,v+h*k1/2,c)
        k3=dydx(x[i]+h/2,v+h*k2/2,c)
        k4=dydx(x[i]+h,v+h*k3,c)
        y[i+1]=v+h*(k1+2*k2+2*k3+k4)/6
        if not np.all(np.isfinite(y[i+1])): raise FloatingPointError('nonfinite integration result')
    a=np.exp(x)
    return dict(a=a, z=1/a-1, H=np.array([hubble(ai,c) for ai in a]),
                D=y[:,0]/y[-1,0], elapsed_Gyr=y[:,2])
