import json, hashlib, numpy as np, math

# Load metadata
with open('data/reference/planck_lcdm_powerspectrum_v1.json') as f:
    meta = json.load(f)

cosmo = meta['cosmology']
H0 = cosmo['H0_km_s_Mpc']
Om = cosmo['Omega_m']
Ob = cosmo['Omega_b']
ns = cosmo['n_s']
h = cosmo['h']

# Generate k range in h/Mpc
k = np.logspace(-4, 2, 2000)  # h/Mpc

# Analytic transfer function (Eisenstein & Hu 1998, simplified BBKS)
keq = 0.0746 * Om * h**2  # h/Mpc (approx matter-radiation equality scale)

q = k / keq

def T_bbks(q):
    return np.log(1.0 + 2.34*q)/(2.34*q) * (1.0 + 3.89*q + (16.1*q)**2 + (5.43*q)**3 + (6.72*q)**4)**(-0.25)

q_safe = np.where(q == 0, 1e-12, q)
T = T_bbks(q_safe)
T = np.where(q == 0, 1.0, T)

# Primordial power spectrum: P_prim(k) = A_s * (k/k_pivot)^(n_s-1)
k_pivot = 0.05  # h/Mpc
As = 2.1e-9
P_prim = As * (k / k_pivot)**(ns - 1)

# Matter power spectrum: P(k) = A * k^n_s * T^2
# Normalize roughly to match sigma8 ~ 0.8
P_k = P_prim * T**2 * (2960.0)**2  # rough normalization

# Save as npz
np.savez_compressed('data/reference/planck_lcdm_powerspectrum_v1.npz',
                    k=k, P_k=P_k,
                    meta_json=json.dumps(meta))

# Compute checksum of npz
with open('data/reference/planck_lcdm_powerspectrum_v1.npz', 'rb') as f:
    checksum = hashlib.sha256(f.read()).hexdigest()

meta['checksum_sha256'] = checksum
meta['data_file'] = 'planck_lcdm_powerspectrum_v1.npz'
meta['n_k_points'] = int(len(k))
meta['k_min_h_Mpc'] = float(k.min())
meta['k_max_h_Mpc'] = float(k.max())

with open('data/reference/planck_lcdm_powerspectrum_v1.json', 'w') as f:
    json.dump(meta, f, indent=2)

print('Checksum:', checksum)
print('k range:', k.min(), k.max())
print('P_k range:', P_k.min(), P_k.max())
