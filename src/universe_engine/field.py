"""Illustrative Gaussian 3D density field. NOT a calibrated ΛCDM transfer function.
Fixed Fourier realization is scaled by linear growth. No particles evolve.
"""
import numpy as np

def gaussian_field(n, seed, initial_rms, spectral_index):
    if not (8 <= n <= 256 and n & (n - 1) == 0):
        raise ValueError('grid_size must be power-of-two between 8 and 256')
    if not (0 < initial_rms < 0.1):
        raise ValueError('initial_rms must lie in (0, 0.1)')
    rng = np.random.default_rng(seed)
    white = rng.normal(size=(n,n,n))
    k = np.fft.fftfreq(n)*n
    KX,KY,KZ=np.meshgrid(k,k,k,indexing='ij')
    k2=KX*KX+KY*KY+KZ*KZ
    # Synthetic spectrum only; it omits baryon acoustic oscillations,
    # radiation transfer, normalization and scale-dependent growth.
    filt = np.zeros_like(k2,dtype=float)
    mask=k2>0
    filt[mask]=k2[mask]**((spectral_index-4)/4)
    sample = np.fft.ifftn(np.fft.fftn(white)*filt).real
    sample -= sample.mean()
    sample *= initial_rms/sample.std()
    return sample
