import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1  # Bessel J1
from numpy.fft import fftshift, fft, fftfreq


wavelength0 = 550e-9
D = 0.01
f = 0.1
pixel_size = 1e-6
N = 2048


L = N * pixel_size
x = np.linspace(-L/2, L/2, N)


r1d = np.linspace(0, L/2, 2000)

# simple fraunhofer / airy disk functions from derivation of paper
def airy_psf_radial(r, wavelength):

    # Compute the Fraunhofer diffraction pattern (Airy disk) for a circular aperture. 3 Main Points:
    # 1) Single-slit diffraction: I(theta) ∝ (sin(beta)/beta)^2 -> beta = pi * slit_width * sin(theta) / lambda
    # 2) Circular aperture: I(theta) ∝ [2 * J1(k a sin(theta)) / (k a sin(theta))]^2
    # 3) Metamaterials: replace lambda with lambda_eff = lambda0 / |n| to shrink Airy disk
    k = 2*np.pi / wavelength
    a = D / 2
    sin_theta = r / f
    arg = k * a * sin_theta
    small = (arg == 0)
    out = np.zeros_like(arg)
    out[~small] = (2 * j1(arg[~small]) / arg[~small])**2
    out[small] = 1.0
    return out

n_list = [1.0, 0.8, 0.5, 0.2, -1.0]  # last entry is NIM
labels = ['n=1.0', 'n=0.8', 'n=0.5', 'n=0.2', 'n=-1.0']

# Plot radial point spread function (Note: used open source code for this - info from choglke - psf)
plt.figure(figsize=(8,6))
for n, lbl in zip(n_list, labels):
    lam_eff = wavelength0 / abs(n)  # Metamaterial: lambda_eff = lambda0 / |n|
    I = airy_psf_radial(r1d, lam_eff)
    I /= I.max()  # normalize peak to 1
    plt.plot(r1d*1e6, I, label=lbl)
plt.xlabel('radial coordinate in image plane (µm)')
plt.ylabel('Normalized intensity')
plt.title('Airy PSF radial profiles for various effective indices (|n|)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Compute full width at half maximum using the point spread function (Note:  used AI in construction of logic) ---
def find_fwhm(r, I):
    half = 0.5 * np.max(I)
    idx = np.where(I < half)[0]
    if len(idx) == 0:
        return np.nan
    r_half = r[idx[0]]
    return 2 * r_half  # appx. full width

fwhm_list = []
for n in n_list:
    lam_eff = wavelength0 / abs(n)
    I = airy_psf_radial(r1d, lam_eff)
    I /= I.max()
    fwhm = find_fwhm(r1d, I)
    fwhm_list.append(fwhm)

print("n_list:", n_list)
print("FWHM (microns):", [f*1e6 for f in fwhm_list])

# building the point spread function grid (centered) to be graphed on matplotlib
def airy_psf_2d(wavelength):
    """
    Note: I am using a Fraunhofer point spread function here for airy disk of circular aperture
    """
    X = np.linspace(-L/2, L/2, N)
    Y = X.copy()
    XX, YY = np.meshgrid(X, Y)
    R = np.sqrt(XX**2 + YY**2)
    I2d = airy_psf_radial(R, wavelength)
    return I2d / I2d.max()

# Modulation transfer function --> measure detail degradation in NIm compared to different indices and size of airy disk
plt.figure(figsize=(8,4))
for n, lbl in zip(n_list, labels):
    lam_eff = wavelength0 / abs(n)
    psf2 = airy_psf_2d(lam_eff)
    # MTF via Fourier transform of PSF
    OTF = fftshift(fft(fft(psf2, axis=0), axis=1))
    MTF = np.abs(OTF)
    MTF /= MTF.max()
    # radial average of MTF along central row
    center = N//2
    mtf_line = MTF[center, center:]
    freqs = fftshift(fftfreq(N, d=pixel_size))
    freq_line = freqs[center:]
    plt.plot(freq_line*1e-3, mtf_line, label=lbl)  # cycles/mm

plt.xlim(0, 1.5)
plt.xlabel('spatial frequency (cycles/mm)')
plt.ylabel('MTF (normalized)')
plt.title('MTF (central cut) for different effective indices')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
