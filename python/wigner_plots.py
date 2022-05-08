import numpy as np
import matplotlib.pyplot as plt

import qutip as qt

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(6, 9), sharex=True, sharey=True)
axes[0][0].set_title("$| 0_L \\rangle $", fontsize=14)
axes[0][1].set_title("$| 1_L \\rangle $", fontsize=14)
axes[0][0].set_ylabel("Cat", fontsize=14)
axes[1][0].set_ylabel("Binomial", fontsize=14)
axes[2][0].set_ylabel("GKP", fontsize=14)

for ax in axes.flatten():
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

N = 200
alpha = 2.6

C0 = (qt.coherent(N, alpha) + qt.coherent(N, -alpha)) / np.sqrt(2.)
C1 = (qt.coherent(N, alpha) - qt.coherent(N, -alpha)) / np.sqrt(2.)

B0 = (qt.basis(N, 0) + qt.basis(N, 4)) / np.sqrt(2.)
B1 = qt.basis(N, 2)

delta = 1.
delta_tilde = .25
psi0 = qt.squeeze(N, delta) * qt.basis(N, 0)
GKP0 = sum([np.exp(-2. * np.pi * delta_tilde**2 * s**2) * qt.displace(N, s * np.sqrt(2. * np.pi)) * psi0 for s in range(-20, 20)])
GKP1 = sum([np.exp(-np.pi * delta_tilde**2 * (2. * s + 1)**2 / 2.) * qt.displace(N, s * np.sqrt(2. * np.pi)) * qt.displace(N, np.sqrt(np.pi / 2.)) * psi0 for s in range(-20, 20)])

x = np.linspace(-6., 6., 100)

for ax, psi in zip(axes.flatten(), [C0, C1, B0, B1, GKP0, GKP1]):
    w = qt.wigner(psi, x[1:], x[1:])
    w /= np.max(np.abs(w))
    im = ax.pcolormesh(x, x, w, cmap="bwr", vmin=-1., vmax=1.)
    ax.set_aspect("equal")


plt.tight_layout()
plt.savefig("../paper/figures/cat_binom_gkp.pdf")
