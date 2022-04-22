import numpy as np
import qutip as qt

###
### Generates the Kerr-nonlinear part of the Hamiltonian with Kerr nonlinearity
### K, a Hilbert space of dimension N *for each resonator*, and nres resonators.
###
def H_kerr(K, N, nres=1):
    H = -K * qt.create(N)**2 * qt.destroy(N)**2
    return qt.tensor([H] * nres)

###
### The two-photon drive for a single resonator
###
def H_2ph(e_2, N, strength_factor=(lambda t, _: 1.), nres=1, res_index=0):
    H = e_2 * qt.create(N)**2 + np.conj(e_2) * qt.destroy(N)**2
    H_list = [qt.identity(N)] * nres
    H_list[res_index] = H
    return [qt.tensor(H_list), strength_factor]

###
### The four-photon drive for a single resonator
###
def H_4ph(e_2, N, strength_factor=(lambda t, _: 1.), nres=1, res_index=0):
    H = e_2 * qt.create(N)**4 + np.conj(e_2) * qt.destroy(N)**4
    H_list = [qt.identity(N)] * nres
    H_list[res_index] = H
    return [qt.tensor(H_list), strength_factor]

###
### The coupling Hamiltonian. Note that the number of resonators is not an
### argument because it only makes sense to do this with two resonators.
###
def H_c(g, N, strength_factor=(lambda t, _: 1.)):
    return [g * (tensor(qt.create(N), qt.destroy(N)) + qt.tensor(qt.destroy(N), qt.create(N))), strength_factor]

###
### A function to generate our basis (cat) states for us
###
def cat_states(K, e_2, N):
    alpha = np.sqrt(e_2 / K)
    psi_0 = (qt.coherent(N, alpha) + qt.coherent(N, -alpha)) / np.sqrt(2. * (1 + np.exp(-2. * np.abs(alpha)**2)))
    psi_1 = (qt.coherent(N, alpha) - qt.coherent(N, -alpha)) / np.sqrt(2. * (1 - np.exp(-2. * np.abs(alpha)**2)))
    return psi_0, psi_1

###
### Generates the basis states specifically for two resonators. The ordering
### is 00, 01, 10, 11.
###
def basis(K, e_2, N):
    psi_0, psi_1 = cat_states(K, e_2, N)
    return qt.tensor(psi_0, psi_0), qt.tensory(psi_0, psi_1), qt.tensor(psi_1, psi_0), qt.tensor(psi_1, psi_1)

###
### Compute the probability that one resonator is in the given state
###
def calculate_prob(rho, psi, N, nres=1, res_index=0):
    if nres > 1:
        rho = rho.ptrace(res_index)
    return (psi.dag() * rho.full() * psi).flatten().real

###
### Hyperbolic tangent ramp function, for convenience
###
def tanh_ramp(tau, t0, direction=1):
    return lambda t, _: 0.5 * np.tanh(4. * (direction * t - t0) / tau - 2.) + 0.5
