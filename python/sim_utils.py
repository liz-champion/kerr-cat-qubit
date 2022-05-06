import numpy as np
import qutip as qt

###
### Generates the Kerr-nonlinear part of the Hamiltonian with Kerr nonlinearity
### K, a Hilbert space of dimension N *for each resonator*, and nres resonators.
###
def H_kerr(K, N, nres=1, res_index=0):
    H = -K * qt.create(N)**2 * qt.destroy(N)**2
    H_list = [qt.identity(N)] * nres
    H_list[res_index] = H
    return qt.tensor(H_list)

###
### The two-photon drive for a single resonator
###
def H_2ph(e_2, N, tau=None, t0=None, direction=1, nres=1, res_index=0):
    H = e_2 * qt.create(N)**2 + np.conj(e_2) * qt.destroy(N)**2
    H_list = [qt.identity(N)] * nres
    H_list[res_index] = H
    if tau is not None and t0 is not None:
        strength_factor = "0.5 * tanh(4. * ({0} * t - {1}) / {2} - 2.) + 0.5".format(direction, t0, tau)
    else:
        strength_factor = "1."
    return [qt.tensor(H_list), strength_factor]

###
### The coupling Hamiltonian. Note that the number of resonators is not an
### argument because it only makes sense to do this with two resonators.
###
def H_c(g_12, N, tau=None, t0=None, direction=1):
    if tau is not None and t0 is not None:
        strength_factor = "0.5 * tanh(4. * ({0} * t - {1}) / {2} - 2.) + 0.5".format(direction, t0, tau)
    else:
        strength_factor = "1."
    return [g_12 * (qt.tensor(qt.create(N), qt.destroy(N)) + qt.tensor(qt.destroy(N), qt.create(N))), strength_factor]

###
### A function to generate our basis (cat) states for us
###
def cat_states(K, e, N):
    alpha = np.sqrt(e / K)
    psi_0 = (qt.coherent(N, alpha) + qt.coherent(N, -alpha)) / np.sqrt(2. * (1 + np.exp(-2. * np.abs(alpha)**2)))
    psi_1 = (qt.coherent(N, alpha) - qt.coherent(N, -alpha)) / np.sqrt(2. * (1 - np.exp(-2. * np.abs(alpha)**2)))
    return psi_0, psi_1

###
### Generates the basis states specifically for two resonators. The ordering
### is 00, 01, 10, 11.
###
def basis(K, e_2, N):
    psi_0, psi_1 = cat_states(K, e_2, N)
    return [qt.tensor(psi_0, psi_0), qt.tensor(psi_0, psi_1), qt.tensor(psi_1, psi_0), qt.tensor(psi_1, psi_1)]

###
### Generates the Bell states
###
def bell_states(K, e_2, N):
    [C_00, C_01, C_10, C_11] = basis(K, e_2, N)
    phi_plus = (C_00 + 1.j * C_11) / np.sqrt(2.)
    phi_minus = (C_00 - 1.j * C_11) / np.sqrt(2.)
    psi_plus = (C_01 + 1.j * C_10) / np.sqrt(2.)
    psi_minus = (C_01 - 1.j * C_10) / np.sqrt(2.)
    return [phi_plus, phi_minus, psi_plus, psi_minus]


###
### Hamiltonian for the entangling gate, which has the form sigmax ⊗ sigmax
###
def H_entangle(K, e_2, g_12, N, t0):
    omega = 2. * e_2 * g_12 / K
    duration = 0.25 * np.pi / omega
    tau = 0.0001 * duration
    strength_factor = "0.5 * tanh(4. * (t - {0}) / {1} - 2.) - 0.5 * tanh(4. * (t - {0} - {2}) / {1} - 2.)".format(t0, tau, duration)
    #strength_factor = "0.5 * tanh(4. * (t - {0}) / {1} - 2.) + 0.5".format(t0, tau)
    return [g_12 * (qt.tensor(qt.create(N), qt.destroy(N)) + qt.tensor(qt.destroy(N), qt.create(N))), strength_factor]

###
### Hamiltonian for the X(theta) gate
###
def H_x(K, e_2, e_x, N, t0, theta, nres=1, res_index=0):
    omega = np.real(4. * e_x * np.sqrt(e_2 / K))
    duration = theta / omega
    tau = 0.0001 * duration
    strength_factor = "0.5 * tanh(4. * (t - {0}) / {1} - 2.) - 0.5 * tanh(4. * (t - {0} - {2}) / {1} - 2.)".format(t0, tau, duration)
    H_list = [qt.identity(N)] * nres
    H_list[res_index] = e_x * qt.create(N) + np.conj(e_x) * qt.destroy(N)
    return [qt.tensor(H_list), strength_factor]

def H_z(K, e_2, N, t0, theta, nres=1, res_index=0):
    duration = 0.5 * theta / K
    tau = 0.0001 * duration
    strength_factor = "-0.5 * tanh(4. * (t - {0}) / {1} - 2.) + 0.5 * tanh(4. * (t - {0} - {2}) / {1} - 2.)".format(t0, tau, duration)
    H_list = [qt.identity(N)] * nres
    H_list[res_index] = e_2 * qt.create(N)**2 + np.conj(e_2) * qt.destroy(N)**2
    return [qt.tensor(H_list), strength_factor]

def gate_durations(K, e_2, e_x, g):
    t_entangle = 0.125 * np.pi * K / (e_2 * g)
    t_H_x_half = 0.5 * np.pi / np.real(4. * e_x * np.sqrt(e_2 / K))
    t_H_x_full = np.pi / np.real(4. * e_x * np.sqrt(e_2 / K))
    t_H_z_half = 0.5 * np.pi / K
    return t_entangle, t_H_x_half, t_H_x_full, t_H_z_half
