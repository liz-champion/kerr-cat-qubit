import numpy as np
import qutip as qt

MHz = 1. # work in units of MHz
pi = np.pi


class KCQ:

    def __init__(self,
            N = 15,                             # Dimensionality of the Hilbert space, per resonator
            nqubits = 1,                        # Number of cat qubits (i.e. resonators)
            K = 2. * pi * 6.7 * MHz,            # Kerr nonlinearity
            e_2 = 2. * pi * 17.5 * MHz,         # Squeezing drive strength
            tau = 0.32 * MHz,                   # Squeezing drive ramp time
            e_x = 2. * pi * 6. * MHz,           # X-rotation drive strength
            e_x_rabi = 2. * pi * 0.74 * MHz,    # Rabi oscillation drive strength
            n_th = 0.04,                        # Population of n=1 Fock state in initial thermal state
            chi = 2. * pi * 1. * MHz,           # Coupling strength
            T_1 = 15.5 / MHz):                  # Single-photon decay time

        self.N = N
        self.nqubits = nqubits
        self.K = K
        self.e_2 = e_2
        self.tau = tau
        self.e_x = e_x
        self.e_x_rabi = e_x_rabi
        self.n_th = n_th
        self.chi = chi
        self.T_1 = T_1
        self.kappa = 1. / T_1

        self.t0 = 0.
        self.alpha = np.sqrt(self.e_2 / self.K)
        
        self.C_0 = (qt.coherent(self.N, self.alpha) + qt.coherent(self.N, -self.alpha)) / np.sqrt(2. * (1. + np.exp(-2. * np.abs(self.alpha)**2)))
        self.C_1 = (qt.coherent(self.N, self.alpha) - qt.coherent(self.N, -self.alpha)) / np.sqrt(2. * (1. - np.exp(-2. * np.abs(self.alpha)**2)))

        mag0 = np.real((self.C_0.dag() * self.C_0).full().item())
        mag1 = np.real((self.C_1.dag() * self.C_1).full().item())
        # first check the normalization to get an idea of whether or not we have a high enough dimensionality
        if mag0 < 0.995 or mag1 < 0.995:
            print("Warning: Hilbert space dimensionality is likely too small for the given system parameters, consider increasing it")

        # ... then just ensure that it's exactly normalized
        self.C_0 /= np.sqrt(mag0)
        self.C_1 /= np.sqrt(mag1)

        self.H = []
        for i in range(self.nqubits):
            self.kerr(i)
            self.two_photon(i)
        
        self.t0 += 1.1 * self.tau # for state preparation


    def kerr(self, qubit):
        H_list = [qt.identity(self.N)] * self.nqubits
        H_list[qubit] = -self.K * qt.create(self.N)**2 * qt.destroy(self.N)**2
        self.H.append(qt.tensor(H_list))

    def two_photon(self, qubit):
        H_list = [qt.identity(self.N)] * self.nqubits
        H_list[qubit] = self.e_2 * qt.create(self.N)**2 + np.conj(self.e_2) * qt.destroy(self.N)**2
        const = 0.5 * np.tanh(-2.) + 0.5
        s = "(0.5 * tanh(4. * (t - {0}) / {1} - 2.) + 0.5 - {2}) / (1. - {2})".format(self.t0, self.tau, const)
        self.H.append([qt.tensor(H_list), s])

    def add_identity_gate(self, duration):
        self.t0 += duration

    def add_X_gate(self, theta, qubit):
        omega = np.real(4. * self.e_x * self.alpha)
        duration = theta / omega
        H_list = [qt.identity(self.N)] * self.nqubits
        H_list[qubit] = self.e_x * qt.create(self.N) + np.conj(self.e_x) * qt.destroy(self.N)
        self.H.append([qt.tensor(H_list), self.pulse(self.t0, duration)])
        self.t0 += 1.1 * duration

    def add_Z_gate(self, qubit):
        duration = pi / (2. * self.K)
        H_list = [qt.identity(self.N)] * self.nqubits
        H_list[qubit] = -self.e_2 * qt.create(self.N)**2 - np.conj(self.e_2) * qt.destroy(self.N)**2 - self.K * qt.create(self.N) * qt.destroy(self.N) #+ self.K * qt.create(self.N)**2 * qt.destroy(self.N)**2 - self.K * (qt.create(self.N) * qt.destroy(self.N))**2
        self.H.append([qt.tensor(H_list), self.pulse(self.t0, duration)])
        self.t0 += 1.1 * duration

    def add_entangling_gate(self):
        omega = 2. * self.e_2 * self.chi / self.K
        duration = 0.25 * pi / omega
        self.H.append([self.chi * (qt.tensor(qt.create(self.N), qt.destroy(self.N)) + qt.tensor(qt.destroy(self.N), qt.create(self.N))), self.pulse(self.t0, duration)])
        self.t0 += 1.1 * duration

    def add_hadamard_gate(self, qubit):
        self.add_X_gate(pi/2., qubit)
        self.add_Z_gate(qubit)

    def add_rabi_oscillation(self, qubit):
        H_list = [qt.identity(self.N)] * self.nqubits
        H_list[qubit] = self.e_x_rabi * qt.create(self.N) + np.conj(self.e_x_rabi) * qt.destroy(self.N)
        self.H.append([qt.tensor(H_list), self.pulse(self.t0, 100.)])

    def pulse(self, t0, duration, tau=None):
        if tau is None:
            tau =1.0e-5 * duration
        return "0.5 * tanh((t - {0}) / {1}) - 0.5 * tanh((t - {0} - {2}) / {1})".format(t0, tau, duration)

    def loss_terms(self):
        ret = []
        for i in range(self.nqubits):
            H_list_create = [qt.identity(self.N)] * self.nqubits
            H_list_destroy = [qt.identity(self.N)] * self.nqubits
            H_list_create[i] = qt.create(self.N)
            H_list_destroy[i] = qt.destroy(self.N)
            ret.append(np.sqrt(self.kappa * self.n_th) * qt.tensor(H_list_create))
            ret.append(np.sqrt(self.kappa * (1. + self.n_th)) * qt.tensor(H_list_destroy))
        return ret

    def cat_states(self):
        return [self.C_0, self.C_1]

    def basis_states(self):
        return [qt.tensor(self.C_0, self.C_0), qt.tensor(self.C_0, self.C_1), qt.tensor(self.C_1, self.C_0), qt.tensor(self.C_1, self.C_1)]

    def bell_states(self):
        [C_00, C_01, C_10, C_11] = self.basis_states()
        phi_plus = (C_00 + C_11) / np.sqrt(2.)
        phi_minus = (C_00 - C_11) / np.sqrt(2.)
        psi_plus = (C_01 + C_10) / np.sqrt(2.)
        psi_minus = (C_01 - C_10) / np.sqrt(2.)
        return [phi_plus, phi_minus, psi_plus, psi_minus]

    def thermal_state(self):
        rho_single = (1. - self.n_th) * qt.basis(self.N, 0) * qt.basis(self.N, 0).dag() + self.n_th * qt.basis(self.N, 1) * qt.basis(self.N, 1).dag()
        return qt.tensor([rho_single] * self.nqubits)

    def bloch_states(self):
        x_plus = (self.C_0 + self.C_1) / np.sqrt(2.)
        x_minus = (self.C_0 - self.C_1) / np.sqrt(2.)
        y_plus = (self.C_0 + 1.0j * self.C_1) / np.sqrt(2.)
        y_minus = (self.C_0 - 1.0j * self.C_1) / np.sqrt(2.)
        z_plus = self.C_0
        z_minus = self.C_1
        return [x_plus, x_minus, y_plus, y_minus, z_plus, z_minus]

    def set_N(self, N):
        self.N = N
        self.C_0 = (qt.coherent(self.N, self.alpha) + qt.coherent(self.N, -self.alpha)) / np.sqrt(2. * (1. + np.exp(-2. * np.abs(self.alpha)**2)))
        self.C_1 = (qt.coherent(self.N, self.alpha) - qt.coherent(self.N, -self.alpha)) / np.sqrt(2. * (1. - np.exp(-2. * np.abs(self.alpha)**2)))
