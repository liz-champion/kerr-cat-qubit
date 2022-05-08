import numpy as np

E = (np.identity(4) - 1.0j * np.flipud(np.identity(4)))# / np.sqrt(2.)

CNOT = np.zeros((4, 4)).astype(np.complex128)
CNOT[0,0] = 1.
CNOT[1,1] = 1.
CNOT[2,3] = 1.
CNOT[3,2] = 1.

print(CNOT @ np.conj(E))

import qutip as qt

print()
print(qt.tensor(qt.sigmax(), qt.identity(2)))
