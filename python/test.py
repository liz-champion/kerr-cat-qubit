import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

duration = 0.25 * np.pi

H = np.flipud(np.identity(4))

def f(t, y):
    return H @ y / 1.0j

y0 = np.zeros(4).astype(np.complex128)
y0[2] = 1.0
#y0[2] = np.sqrt(0.5)

res = solve_ivp(f, [0., duration], y0, t_eval=np.linspace(0., duration, 100))

plt.plot(res.t/np.pi, np.real(res.y[0]), color="black", label="00")
plt.plot(res.t/np.pi, np.imag(res.y[0]), "--", color="black")

plt.plot(res.t/np.pi, np.real(res.y[1]), color="red", label="01")
plt.plot(res.t/np.pi, np.imag(res.y[1]), "--", color="red")

plt.plot(res.t/np.pi, np.real(res.y[2]), color="blue", label="10")
plt.plot(res.t/np.pi, np.imag(res.y[2]), "--", color="blue")

plt.plot(res.t/np.pi, np.real(res.y[3]), color="green", label="11")
plt.plot(res.t/np.pi, np.imag(res.y[3]), "--", color="green")

plt.legend()

plt.show()
