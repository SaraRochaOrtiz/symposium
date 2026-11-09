import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#parámetros (dados por el artículo)

m = 50.0
I = np.diag([15.95, 13.56, 3.99])
A = np.array([0.0, 0.12, 0.875])
g = 9.81

#condiciones inciales
estado0 = [0.0, 0.01, 1.25, 0.5]
t_final= 20.0

E1= np.array([1.0, 0.0, 0.0])

#geometría del marco de la cuchilla

def vectores(theta):
    Gamma = np.array([0.0, np.sin(theta), np.cos(theta)])
    alpha = np.array([0.0, -np.cos(theta), np.sin(theta)])
    return Gamma, alpha

#ec. de movimiento
def residuo(x, theta, O1, O2, v):
    dO1, dO2, dv, kappa, lam, mu = x
    Gamma, alpha = vectores(theta)

    Omega = O1 * E1 + O2 * Gamma
    Y = v *E1
    P = m *(Y+np.cross(Omega, A))
    Pi = I @ Omega + np.cross(A, P)

    #derivadas temporales
    dGamma = -O1 * alpha
    dOmega = dO1 * E1 + dO2 * Gamma + O2 * dGamma
    dY = dv *E1 
    dP = m * (dY + np.cross(dOmega, A))
    dPi = I @ dOmega + np.cross(A, dP)

    angular = dPi + np.cross(Omega, Pi) - m*g*np.cross(Gamma, A) + np.cross(Y, P) - kappa * alpha
    lineal = dP + np.cross(Omega, P) - lam *Gamma - mu *alpha
    return np.concatenate([angular, lineal])

def aceleraciones(theta, O1, O2, v):
    r0 = residuo(np.zeros(6), theta, O1, O2, v)
    M = np.column_stack([residuo(np.eye(6)[j],theta, O1, O2, v) - r0 for j in range (6)])
    x = np.linalg.solve(M, -r0)
    return x[:3]

def ec_movimiento(t, estado):
    theta, O1, O2, v = estado
    dO1, dO2, dv = aceleraciones(theta, O1, O2, v)
    return [O1, dO1, dO2, dv]

def energia(theta, O1, O2, v): 
    Gamma, _ = vectores(theta)
    Omega = O1 * E1 + O2 * Gamma
    v_cm = v * E1 + np.cross(Omega, A)
    return 0.5 * Omega @ I @ Omega + 0.5 * m * v_cm @ v_cm + m * g * A @ Gamma

#integracion

t = np.linspace(0, t_final, 4001)
sol= solve_ivp(ec_movimiento,(0, t_final), estado0, t_eval=t, method="DOP853", rtol=1e-10, atol=1e-12)
theta, O1, O2, v = sol.y

#comprobacion (la energia debe conservarse)
E = np.array([energia(*s) for s in sol.y.T])
print(f"Variación máxima de la energía: {np.abs(E - E[0]).max():.1e} J  (E0 = {E[0]:.3f} J)")

#ceros de Omea y de v

def ceros(t, f):
    i = np.where(np.sign(f[:-1])* np.sign(f[1:]) < 0 )[0]
    return t[i] - f[i] * (t[i + 1] - t[i]) / (f[i + 1] - f[i])

t_O1 = ceros(t, O1)
t_v = ceros(t, v)

#plot 
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
 
ax1.plot(t, O1, color="black", lw=1.3)
ax1.plot(t_O1, np.zeros_like(t_O1), "*", color="tab:orange", ms=9, label=r"$\Omega_1 = 0$")
ax1.set_ylabel(r"$\Omega_1 = \dot\theta$  (1/s)")
ax1.set_title(r"$A_1 = 0$ (caso integrable)")
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(alpha=0.3)
 
ax2.plot(t, v, color="black", lw=1.3)
ax2.plot(t_v, np.zeros_like(t_v), "x", color="tab:blue", ms=8, mew=2, label="$v = 0$")
ax2.set_ylabel("$v$  (m/s)")
ax2.set_xlabel("$t$  (s)")
ax2.legend(loc="upper left", fontsize=9)
ax2.grid(alpha=0.3)
 
fig.tight_layout()
fig.savefig("fig_velangularcuchilla.png", dpi=180)
plt.show()




    
