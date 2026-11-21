import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
#parametros

m = 50.0                                 
I = np.diag([15.95, 13.56, 3.99])        
A1 = 0.10                                 
A = np.array([A1, 0.12, 0.875])           
g = 9.81 

#condiciones iniciales
estado0 = [0.0, 0.01, 1.25, 0.5]
t_final = 30.0

E1 = np.array([1.0, 0.0, 0.0])

#geometria
def vectores(theta):
    Gamma = np.array([0.0, np.sin(theta), np.cos(theta)])
    alpha = np.array([0.0, -np.cos(theta), np.sin(theta)])
    return Gamma, alpha

#ecs. de movimiento

def residuo(x, theta, O1, O2, v):
    dO1, dO2, dv, kappa, lam, mu = x
    Gamma, alpha = vectores(theta)

    Omega = O1 * E1 + O2 * Gamma
    Y = v * E1
    P = m * (Y + np.cross(Omega, A))
    Pi = I @ Omega + np.cross(A, P)
    
    dGamma = -O1 * alpha
    dOmega = dO1 * E1 + dO2 * Gamma + O2 * dGamma
    dY = dv * E1
    dP = m * (dY + np.cross(dOmega, A))
    dPi = I @ dOmega + np.cross(A, dP)

    angular = dPi + np.cross(Omega, Pi) - m * g * np.cross(Gamma, A) + np.cross(Y, P) - kappa * alpha
    lineal = dP + np.cross(Omega, P) - lam * Gamma - mu * alpha
    return np.concatenate([angular, lineal])


def aceleraciones(theta, O1, O2, v):
    
    r0 = residuo(np.zeros(6), theta, O1, O2, v)
    M = np.column_stack([residuo(np.eye(6)[j], theta, O1, O2, v) - r0 for j in range(6)])
    x = np.linalg.solve(M, -r0)
    return x[:3]


def ec_mov(t, estado):
    dO1, dO2, dv = aceleraciones(theta, O1, O2, v)
    return [O1, dO1, dO2, dv]           

def energia(theta, O1, O2, v):
   
    Gamma, _ = vectores(theta)
    Omega = O1 * E1 + O2 * Gamma
    v_cm = v * E1 + np.cross(Omega, A)
    return 0.5 * Omega @ I @ Omega + 0.5 * m * v_cm @ v_cm + m * g * A @ Gamma

#constantes de movimiento del caso integrable

I1, I2, I3 = np.diag(I)
A2, A3 = A[1], A[2]


def J1(theta, O2):
    return O2 * (I2 * np.sin(theta) ** 2 + I3 * np.cos(theta) ** 2)

def beta1(theta):
     D = I2 - I3
     return (A3 / np.sqrt(I3 * D) * np.arctan(np.sqrt(D / I3) * np.sin(theta))
            - A2 / np.sqrt(I2 * D) * np.arctanh(np.sqrt(D / I2) * np.cos(theta)))



def J2(theta, O2, v):
     phi = -A2 * np.cos(theta) + A3 * np.sin(theta)          
     return v + O2 * phi + J1(theta, O2) * beta1(theta)

#integracion
t = np.linspace(0, t_final, 6001)
sol = solve_ivp(ec_mov, (0, t_final), estado0, t_eval=t,
                method="DOP853", rtol=1e-10, atol=1e-12)
theta, O1, O2, v = sol.y

E = np.array([energia(*s) for s in sol.y.T])
dE = E - E[0]
dJ1 = J1(theta, O2) - J1(theta[0], O2[0])
dJ2 = J2(theta, O2, v) - J2(theta[0], O2[0], v[0])
print(f"A1 = {A1} m")
print(f"  max |E  - E(0)|  = {np.abs(dE).max():.1e}")
print(f"  max |J1 - J1(0)| = {np.abs(dJ1).max():.1e}")
print(f"  max |J2 - J2(0)| = {np.abs(dJ2).max():.1e}")


#plot
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(t, dE, color="black", lw=1.4, label=r"$E(t) - E(0)$")
ax.plot(t, dJ1, color="blue", lw=1.2, label=r"$J_1(t) - J_1(0)$")
ax.plot(t, dJ2, color="red", lw=1.2, label=r"$J_2(t) - J_2(0)$")
ax.set_xlabel("$t$ (s)")
ax.set_title(rf"Variaciones de $E$, $J_1$ y $J_2$  ($A_1$ = {A1*100:g} cm)")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(f"fig_variacion{A1*100:g}cm.png", dpi=180)
plt.show()