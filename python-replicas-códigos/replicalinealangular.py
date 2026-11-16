import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#parametros
m = 50.0                                  
I = np.diag([15.95, 13.56, 3.99])         
A = np.array([0.10, 0.12, 0.875])       
g = 9.81  

#condiciones del sistema 
estado_ref = [0.0, 0.01, 1.25, 0.5]
Omega2_iniciales = [1.25, 1.0, 1.5, 0.8]
t_inicio, t_final = 10.0, 80.0

E1 = np.array([1.0, 0.0, 0.0])
#geometria

def vectores(theta):
    Gamma = np.array([0.0, np.sin(theta), np.cos(theta)])
    alpha = np.array([0.0, -np.cos(theta), np.sin(theta)])
    return Gamma, alpha

#ec. mov
def residuo(x, theta, O1, O2, v):
    dO1, dO2, dv, kappa, lam, mu = x
    Gamma, alpha = vectores(theta)

    Omega = O1 * E1 + O2 * Gamma
    Y = v * E1
    P = m * (Y + np.cross(Omega, A))
    Pi = I @ Omega + np.cross(A, P)

#derivadas temporales}
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

def ec_movimiento(t, estado):
    theta, O1, O2, v = estado
    dO1, dO2, dv = aceleraciones(theta, O1, O2, v)
    return [O1, dO1, dO2, dv] 

def energia(theta, O1, O2, v):
    Gamma, _ = vectores(theta)
    Omega = O1 * E1 + O2 * Gamma
    v_cm = v * E1 + np.cross(Omega, A)
    return 0.5 * Omega @ I @ Omega + 0.5 * m * v_cm @ v_cm + m * g * A @ Gamma

def v_misma_energia(theta, O1, O2, E_objetivo):
    e0 = energia(theta, O1, O2, 0.0)
    e1 = energia(theta, O1, O2, 1.0)
    em = energia(theta, O1, O2, -1.0)
    a = (e1 + em) / 2 - e0
    b = (e1 - em) / 2
    c = e0 - E_objetivo
    return (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
#condiciones iniciales con la misma energia
E0 = energia(*estado_ref)
condiciones = []
for O2 in Omega2_iniciales:
    v0 = v_misma_energia(0.0, 0.01, O2, E0)
    condiciones.append([0.0, 0.01, O2, v0])
print(f"Energía común: E0 = {E0:.3f} J")

#integracion
t = np.linspace(0, t_final, 8001)
soluciones = []
for k, estado0 in enumerate(condiciones):
    sol = solve_ivp(ec_movimiento, (0, t_final), estado0, t_eval=t, method="DOP853", rtol=1e-10, atol=1e-12)
    E = np.array([energia(*s) for s in sol.y.T])
    print(f"Trayectoria {k+1}: Omega2(0) = {estado0[2]:.2f}, v(0) = {estado0[3]:.3f}  " f"->  variación máxima de la energía: {np.abs(E - E0).max():.1e} J")
    soluciones.append(sol.y)
#plot
colores = ["steelblue", "indigo", "limegreen", "crimson"]  
mostrar = t >= t_inicio 

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(projection="3d")
for y, c, O2 in zip(soluciones, colores, Omega2_iniciales):
    theta, O1, O2t, v = y
    ax.plot(O1[mostrar], O2t[mostrar], v[mostrar], color=c, lw=0.6,
            label=rf"$\Omega_2(0) = {O2}$")
ax.set_xlabel(r"$\Omega_1$ (1/s)")
ax.set_ylabel(r"$\Omega_2$ (1/s)")
ax.set_zlabel("$v$ (m/s)")
ax.set_title(r"$A_1 = 10$ cm: 4 trayectorias con la misma energía ($10 < t < 80$ s)")
ax.legend(loc="upper left", fontsize=9)

ax.view_init(elev=18, azim=-20)
ax.set_box_aspect((0.6, 2.2, 1.1))
ax.set_xticks([-5, 0, 5])          # pocas marcas en Omega1 para que no se amontonen
ax.yaxis.labelpad = 12

fig.tight_layout()
fig.savefig("figvelangularlineal.png", dpi=180)
plt.show()