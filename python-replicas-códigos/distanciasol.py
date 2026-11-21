import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#PARAMETROS
m = 50.0                                 
I = np.diag([15.95, 13.56, 3.99])         
A = np.array([0.10, 0.12, 0.875])    
g=9.81    

#condicion inicial de referencia
estado_ref = [0.0, 0.01, 1.25, 0.5]
Omega2_iniciales = [1.25, 1.0, 1.5]
perturbacion = 1e-8

t_final = 100.0
t_ajuste = 10.0              
delta_saturacion = 1e-1

E1 = np.array([1.0, 0.0, 0.0]) 

#geometria
def vectores(theta):
    Gamma = np.array([0.0, np.sin(theta), np.cos(theta)])
    alpha = np.array([0.0, -np.cos(theta), np.sin(theta)])
    return Gamma, alpha

#ec. de movimiento
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
    theta, O1, O2, v = estado
    dO1, dO2, dv = aceleraciones(theta, O1, O2, v)
    return [O1, dO1, dO2, dv]

def ec_pareja(t, estado):
     return ec_mov(t, estado[:4]) + ec_mov(t, estado[4:])

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

# parejas de condiciones iniciales en la misma superficie de energia

E0 = energia(*estado_ref)
print(f"Energía común: E0 = {E0:.3f} J")
parejas = []
for O2 in Omega2_iniciales:
    a = [0.0, 0.01, O2, v_misma_energia(0.0, 0.01, O2, E0)]
    O1b = 0.01 + perturbacion                                   
    b = [0.0, O1b, O2, v_misma_energia(0.0, O1b, O2, E0)]   
    parejas.append(a + b)

#integracion

t = np.linspace(0, t_final, 5001)
deltas = []
for k, estado0 in enumerate(parejas):
    sol = solve_ivp(ec_pareja, (0, t_final), estado0, t_eval=t,
                    method="DOP853", rtol=1e-12, atol=1e-14)
    a, b = sol.y[:4], sol.y[4:]
    delta = np.sqrt(((a - b) ** 2).sum(axis=0))
    deltas.append(delta)
    print(f"Pareja {k+1} (Omega2(0) = {estado0[2]}): delta(0) = {delta[0]:.1e}, delta(final) = {delta[-1]:.1e}")
deltas = np.array(deltas)

#exponente de Lyapunov
t_ajuste_todos, log_ajuste_todos = [], []
for delta in deltas:
    satura = np.argmax(delta >= delta_saturacion) if np.any(delta >= delta_saturacion) else len(t)
    usar = (t >= t_ajuste) & (np.arange(len(t)) < satura)
    t_ajuste_todos.append(t[usar])
    log_ajuste_todos.append(np.log(delta[usar]))
t_aj = np.concatenate(t_ajuste_todos)
log_aj = np.concatenate(log_ajuste_todos)
lam, intercepto = np.polyfit(t_aj, log_aj, 1)
print(f"Exponente de Lyapunov (pendiente del ajuste): lambda = {lam:.3f} 1/s   (artículo: ~0.165 1/s)")

#plot
colores = ["mediumvioletred", "violet", "orange"]              
fig, ax = plt.subplots(figsize=(6, 5))
for delta, c, O2 in zip(deltas, colores, Omega2_iniciales):
    ax.plot(t, np.log(delta), color=c, lw=0.9, label=rf"$\Omega_2(0) = {O2}$")
t_recta = np.array([t_aj.min(), t_aj.max()])
ax.plot(t_recta, lam * t_recta + intercepto, color="red", lw=2.5,
        label=rf"ajuste: $\lambda \approx {lam:.3f}$ 1/s")
ax.set_xlabel("$t$ (s)")
ax.set_ylabel(r"$\log(\delta)$")
ax.set_title("Distancia entre soluciones ($A_1 = 10$ cm)")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("distanciasol.png", dpi=180)
plt.show()
