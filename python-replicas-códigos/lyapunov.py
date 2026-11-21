import time
import numpy as np
import matplotlib.pyplot as plt

#parametros

m = 50.0                                   
I1, I2, I3 = 15.95, 13.56, 3.99            
A2, A3 = 0.12, 0.875                      
g =9.81

A1_valores = np.linspace(0.0, 0.10, 21)    
Omega2_iniciales = np.linspace(0.7, 1.7, 8)  
perturbacion = 1e-8                        
t_final = 400.0                            
dt = 4e-3                                  
tau = 1.0                                 
t_transitorio = 20.0

#ec. de movimiento

Ivec = np.array([I1, I2, I3])[:, None]


def cruz(a, b):
    return np.cross(a, b, axis=0)


def punto(a, b):
    return (a * b).sum(axis=0)


def vectores(theta):
    N = theta.size
    cero, uno = np.zeros(N), np.ones(N)
    E1 = np.array([uno, cero, cero])
    Gamma = np.array([cero, np.sin(theta), np.cos(theta)])
    alpha = np.array([cero, -np.cos(theta), np.sin(theta)])
    return E1, Gamma, alpha


def campo(z, A1):
    
    theta, O1, O2, v = z
    N = theta.size
    A = np.array([A1, np.full(N, A2), np.full(N, A3)])
    E1, Gamma, alpha = vectores(theta)

    Omega = O1 * E1 + O2 * Gamma
    Y = v * E1
    P = m * (Y + cruz(Omega, A))
    Pi = Ivec * Omega + cruz(A, P)

    
    dOmega0 = -O1 * O2 * alpha                       
    dP0 = m * cruz(dOmega0, A)
    ang0 = Ivec * dOmega0 + cruz(A, dP0) + cruz(Omega, Pi) - m * g * cruz(Gamma, A) + cruz(Y, P)
    lin0 = dP0 + cruz(Omega, P)
    b = np.array([punto(E1, ang0), punto(Gamma, ang0), punto(E1, lin0)])

   
    def columna(dOmega, dY):
        dP = m * (dY + cruz(dOmega, A))
        ang = Ivec * dOmega + cruz(A, dP)
        return np.array([punto(E1, ang), punto(Gamma, ang), punto(E1, dP)])

    cero3 = 0 * E1
    M = np.stack([columna(E1, cero3), columna(Gamma, cero3), columna(cero3, E1)], axis=1)  # (3, 3, N)
    x = np.linalg.solve(M.transpose(2, 0, 1), -b.T[..., None])[..., 0].T                   # (3, N)
    return np.array([O1, x[0], x[1], x[2]])


def paso_rk4(z, A1):
    k1 = campo(z, A1)
    k2 = campo(z + 0.5 * dt * k1, A1)
    k3 = campo(z + 0.5 * dt * k2, A1)
    k4 = campo(z + dt * k3, A1)
    return z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def energia(z, A1):
   
    theta, O1, O2, v = z
    N = theta.size
    A = np.array([A1 * np.ones(N), np.full(N, A2), np.full(N, A3)])
    E1, Gamma, _ = vectores(theta)
    Omega = O1 * E1 + O2 * Gamma
    v_cm = v * E1 + cruz(Omega, A)
    return 0.5 * punto(Omega, Ivec * Omega) + 0.5 * m * punto(v_cm, v_cm) + m * g * punto(A, Gamma)


def v_misma_energia(theta, O1, O2, A1, E_objetivo):
   
    f = lambda vv: energia(np.array([[theta], [O1], [O2], [vv]]), A1)[0]
    e0, e1, em = f(0.0), f(1.0), f(-1.0)
    a, b, c = (e1 + em) / 2 - e0, (e1 - em) / 2, e0 - E_objetivo
    return (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)

#condiciones iniciales
base, pert, A1_col = [], [], []
for A1 in A1_valores:
    E0 = energia(np.array([[0.0], [0.01], [1.25], [0.5]]), A1)[0]    
    for O2 in Omega2_iniciales:
        base.append([0.0, 0.01, O2, v_misma_energia(0.0, 0.01, O2, A1, E0)])
        O1b = 0.01 + perturbacion
        pert.append([0.0, O1b, O2, v_misma_energia(0.0, O1b, O2, A1, E0)])
        A1_col.append(A1)
n = len(base)
z = np.concatenate([np.array(base).T, np.array(pert).T], axis=1)      
A1_todas = np.tile(np.array(A1_col), 2)
E_inicial = energia(z, A1_todas)

#integracion
d0 = np.sqrt(((z[:, :n] - z[:, n:]) ** 2).sum(axis=0))   
suma_log = np.zeros(n)                                   
cada = int(round(tau / dt))                              
pasos = int(round(t_final / dt))
inicio = time.time()
print(f"Integrando {2*n} trayectorias hasta t = {t_final:.0f} s ...")
for k in range(1, pasos + 1):
    z = paso_rk4(z, A1_todas)
    if k % cada == 0:
        dif = z[:, n:] - z[:, :n]                        
        d = np.sqrt((dif ** 2).sum(axis=0))              
        if k * dt > t_transitorio:
            suma_log += np.log(d / d0)
        z[:, n:] = z[:, :n] + dif * (d0 / d)             
        print(f"  t = {k*dt:5.0f} s   ({time.time() - inicio:4.0f} s de cálculo)")
print(f"Variación máxima de la energía (trayectorias base): "
      f"{np.abs(energia(z[:, :n], A1_todas[:n]) - E_inicial[:n]).max():.1e} J")
lambda_pareja = suma_log / (t_final - t_transitorio)

#exponente de Lyapunov para c/A1

npar = len(Omega2_iniciales)
L = lambda_pareja.reshape(len(A1_valores), npar)
lambdas = L.mean(axis=1)
errores = L.std(axis=1, ddof=1) / np.sqrt(npar)
for A1, lam, err in zip(A1_valores, lambdas, errores):
    print(f"  A1 = {A1*100:4.1f} cm   lambda = {lam:.3f} ± {err:.3f} 1/s")


#plot

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.errorbar(A1_valores, lambdas, yerr=errores, fmt="-o", color="dimgray", ms=4, mfc="white",
            lw=1.2, capsize=2, elinewidth=0.8)
ax.set_xlabel("$A_1$ (m)")
ax.set_ylabel(r"$\lambda$ (1/s)")
ax.set_title("Exponentes de Lyapunov")
ax.set_xlim(-0.003, 0.103)
ax.set_ylim(bottom=0)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("figlyapunov.png", dpi=180)
plt.show()