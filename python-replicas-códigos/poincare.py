import time
import numpy as np
import matplotlib.pyplot as plt

m = 50.0                                   
I1, I2, I3 = 15.95, 13.56, 3.99            
A2, A3 = 0.12, 0.875                       
g = 9.81                                   

A1_valores = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]  
n_trayectorias = 30                        
t_final = 600.0                            
dt = 4e-3                                  

#ecs, mov

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
    M = np.stack([columna(E1, cero3), columna(Gamma, cero3), columna(cero3, E1)], axis=1)  
    x = np.linalg.solve(M.transpose(2, 0, 1), -b.T[..., None])[..., 0].T                   
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

#constantes de movimiento
def J1(theta, O2):
    return O2 * (I2 * np.sin(theta) ** 2 + I3 * np.cos(theta) ** 2)


def beta1(theta):
    D = I2 - I3
    return (A3 / np.sqrt(I3 * D) * np.arctan(np.sqrt(D / I3) * np.sin(theta))
            - A2 / np.sqrt(I2 * D) * np.arctanh(np.sqrt(D / I2) * np.cos(theta)))


def J2(theta, O2, v):
    return v + O2 * (-A2 * np.cos(theta) + A3 * np.sin(theta)) + J1(theta, O2) * beta1(theta)

#condiciones iniciales
def condiciones(A1, n):
    E0 = energia(np.array([[0.0], [0.01], [1.25], [0.5]]), A1)[0]
    with np.errstate(invalid="ignore"):
        prueba = np.linspace(-6, 6, 2401)
        validos = prueba[[np.isfinite(v_misma_energia(0.0, 0.01, O2, A1, E0)) for O2 in prueba]]
    O2s = np.linspace(validos.min(), validos.max(), n + 2)[1:-1]       
    return [[0.0, 0.01, O2, v_misma_energia(0.0, 0.01, O2, A1, E0)] for O2 in O2s]


z, A1_col, panel = [], [], []
for p, A1 in enumerate(A1_valores):
    for ci in condiciones(A1, n_trayectorias):
        z.append(ci); A1_col.append(A1); panel.append(p)

z = np.array(z).T
A1_col = np.array(A1_col)
panel = np.array(panel)
N = z.shape[1]
E_inicial = energia(z, A1_col)

#integracion
cortes = [[] for _ in range(N)]
pasos = int(round(t_final / dt))
inicio = time.time()
print(f"Integrando {N} trayectorias hasta t = {t_final:.0f} s ...")
for k in range(1, pasos + 1):
    z_nuevo = paso_rk4(z, A1_col)
    
    for i in np.where((z[1] > 0) & (z_nuevo[1] <= 0))[0]:
        w = z[1, i] / (z[1, i] - z_nuevo[1, i])                  
        cortes[i].append((1 - w) * z[:, i] + w * z_nuevo[:, i])
    z = z_nuevo
    if k % 25000 == 0:
        print(f"  t = {k*dt:5.0f} s   ({time.time() - inicio:4.0f} s de cálculo)")
print(f"Variación máxima de la energía: {np.abs(energia(z, A1_col) - E_inicial).max():.1e} J")
cortes = [np.array(c) for c in cortes]


def en_J(c):
    th, O1, O2, v = c.T
    return J1(th, O2), J2(th, O2, v)


#plot
fig, ejes = plt.subplots(2, 3, figsize=(15, 9))
for p, (ax, A1) in enumerate(zip(ejes.flat, A1_valores)):
    idx = np.where(panel == p)[0]
    colores = plt.cm.viridis(np.linspace(0, 1, len(idx)))
    for c, i in zip(colores, idx):
        if len(cortes[i]) == 0:
            continue
        j1, j2 = en_J(cortes[i])
        ax.plot(j1, j2, ".", ms=1.5 if A1 > 0 else 6, color=c, rasterized=True)
    ax.set_title(f"$A_1$ = {A1*100:g} cm", fontsize=12)
    ax.set_xlabel("$J_1$")
    ax.set_ylabel("$J_2$")
    ax.grid(alpha=0.3)
ejes.flat[0].text(0.03, 0.04, "cada trayectoria\n= un solo punto", transform=ejes.flat[0].transAxes, fontsize=9)
fig.suptitle(r"Sección de Poincaré ($\Omega_1 = 0$) en el plano $(J_1, J_2)$ para distintos valores de $A_1$",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("extra_poincare_A1.png", dpi=170)
plt.show()
