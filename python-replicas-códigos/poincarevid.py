
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
 

# Parametros 

m = 50.0                                   
I1, I2, I3 = 15.95, 13.56, 3.99            
A2, A3 = 0.12, 0.875                      
g = 9.81                                   

A1_valores = np.concatenate([np.arange(0.0, 0.02, 0.001), np.arange(0.02, 0.1001, 0.0025)])
n_trayectorias = 30                        
t_final = 400.0                            
dt = 4e-3                                  


# Ec.mov


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
    """Devuelve dz/dt para todas las trayectorias."""
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
    """v > 0 con energía E_objetivo (E es una parábola en v: fórmula cuadrática)."""
    f = lambda vv: energia(np.array([[theta], [O1], [O2], [vv]]), A1)[0]
    e0, e1, em = f(0.0), f(1.0), f(-1.0)
    a, b, c = (e1 + em) / 2 - e0, (e1 - em) / 2, e0 - E_objetivo
    return (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)




# Constantes 
def J1(theta, O2):
    return O2 * (I2 * np.sin(theta) ** 2 + I3 * np.cos(theta) ** 2)


def beta1(theta):
    D = I2 - I3
    return (A3 / np.sqrt(I3 * D) * np.arctan(np.sqrt(D / I3) * np.sin(theta))
            - A2 / np.sqrt(I2 * D) * np.arctanh(np.sqrt(D / I2) * np.cos(theta)))


def J2(theta, O2, v):
    return v + O2 * (-A2 * np.cos(theta) + A3 * np.sin(theta)) + J1(theta, O2) * beta1(theta)


# Condiciones iniciales
def condiciones(A1, n):
    E0 = energia(np.array([[0.0], [0.01], [1.25], [0.5]]), A1)[0]
    with np.errstate(invalid="ignore"):
        prueba = np.linspace(-6, 6, 2401)
        validos = prueba[[np.isfinite(v_misma_energia(0.0, 0.01, O2, A1, E0)) for O2 in prueba]]
    O2s = np.linspace(validos.min(), validos.max(), n + 2)[1:-1]       # n valores, sin tocar los bordes
    return [[0.0, 0.01, O2, v_misma_energia(0.0, 0.01, O2, A1, E0)] for O2 in O2s]




 
# simulacion
def simular():
    z, A1_col, cuadro = [], [], []
    for c, A1 in enumerate(A1_valores):
        for ci in condiciones(A1, n_trayectorias):
            z.append(ci); A1_col.append(A1); cuadro.append(c)
    z = np.array(z).T
    A1_col = np.array(A1_col)
    cuadro = np.array(cuadro)
    N = z.shape[1]
    E_inicial = energia(z, A1_col)

    cortes_J1, cortes_J2, cortes_traj = [], [], []
    pasos = int(round(t_final / dt))
    inicio = time.time()
    print(f"Integrando {N} trayectorias ({len(A1_valores)} valores de A1) hasta t = {t_final:.0f} s ...")
    for k in range(1, pasos + 1):
        z_nuevo = paso_rk4(z, A1_col)
        i = np.where((z[1] > 0) & (z_nuevo[1] <= 0))[0]           # Omega1 pasa de + a -
        if i.size:
            w = z[1, i] / (z[1, i] - z_nuevo[1, i])
            c = (1 - w) * z[:, i] + w * z_nuevo[:, i]             # estado en el instante del corte
            cortes_J1.append(J1(c[0], c[2]))
            cortes_J2.append(J2(c[0], c[2], c[3]))
            cortes_traj.append(i)
        z = z_nuevo
        if k % 10000 == 0:
            print(f"  t = {k*dt:5.0f} s   ({time.time() - inicio:5.0f} s de cálculo)", flush=True)
    print(f"Variación máxima de la energía: {np.abs(energia(z, A1_col) - E_inicial).max():.1e} J")

    traj = np.concatenate(cortes_traj)
    np.savez("datos_animacion_poincare.npz",
             J1=np.concatenate(cortes_J1), J2=np.concatenate(cortes_J2),
             traj=traj, cuadro=cuadro[traj], orden=traj % n_trayectorias,        # posición de la trayectoria dentro de su cuadro (para el color)
             A1_valores=A1_valores, n_trayectorias=n_trayectorias)
    print("Guardado: datos_animacion_poincare.npz")


#  animacion

def animar(fps=6):
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

    d = np.load("datos_animacion_poincare.npz")
    J1c, J2c, cuadro, orden = d["J1"], d["J2"], d["cuadro"], d["orden"]
    A1s, n = d["A1_valores"], int(d["n_trayectorias"])
    colores = plt.cm.viridis(np.linspace(0, 1, n))           

    fig, ax = plt.subplots(figsize=(8, 6))
    puntos = ax.scatter([], [], s=1.5, lw=0)
    ax.set_xlim(-130, 130)
    ax.set_ylim(-17, 17)
    ax.set_xlabel("$J_1$", fontsize=12)
    ax.set_ylabel("$J_2$", fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_title(r"Sección de Poincaré ($\Omega_1 = 0$) en el plano $(J_1, J_2)$", fontsize=12)
    rotulo = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=15, fontweight="bold", va="top",
                     bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
    nota = ax.text(0.5, 0.40, "", transform=ax.transAxes, fontsize=11, ha="center", color="dimgray")
    fig.tight_layout()

    
    secuencia = [0] * 8 + list(range(len(A1s))) + [len(A1s) - 1] * 12

    def dibujar(c):
        m = cuadro == c
        puntos.set_offsets(np.c_[J1c[m], J2c[m]])
        puntos.set_color(colores[orden[m]])
        puntos.set_sizes([12 if A1s[c] == 0 else 2])
        rotulo.set_text(f"$A_1$ = {A1s[c]*100:.2f} cm")
        nota.set_text("cada trayectoria = un solo punto\n($J_1$ y $J_2$ se conservan)" if A1s[c] == 0 else "")
        return puntos, rotulo, nota

    anim = FuncAnimation(fig, dibujar, frames=secuencia, blit=False)
    anim.save("animacion_poincare.gif", writer=PillowWriter(fps=fps), dpi=90)
    print("Guardado: animacion_poincare.gif")
    try:
        anim.save("animacion_poincare.mp4", dpi=150,
                  writer=FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]))
        print("Guardado: animacion_poincare.mp4")
    except Exception as e:
        print("No se pudo hacer el .mp4 (¿falta ffmpeg?):", e)
    plt.close(fig)


if __name__ == "__main__":
    tarea = sys.argv[1] if len(sys.argv) > 1 else "todo"
    if tarea in ("simular", "todo"):
        simular()
    if tarea in ("animar", "todo"):
        animar()