import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#parametro

m= 50.0
I = np.diag([15.95, 13.56, 3.99])
A = np.array([0.10, 0.12, 0.875])
g = 9.81  

#condiciones iniciales

estado0= [0.0, 0.01, 1.25, 0.5, 0.0, 0.0, 0.0]
t_final = 20.0
E1 = np.array([1.0, 0.0, 0.0])

#geometria: marco de la cuchilla 
def vectores(theta):
    Gamma = np.array([0.0, np.sin(theta), np.cos(theta)])
    alpha = np.array([0.0, -np.cos(theta), np.sin(theta)])
    return Gamma, alpha

#ec. mov
def residuo(x, theta, O1, O2, v): 
  dO1, dO2, dv, kappa,lam, mu = x
  Gamma, alpha =vectores(theta)

  Omega = O1 *E1 + O2  *Gamma
  Y = v *E1
  P =m * (Y + np.cross (Omega, A))
  Pi = I @ Omega + np.cross(A, P)

  #derivadas temporales
  dGamma = -O1 *alpha
  dOmega = dO1 *E1 + dO2 *Gamma + O2 * dGamma
  dY= dv *E1
  dP = m *( dY + np.cross(dOmega, A))
  dPi = I @ dOmega + np.cross(A, dP)

  angular= dPi + np.cross(Omega, Pi) - m*g*np.cross(Gamma, A ) + np.cross(Y, P)- kappa * alpha
  lineal= dP + np.cross (Omega, P) - lam *Gamma - mu *alpha
  return np.concatenate([angular, lineal])

def aceleraciones(theta, O1, O2, v):
   r0= residuo(np.zeros(6), theta, O1, O2, v)
   M= np.column_stack([residuo(np.eye(6)[j], theta, O1, O2, v) -r0 for j in range(6)])
   x= np.linalg.solve(M, -r0)
   return x[:3]

def ec_mov(t, estado):
   theta, O1, O2, v, psi, x, y =estado
   dO1, dO2, dv =aceleraciones(theta, O1, O2, v)
   return[O1, dO1, dO2, dv, O2, v * np.cos(psi), v*np.sin(psi)]

def energia(theta, O1, O2, v):
   Gamma, _ = vectores(theta)
   Omega = O1 * E1 + O2*Gamma
   v_cm = v*E1 + np.cross(Omega, A)
   return 0.5*Omega @ I @ Omega + 0.5 * m * v_cm @ v_cm + m * g * A @ Gamma 

#integracion
t = np.linspace(0, t_final, 4001)
sol = solve_ivp(ec_mov, (0, t_final), estado0, t_eval=t,
                method="DOP853", rtol=1e-10, atol=1e-12)
theta, O1, O2, v, psi, x, y = sol.y

# comprobacion: la energía debe conservarse
E = np.array([energia(*s) for s in sol.y[:4].T])
print(f"Variación máxima de la energía: {np.abs(E - E[0]).max():.1e} J  (E0 = {E[0]:.3f} J)")

#tiempos y posicion en el hielo
def ceros(t, f):
    i = np.where(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]
    return t[i] - f[i] * (t[i + 1] - t[i]) / (f[i + 1] - f[i])

t_O1 = ceros(t, O1)
t_v = ceros(t, v)
x_O1, y_O1 = np.interp(t_O1, t, x), np.interp(t_O1, t, y)
x_v, y_v = np.interp(t_v, t, x), np.interp(t_v, t, y)

#plot
fig = plt.figure(figsize=(6, 10))


ax1 = fig.add_subplot(2, 1, 1, projection="3d")
ax1.plot(O1, O2, v, color="tab:blue", lw=1.2)
ax1.set_xlabel(r"$\Omega_1$ (1/s)")
ax1.set_ylabel(r"$\Omega_2$ (1/s)")
ax1.set_zlabel("$v$ (m/s)")
ax1.set_title(r"$A_1 = 10$ cm: espacio $(\Omega_1, \Omega_2, v)$")


ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(x, y, color="black", lw=1.2)
ax2.plot(x_O1, y_O1, "*", color="tab:orange", ms=10, label=r"inflexión ($\Omega_1 = 0$)")
ax2.plot(x_v, y_v, "x", color="tab:blue", ms=8, mew=2, label="cúspide ($v = 0$)")
ax2.set_aspect("equal", "datalim")
ax2.set_xlabel("$x$ (m)")
ax2.set_ylabel("$y$ (m)")
ax2.set_title(r"$A_1 = 10$ cm: cuchilla sobre el hielo ($0 \leq t \leq 20$ s)")
ax2.legend(loc="best", fontsize=9)
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("fig3_10cm.png", dpi=180)
plt.show()
