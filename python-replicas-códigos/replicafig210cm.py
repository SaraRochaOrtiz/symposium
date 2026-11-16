import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#parametros

m= 50.0
I= np.diag([15.95, 13.56, 3.99])
A= np.array([0.10, 0.12, 0.875])
g= 9.81

#condiciones iniciales
estado0= [0.0, 0.01, 1.25, 0.5]
t_final= 20.0

E1= np.array([1.0, 0.0, 0.0])

#geometria: marco de la cuchilla
def vectores(theta):
    Gamma= np.array([0.0, np.sin(theta), np.cos(theta)])
    alpha= np.array([0.0, -np.cos(theta), np.sin(theta)])
    return Gamma, alpha

#ecs. de movimiento
def residuo(x, theta, O1, O2, v):
   dO1, dO2, dv, kappa, lam, mu = x