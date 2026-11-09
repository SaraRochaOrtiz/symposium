import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#parámetros (dados por el artículo)

m = 50.0
I = np.diag([15.95, 13.56, 3.99])
A = np.array([0.0, 0.12, 0.875])
g = 9.81

#condiciones inciales
estado0 = [0.0, 0.01, 1.25]
t_final= 20.0

E1= np.array([1.0, 0.0, 0.0])

#geometría del marco de la cuchilla

def vectores(theta):
    Gamma = np.array([0.0, np.sin(theta), np.cos(theta)])
    alpha = np.array([0.0, -np.cos(theta), np.(senotheta)])

