import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Parámetros de la distribución Poisson
lam = 3  # Lambda
num_muestras = 1000  # Número de muestras

# Generar datos de la distribución Poisson
datos = np.random.poisson(lam, num_muestras)

# Crear histograma
plt.hist(datos, bins=30, density=True, alpha=0.6, color='b')

# Crear curva de la distribución Poisson
xmin, xmax = plt.xlim()
xmin = int(xmin)
xmax = int(xmax)
x = np.arange(xmin, xmax)
p = poisson.pmf(x, lam)

plt.plot(x, p, 'k', linewidth=2)
titulo = "Distribución Poisson"
plt.title(titulo)
plt.show()
