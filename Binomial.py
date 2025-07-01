import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# Parámetros de la distribución binomial
n, p = 40, 0.5  # número de ensayos, probabilidad de éxito
num_muestras = 1000  # Número de muestras

# Generar datos de la distribución binomial
datos = np.random.binomial(n, p, num_muestras)

# Crear histograma
plt.hist(datos, bins=30, density=True, alpha=0.6, color='b')

# Crear curva de la distribución binomial
xmin, xmax = plt.xlim()
xmin = int(xmin)
xmax = int(xmax)
x = np.arange(xmin, xmax)
p = binom.pmf(x, n, p)

plt.plot(x, p, 'k', linewidth=2)
titulo = "Distribución Binomial"
plt.title(titulo)
plt.show()
