import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# Parámetros de la distribución gamma
shape, scale = 2, 2  # alpha, beta
num_muestras = 1000  # Número de muestras

# Generar datos de la distribución gamma
datos = np.random.gamma(shape, scale, num_muestras)

# Crear histograma
plt.hist(datos, bins=30, density=True, alpha=0.6, color='b')

# Crear curva de la distribución gamma
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = gamma.pdf(x, shape, scale=scale)

plt.plot(x, p, 'k', linewidth=2)
titulo = "Distribución Gamma"
plt.title(titulo)
plt.show()
