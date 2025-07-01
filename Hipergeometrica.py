import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import hypergeom

# Parámetros de la distribución hipergeométrica
ngood, nbad, nsample = 10, 20, 5  # número de éxitos, fracasos y tamaño de la muestra
num_muestras = 1000  # Número de muestras

# Generar datos de la distribución hipergeométrica
datos = np.random.hypergeometric(ngood, nbad, nsample, num_muestras)

# Crear histograma
plt.hist(datos, bins=30, density=True, alpha=0.6, color='b')

# Crear curva de la distribución hipergeométrica
xmin, xmax = plt.xlim()
xmin = int(xmin)
xmax = int(xmax)
M = ngood + nbad
x = np.arange(xmin, xmax)
p = hypergeom.pmf(x, M, ngood, nsample)

plt.plot(x, p, 'k', linewidth=2)
titulo = "Distribución Hipergeométrica"
plt.title(titulo)
plt.show()
