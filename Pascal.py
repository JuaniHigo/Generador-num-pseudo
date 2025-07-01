import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import nbinom

# Parámetros de la distribución Pascal
n, p_val = 10, 0.5  # número de éxitos, probabilidad de éxito
num_muestras = 1000  # Número de muestras

# Generar datos de la distribución Pascal
datos = np.random.negative_binomial(n, p_val, num_muestras)

# Crear histograma
plt.hist(datos, bins=30, density=True, alpha=0.6, color='b')

# Crear curva de la distribución Pascal
xmin, xmax = plt.xlim()
xmin = int(xmin)
xmax = int(xmax)
x = np.arange(xmin, xmax)
p = nbinom.pmf(x, n, p_val)

plt.plot(x, p, 'k', linewidth=2)
titulo = "Distribución Pascal"
plt.title(titulo)
plt.show()
