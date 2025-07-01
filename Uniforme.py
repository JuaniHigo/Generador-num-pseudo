import numpy as np
import matplotlib.pyplot as plt

# Parámetros de la distribución uniforme
a, b = 0, 1  # Intervalo [a, b]
num_muestras = 1000  # Número de muestras

# Generar datos de la distribución uniforme
datos = np.random.uniform(a, b, num_muestras)

# Crear histograma
plt.hist(datos, bins=30, density=True, alpha=0.6, color='b')

# Crear curva de la distribución uniforme
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = np.ones_like(x) / (b - a)

plt.plot(x, p, 'k', linewidth=2)
titulo = "Distribución Uniforme"
plt.title(titulo)
plt.show()