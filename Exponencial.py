import numpy as np
import matplotlib.pyplot as plt

# Parámetros de la distribución exponencial
lam = 1  # Lambda
num_muestras = 1000  # Número de muestras

# Generar datos de la distribución exponencial
datos = np.random.exponential(1/lam, num_muestras)

# Crear histograma
plt.hist(datos, bins=30, density=True, alpha=0.6, color='b')

# Crear curva de la distribución exponencial
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = lam * np.exp(-lam * x)

plt.plot(x, p, 'k', linewidth=2)
titulo = "Distribución Exponencial"
plt.title(titulo)
plt.show()
