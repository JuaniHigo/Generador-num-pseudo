import numpy as np
import matplotlib.pyplot as plt

# Parámetros de la distribución normal
media = 0  # Media
desviacion_estandar = 1  # Desviación estándar
num_muestras = 1000  # Número de muestras

# Generar datos de la distribución normal
datos = np.random.normal(loc=media, scale=desviacion_estandar, size=num_muestras)

# Crear histograma
plt.hist(datos, bins=30, density=True, alpha=0.6, color='b')

# Crear curva de la distribución normal
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = np.exp(-((x - media) ** 2) / (2 * desviacion_estandar ** 2)) / (np.sqrt(2 * np.pi) * desviacion_estandar)

plt.plot(x, p, 'k', linewidth=2)
titulo = "Distribución Normal"
plt.title(titulo)
plt.show()
