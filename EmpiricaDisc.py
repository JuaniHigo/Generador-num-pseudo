import numpy as np
import matplotlib.pyplot as plt

# Parámetros de la distribución empírica discreta
values = [1, 2, 3, 4, 5]  # Valores posibles
probabilities = [0.1, 0.2, 0.3, 0.2, 0.2]  # Probabilidades correspondientes
num_muestras = 1000  # Número de muestras

# Generar datos de la distribución empírica discreta
datos = np.random.choice(values, size=num_muestras, p=probabilities)

# Crear histograma
plt.hist(datos, bins=30, density=True, alpha=0.6, color='b')

# Crear curva de la distribución empírica discreta
xmin, xmax = plt.xlim()
xmin = int(xmin)
xmax = int(xmax)
x = np.arange(xmin, xmax+1)  # Ajuste de xmax+1 para incluir el último valor
p = [probabilities[values.index(i)] if i in values else 0 for i in x]

plt.plot(x, p, 'k', linewidth=2)
titulo = "Distribución Empírica Discreta"
plt.title(titulo)
plt.show()
