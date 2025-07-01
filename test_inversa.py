import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, expon, norm

def inverse_transform_uniform(a, b, size=1000):
    """Genera números aleatorios de una distribución uniforme U(a, b)."""
    u = np.random.rand(size)
    return a + (b - a) * u

def inverse_transform_exponential(lambda_param, size=1000):
    """Genera números aleatorios de una distribución exponencial Exp(λ)."""
    u = np.random.rand(size)
    return -np.log(1 - u) / lambda_param

def inverse_transform_normal(mu, sigma, size=1000):
    """Genera números aleatorios de una distribución normal N(μ, σ²) usando el método de Box-Muller."""
    u1 = np.random.rand(size)
    u2 = np.random.rand(size)
    z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    return mu + sigma * z0

# Parámetros actualizados           
uniform_a, uniform_b = 0, 20
exponential_lambda = 1
normal_mu, normal_sigma = 175, 10

# Generar muestras
uniform_samples = inverse_transform_uniform(uniform_a, uniform_b)
exponential_samples = inverse_transform_exponential(exponential_lambda)
normal_samples = inverse_transform_normal(normal_mu, normal_sigma)

# Configurar gráficas
plt.figure(figsize=(15, 4))

# Uniforme U(a, b)
plt.subplot(1, 3, 1)
plt.hist(uniform_samples, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black', label='Muestras')
x = np.linspace(uniform_a - 1, uniform_b + 1, 1000)
plt.plot(x, uniform.pdf(x, loc=uniform_a, scale=uniform_b - uniform_a), 'r-', label='PDF Teórica')
plt.title(f'Uniforme U({uniform_a}, {uniform_b})')
plt.xlabel('Valor')
plt.ylabel('Densidad')
plt.xlim(uniform_a - 1, uniform_b + 1)
plt.legend()

# Exponencial Exp(λ)
plt.subplot(1, 3, 2)
plt.hist(exponential_samples, bins=30, density=True, alpha=0.7, color='lightgreen', edgecolor='black', label='Muestras')
x = np.linspace(0, np.max(exponential_samples) + 2, 1000)
plt.plot(x, expon.pdf(x, scale=1/exponential_lambda), 'r-', label='PDF Teórica')
plt.title(f'Exponencial Exp(λ = {exponential_lambda})')
plt.xlabel('Valor')
plt.ylabel('Densidad')
plt.xlim(0, np.max(exponential_samples) + 2)
plt.legend()

# Normal N(μ, σ²)
plt.subplot(1, 3, 3)
plt.hist(normal_samples, bins=30, density=True, alpha=0.7, color='lightsalmon', edgecolor='black', label='Muestras')
x = np.linspace(normal_mu - 4*normal_sigma, normal_mu + 4*normal_sigma, 1000)
plt.plot(x, norm.pdf(x, loc=normal_mu, scale=normal_sigma), 'r-', label='PDF Teórica')
plt.title(f'Normal N(μ = {normal_mu}, σ² = {normal_sigma}²)')
plt.xlabel('Valor')
plt.ylabel('Densidad')
plt.xlim(normal_mu - 4*normal_sigma, normal_mu + 4*normal_sigma)
plt.legend()

plt.tight_layout()
plt.show()

# Imprimir algunas estadísticas
print("Estadísticas:")
print(f"  Uniforme: Media = {np.mean(uniform_samples):.3f} (Teórica: {(uniform_a + uniform_b)/2}), Varianza = {np.var(uniform_samples):.3f} (Teórica: {(uniform_b - uniform_a)**2 / 12})")
print(f"  Exponencial: Media = {np.mean(exponential_samples):.3f} (Teórica: {1/exponential_lambda}), Varianza = {np.var(exponential_samples):.3f} (Teórica: {1/exponential_lambda**2})")
print(f"  Normal: Media = {np.mean(normal_samples):.3f} (Teórica: {normal_mu}), Varianza = {np.var(normal_samples):.3f} (Teórica: {normal_sigma**2})")