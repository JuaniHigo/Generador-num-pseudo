import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, expon, gamma, norm, nbinom, binom, hypergeom, poisson

def rejection_sampling(pdf, a, b, M, size=1000, discrete=False):
    samples = []
    while len(samples) < size:
        x = np.random.uniform(a, b)
        y = np.random.uniform(0, M)
        if discrete:
            x_int = int(round(x))
            if a <= x_int <= b and y <= pdf(x_int):
                samples.append(x_int)
        else:
            if y <= pdf(x):
                samples.append(x)
    return np.array(samples)

# Configurar la figura
plt.figure(figsize=(18, 12))

# Uniforme (1, 6)
a, b = 1, 6
M = 1/(b-a)  # La altura de la PDF uniforme
samples = rejection_sampling(lambda x: uniform.pdf(x, a, b-a), a, b, M)

plt.subplot(3, 3, 1)
plt.hist(samples, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
x = np.linspace(a, b, 100)
plt.plot(x, uniform.pdf(x, a, b-a), 'r-', lw=2, label='PDF')
plt.title('Uniforme (1, 6)')
plt.legend()

# Exponencial (λ = 5)
λ = 5
a, b = 0, 10
M = λ  # Valor máximo de la PDF exponencial
samples = rejection_sampling(lambda x: expon.pdf(x, scale=1/λ), a, b, M)

plt.subplot(3, 3, 2)
plt.hist(samples, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
x = np.linspace(a, b, 100)
plt.plot(x, expon.pdf(x, scale=1/λ), 'r-', lw=2, label='PDF')
plt.title('Exponencial (λ = 5)')
plt.legend()

# Gamma (α = 3, β = 2)
α, β = 3, 2
a, b = 0, 20
# Estimación del valor máximo de la PDF gamma
M = gamma.pdf((α-1)/β, α, scale=1/β) if α >= 1 else 1.0
samples = rejection_sampling(lambda x: gamma.pdf(x, α, scale=1/β), a, b, M)

plt.subplot(3, 3, 3)
plt.hist(samples, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
x = np.linspace(a, b, 100)
plt.plot(x, gamma.pdf(x, α, scale=1/β), 'r-', lw=2, label='PDF')
plt.title('Gamma (α = 3, β = 2)')
plt.legend()

# Normal (μ = 175, σ = 10)
μ, σ = 175, 10
a, b = μ - 4*σ, μ + 4*σ  # Rango ajustado para cubrir μ ± 4σ
M = 1 / (σ * np.sqrt(2 * np.pi))  # Altura máxima de la PDF normal
samples = rejection_sampling(lambda x: norm.pdf(x, μ, σ), a, b, M)

plt.subplot(3, 3, 4)
plt.hist(samples, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
x = np.linspace(a, b, 100)
plt.plot(x, norm.pdf(x, μ, σ), 'r-', lw=2, label='PDF')
plt.title('Normal (μ = 175, σ = 10)')
plt.legend()

# Pascal (r = 3, p = 0.4)
r, p = 3, 0.4
a, b = 0, 20
M = nbinom.pmf(np.arange(a, b+1), r, p).max()  # Máximo de la PMF
samples = rejection_sampling(lambda x: nbinom.pmf(x, r, p), a, b, M, discrete=True)

plt.subplot(3, 3, 5)
plt.hist(samples, bins=range(a, b+2), density=True, alpha=0.7, color='skyblue', edgecolor='black', rwidth=0.8)
x = np.arange(a, b+1)
plt.plot(x, nbinom.pmf(x, r, p), 'ro', ms=8, label='PMF')
plt.title('Pascal (r = 3, p = 0.4)')
plt.legend()

# Binomial (n = 10, p = 0.5)
n, p = 10, 0.5
a, b = 0, n
M = binom.pmf(np.arange(a, b+1), n, p).max()  # Máximo de la PMF
samples = rejection_sampling(lambda x: binom.pmf(x, n, p), a, b, M, discrete=True)

plt.subplot(3, 3, 6)
plt.hist(samples, bins=range(a, b+2), density=True, alpha=0.7, color='skyblue', edgecolor='black', rwidth=0.8)
x = np.arange(a, b+1)
plt.plot(x, binom.pmf(x, n, p), 'ro', ms=8, label='PMF')
plt.title('Binomial (n = 10, p = 0.5)')
plt.legend()

# Hipergeométrica (M = 10, n = 5, N = 3)
M, n, N = 10, 5, 3
a, b = max(0, n+N-M), min(n, N)
M_max = hypergeom.pmf(np.arange(a, b+1), M, n, N).max()  # Máximo de la PMF
samples = rejection_sampling(lambda x: hypergeom.pmf(x, M, n, N), a, b, M_max, discrete=True)

plt.subplot(3, 3, 7)
plt.hist(samples, bins=range(a, b+2), density=True, alpha=0.7, color='skyblue', edgecolor='black', rwidth=0.8)
x = np.arange(a, b+1)
plt.plot(x, hypergeom.pmf(x, M, n, N), 'ro', ms=8, label='PMF')
plt.title('Hipergeométrica\n(M = 10, n = 5, N = 3)')
plt.legend()

# Poisson (λ = 4)
λ = 4
a, b = 0, 15
M = poisson.pmf(np.arange(a, b+1), λ).max()  # Máximo de la PMF
samples = rejection_sampling(lambda x: poisson.pmf(x, λ), a, b, M, discrete=True)

plt.subplot(3, 3, 8)
plt.hist(samples, bins=range(a, b+2), density=True, alpha=0.7, color='skyblue', edgecolor='black', rwidth=0.8)
x = np.arange(a, b+1)
plt.plot(x, poisson.pmf(x, λ), 'ro', ms=8, label='PMF')
plt.title('Poisson (λ = 4)')
plt.legend()

# Empírica Discreta ([1,2,3,4,5], [0.1,0.2,0.2,0.2,0.1])
values = [1, 2, 3, 4, 5]
probs = [0.1, 0.2, 0.2, 0.2, 0.1]
a, b = 0, len(values) - 1
M = max(probs)

def emp_pmf(x):
    if 0 <= x < len(values):
        return probs[int(x)]
    return 0

samples_idx = rejection_sampling(emp_pmf, a, b, M, discrete=True)
samples = [values[idx] for idx in samples_idx]

plt.subplot(3, 3, 9)
plt.hist(samples, bins=range(min(values)-1, max(values)+2), density=True, alpha=0.7, color='skyblue', edgecolor='black', rwidth=0.8)
x = np.arange(min(values), max(values)+1)
y = [probs[values.index(i)] if i in values else 0 for i in x]
plt.plot(x, y, 'ro', ms=8, label='PMF')
plt.title('Empírica Discreta\n[1,2,3,4,5], [0.1,0.2,0.2,0.2,0.1]')
plt.legend()

plt.tight_layout(pad=1.0)
plt.show()