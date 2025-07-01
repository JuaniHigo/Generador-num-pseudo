import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

# Configuración de la visualización
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Semilla para reproducibilidad
np.random.seed(42)

# Función para realizar tests de Anderson-Darling para distribuciones continuas
def test_anderson_darling(data, distribution, params=None, significance=0.05):
    """
    Realiza el test de Anderson-Darling para distribuciones continuas.
    
    Args:
        data: Los datos a analizar
        distribution: Nombre de la distribución ('norm', 'expon', etc.)
        params: Parámetros de la distribución (None para estimar de los datos)
        significance: Nivel de significancia (por defecto 0.05)
    
    Returns:
        resultado: Un diccionario con estadístico, valor crítico y conclusión
    """
    # Valores críticos por defecto para algunas distribuciones (nivel de significancia 0.05)
    critical_values = {
        'norm': 0.751,
        'expon': 0.891,
        'gumbel': 0.759,
        'uniform': 2.492
    }
    
    # Distribuciones que no son compatibles con stats.anderson
    # Usar Kolmogorov-Smirnov como alternativa
    if distribution == 'uniform':
        # Para distribución uniforme, usar el test Kolmogorov-Smirnov
        a, b = 0.0, 1.0  # Parámetros por defecto para uniforme [0,1]
        if params is not None:
            if isinstance(params, tuple) and len(params) == 2:
                a, b = params
        
        # Realizar test Kolmogorov-Smirnov
        ks_stat, ks_pvalue = stats.kstest(data, 'uniform', args=(a, b))
        
        return {
            'Estadístico': ks_stat,
            'P-valor': ks_pvalue,
            'Conclusión': f"No rechazar H₀" if ks_pvalue >= significance else f"Rechazar H₀"
        }
    else:
        # Distribuciones compatibles con stats.anderson
        try:
            # Usando scipy para el test
            result = stats.anderson(data, dist=distribution)
            
            statistic = result.statistic
            
            # Extraer el valor crítico para el nivel de significancia deseado
            if distribution == 'norm':
                # Para distribución normal, stats.anderson proporciona valores críticos
                # Los niveles de significancia son [15%, 10%, 5%, 2.5%, 1%]
                idx = np.where(np.array([0.15, 0.10, 0.05, 0.025, 0.01]) == significance)[0]
                if len(idx) > 0:
                    critical_value = result.critical_values[idx[0]]
                else:
                    critical_value = critical_values.get(distribution, None)
            else:
                critical_value = critical_values.get(distribution, None)
            
            # Determinar la conclusión
            if critical_value:
                rejects_h0 = statistic > critical_value
                conclusion = f"Rechazar H₀" if rejects_h0 else f"No rechazar H₀"
            else:
                conclusion = "No se pueden determinar valores críticos para esta distribución"
            
            return {
                'Estadístico': statistic,
                'Valor Crítico': critical_value,
                'Conclusión': conclusion
            }
        except ValueError as e:
            # Si falla, mostrar las distribuciones disponibles
            return {
                'Estadístico': None,
                'Valor Crítico': None,
                'Conclusión': f"Error: {str(e)}"
            }

# Función para realizar test Chi-cuadrado para distribuciones discretas
def test_chi_square(observed, expected, significance=0.05):
    """
    Realiza el test Chi-cuadrado para distribuciones discretas.
    
    Args:
        observed: Frecuencias observadas
        expected: Frecuencias esperadas
        significance: Nivel de significancia (por defecto 0.05)
    
    Returns:
        resultado: Un diccionario con estadístico, p-valor y conclusión
    """
    # Asegurarse de que las frecuencias esperadas sean al menos 5
    valid_test = all(e >= 5 for e in expected)
    
    if not valid_test:
        return {
            'Estadístico': None,
            'P-valor': None,
            'Conclusión': "No se puede realizar el test: algunas frecuencias esperadas son < 5"
        }
    
    # Realizar el test Chi-cuadrado
    chi2_stat, p_value = stats.chisquare(observed, expected)
    
    # Determinar la conclusión
    if p_value < significance:
        conclusion = f"Rechazar H₀ (p={p_value:.4f} < {significance})"
    else:
        conclusion = f"No rechazar H₀ (p={p_value:.4f} >= {significance})"
    
    return {
        'Estadístico': chi2_stat,
        'P-valor': p_value,
        'Conclusión': conclusion
    }

# Función para visualizar distribuciones continuas
def plot_continuous(data, dist_name, params=None):
    """
    Genera un histograma y una gráfica Q-Q para distribuciones continuas.
    
    Args:
        data: Los datos a analizar
        dist_name: Nombre de la distribución
        params: Parámetros de la distribución (None para estimar de los datos)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histograma con curva de densidad teórica
    ax1.hist(data, bins=30, density=True, alpha=0.7, label='Datos')
    
    # Generar curva teórica
    x = np.linspace(min(data), max(data), 1000)
    
    if dist_name == 'uniform':
        if params is None:
            a, b = min(data), max(data)
        else:
            a, b = params
        y = stats.uniform.pdf(x, loc=a, scale=b-a)
        ax1.plot(x, y, 'r-', lw=2, label=f'Uniforme({a:.2f}, {b:.2f})')
    
    elif dist_name == 'expon':
        if params is None:
            scale = 1/np.mean(data)
        else:
            scale = params
        y = stats.expon.pdf(x, scale=scale)
        ax1.plot(x, y, 'r-', lw=2, label=f'Exponencial(scale={1/scale:.2f})')
    
    elif dist_name == 'gamma':
        if params is None:
            # Estimación de parámetros por método de momentos
            mean = np.mean(data)
            var = np.var(data)
            shape = mean**2 / var
            scale = var / mean
        else:
            shape, scale = params
        y = stats.gamma.pdf(x, a=shape, scale=scale)
        ax1.plot(x, y, 'r-', lw=2, label=f'Gamma(α={shape:.2f}, β={scale:.2f})')
    
    elif dist_name == 'norm':
        if params is None:
            loc, scale = np.mean(data), np.std(data)
        else:
            loc, scale = params
        y = stats.norm.pdf(x, loc=loc, scale=scale)
        ax1.plot(x, y, 'r-', lw=2, label=f'Normal(μ={loc:.2f}, σ={scale:.2f})')
    
    ax1.set_title(f'Histograma para distribución {dist_name}')
    ax1.set_xlabel('Valor')
    ax1.set_ylabel('Densidad')
    ax1.legend()
    
    # Gráfico Q-Q
    if dist_name == 'norm':
        stats.probplot(data, dist="norm", plot=ax2)
        ax2.set_title('Gráfico Q-Q Normal')
    else:
        # Para otras distribuciones, crear un Q-Q plot manual
        data_sorted = np.sort(data)
        n = len(data_sorted)
        
        # Cuantiles empíricos
        p = np.arange(1, n + 1) / (n + 1)
        
        # Cuantiles teóricos
        if dist_name == 'uniform':
            if params is None:
                a, b = min(data), max(data)
            else:
                a, b = params
            q_theo = stats.uniform.ppf(p, loc=a, scale=b-a)
        elif dist_name == 'expon':
            if params is None:
                scale = 1/np.mean(data)
            else:
                scale = params
            q_theo = stats.expon.ppf(p, scale=scale)
        elif dist_name == 'gamma':
            if params is None:
                mean = np.mean(data)
                var = np.var(data)
                shape = mean**2 / var
                scale = var / mean
            else:
                shape, scale = params
            q_theo = stats.gamma.ppf(p, a=shape, scale=scale)
        
        ax2.scatter(q_theo, data_sorted)
        
        # Línea de referencia
        min_val = min(min(q_theo), min(data_sorted))
        max_val = max(max(q_theo), max(data_sorted))
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--')
        
        ax2.set_title(f'Gráfico Q-Q para distribución {dist_name}')
        ax2.set_xlabel('Cuantiles teóricos')
        ax2.set_ylabel('Cuantiles observados')
    
    plt.tight_layout()
    plt.show()

# Función para visualizar distribuciones discretas
def plot_discrete(observed, expected, dist_name, params=None):
    """
    Genera un gráfico de barras comparando frecuencias observadas y esperadas.
    
    Args:
        observed: Frecuencias observadas
        expected: Frecuencias esperadas
        dist_name: Nombre de la distribución
        params: Parámetros de la distribución
    """
    categories = np.arange(len(observed))
    
    plt.figure(figsize=(12, 6))
    width = 0.35
    
    plt.bar(categories - width/2, observed, width, label='Observado')
    plt.bar(categories + width/2, expected, width, label='Esperado')
    
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia')
    plt.title(f'Comparación de frecuencias para distribución {dist_name}')
    
    # Añadir etiqueta con parámetros
    param_text = ""
    if dist_name == "binomial":
        n, p = params
        param_text = f"n={n}, p={p:.2f}"
    elif dist_name == "poisson":
        lambda_param = params
        param_text = f"λ={lambda_param:.2f}"
    elif dist_name == "hypergeom":
        M, n, N = params
        param_text = f"M={M}, n={n}, N={N}"
    elif dist_name == "nbinom" or dist_name == "pascal":
        n, p = params
        param_text = f"n={n}, p={p:.2f}"
    
    if param_text:
        plt.annotate(param_text, xy=(0.05, 0.95), xycoords='axes fraction',
                     fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))
    
    plt.xticks(categories)
    plt.legend()
    plt.tight_layout()
    plt.show()

# =====================================================================
# EJEMPLOS PARA DISTRIBUCIONES CONTINUAS (usando Anderson-Darling)
# =====================================================================

print("=" * 80)
print("PRUEBAS PARA DISTRIBUCIONES CONTINUAS (ANDERSON-DARLING)")
print("=" * 80)

# 1. Distribución Uniforme
print("\n1. DISTRIBUCIÓN UNIFORME")
# Generar datos de una distribución uniforme
uniform_data = np.random.uniform(0, 10, 500)

# Test Anderson-Darling para distribución uniforme
uniform_result = test_anderson_darling(uniform_data, 'uniform')
print(f"Resultados para datos uniformes:")
for key, value in uniform_result.items():
    print(f"  {key}: {value}")

# Visualizar
plot_continuous(uniform_data, 'uniform')

# 2. Distribución Exponencial
print("\n2. DISTRIBUCIÓN EXPONENCIAL")
# Generar datos de una distribución exponencial
lambda_param = 0.5
exponential_data = np.random.exponential(scale=1/lambda_param, size=500)

# Test Anderson-Darling para distribución exponencial
exponential_result = test_anderson_darling(exponential_data, 'expon')
print(f"Resultados para datos exponenciales:")
for key, value in exponential_result.items():
    print(f"  {key}: {value}")

# Visualizar
plot_continuous(exponential_data, 'expon', params=1/lambda_param)

# 3. Distribución Gamma
print("\n3. DISTRIBUCIÓN GAMMA")
# Generar datos de una distribución gamma
shape, scale = 2.0, 2.0
gamma_data = np.random.gamma(shape, scale, 500)

# Test Anderson-Darling para distribución gamma
# Nota: Anderson-Darling en scipy no tiene implementación directa para gamma
# Se puede aproximar con transformaciones o usar test Kolmogorov-Smirnov
# como alternativa
ks_stat, ks_pvalue = stats.kstest(gamma_data, 'gamma', args=(shape, 0, scale))

print(f"Resultados para datos gamma (usando Kolmogorov-Smirnov en lugar de A-D):")
print(f"  Estadístico KS: {ks_stat}")
print(f"  P-valor: {ks_pvalue}")
if ks_pvalue < 0.05:
    print(f"  Conclusión: Rechazar H₀ (p={ks_pvalue:.4f} < 0.05)")
else:
    print(f"  Conclusión: No rechazar H₀ (p={ks_pvalue:.4f} >= 0.05)")

# Visualizar
plot_continuous(gamma_data, 'gamma', params=(shape, scale))

# 4. Distribución Normal
print("\n4. DISTRIBUCIÓN NORMAL")
# Generar datos de una distribución normal
mu, sigma = 0, 1
normal_data = np.random.normal(mu, sigma, 500)

# Test Anderson-Darling para distribución normal
normal_result = test_anderson_darling(normal_data, 'norm')
print(f"Resultados para datos normales:")
for key, value in normal_result.items():
    print(f"  {key}: {value}")

# Visualizar
plot_continuous(normal_data, 'norm', params=(mu, sigma))

# =====================================================================
# EJEMPLOS PARA DISTRIBUCIONES DISCRETAS (usando Chi-cuadrado)
# =====================================================================

print("\n" + "=" * 80)
print("PRUEBAS PARA DISTRIBUCIONES DISCRETAS (CHI-CUADRADO)")
print("=" * 80)

# 5. Distribución Pascal (Binomial Negativa)
print("\n5. DISTRIBUCIÓN PASCAL (BINOMIAL NEGATIVA)")
# Generar datos de una distribución binomial negativa (Pascal)
n, p = 5, 0.3
pascal_data = np.random.negative_binomial(n, p, 1000)

# Calcular frecuencias observadas y esperadas
max_val = max(15, max(pascal_data))
observed = np.bincount(pascal_data, minlength=max_val+1)
x_range = np.arange(max_val+1)

# Calcular probabilidades teóricas y frecuencias esperadas
pascal_pmf = stats.nbinom.pmf(x_range, n, p)
expected = pascal_pmf * len(pascal_data)

# Combinar categorías con frecuencias esperadas bajas
threshold = 5
combined_observed = []
combined_expected = []
current_obs = 0
current_exp = 0

for obs, exp in zip(observed, expected):
    if exp < threshold:
        current_obs += obs
        current_exp += exp
    else:
        if current_obs > 0:
            combined_observed.append(current_obs)
            combined_expected.append(current_exp)
            current_obs = 0
            current_exp = 0
        combined_observed.append(obs)
        combined_expected.append(exp)

# Añadir el último grupo si existe
if current_obs > 0:
    combined_observed.append(current_obs)
    combined_expected.append(current_exp)

# Test Chi-cuadrado
pascal_result = test_chi_square(combined_observed, combined_expected)
print(f"Resultados para distribución Pascal (Binomial Negativa):")
for key, value in pascal_result.items():
    print(f"  {key}: {value}")

# Visualizar (usando las frecuencias originales para mejor visualización)
plot_discrete(observed[:15], expected[:15], 'nbinom', params=(n, p))

# 6. Distribución Binomial
print("\n6. DISTRIBUCIÓN BINOMIAL")
# Generar datos de una distribución binomial
n, p = 10, 0.3
binomial_data = np.random.binomial(n, p, 1000)

# Calcular frecuencias observadas
observed = np.bincount(binomial_data, minlength=n+1)
x_range = np.arange(n+1)

# Calcular probabilidades teóricas y frecuencias esperadas
binomial_pmf = stats.binom.pmf(x_range, n, p)
expected = binomial_pmf * len(binomial_data)

# Test Chi-cuadrado
binomial_result = test_chi_square(observed, expected)
print(f"Resultados para distribución Binomial:")
for key, value in binomial_result.items():
    print(f"  {key}: {value}")

# Visualizar
plot_discrete(observed, expected, 'binomial', params=(n, p))

# 7. Distribución Hipergeométrica
print("\n7. DISTRIBUCIÓN HIPERGEOMÉTRICA")
# Parámetros hipergeométricos
M = 50  # población total
n = 10  # número de éxitos en la población
N = 7   # tamaño de la muestra

# Generar datos de una distribución hipergeométrica
hypergeom_data = np.random.hypergeometric(n, M-n, N, 1000)

# Calcular frecuencias observadas
observed = np.bincount(hypergeom_data, minlength=min(n, N)+1)
x_range = np.arange(min(n, N)+1)

# Calcular probabilidades teóricas y frecuencias esperadas
hypergeom_pmf = stats.hypergeom.pmf(x_range, M, n, N)
expected = hypergeom_pmf * len(hypergeom_data)

# Test Chi-cuadrado
hypergeom_result = test_chi_square(observed, expected)
print(f"Resultados para distribución Hipergeométrica:")
for key, value in hypergeom_result.items():
    print(f"  {key}: {value}")

# Visualizar
plot_discrete(observed, expected, 'hypergeom', params=(M, n, N))

# 8. Distribución de Poisson
print("\n8. DISTRIBUCIÓN DE POISSON")
# Generar datos de una distribución de Poisson
lambda_param = 3.0
poisson_data = np.random.poisson(lambda_param, 1000)

# Calcular frecuencias observadas
max_val = max(15, max(poisson_data))
observed = np.bincount(poisson_data, minlength=max_val+1)
x_range = np.arange(max_val+1)

# Calcular probabilidades teóricas y frecuencias esperadas
poisson_pmf = stats.poisson.pmf(x_range, lambda_param)
expected = poisson_pmf * len(poisson_data)

# Combinar categorías con frecuencias esperadas bajas
threshold = 5
combined_observed = []
combined_expected = []
current_obs = 0
current_exp = 0

for obs, exp in zip(observed, expected):
    if exp < threshold:
        current_obs += obs
        current_exp += exp
    else:
        if current_obs > 0:
            combined_observed.append(current_obs)
            combined_expected.append(current_exp)
            current_obs = 0
            current_exp = 0
        combined_observed.append(obs)
        combined_expected.append(exp)

# Añadir el último grupo si existe
if current_obs > 0:
    combined_observed.append(current_obs)
    combined_expected.append(current_exp)

# Test Chi-cuadrado
poisson_result = test_chi_square(combined_observed, combined_expected)
print(f"Resultados para distribución de Poisson:")
for key, value in poisson_result.items():
    print(f"  {key}: {value}")

# Visualizar (usando las frecuencias originales para mejor visualización)
plot_discrete(observed[:10], expected[:10], 'poisson', params=lambda_param)

# 9. Distribución Empírica Discreta
print("\n9. DISTRIBUCIÓN EMPÍRICA DISCRETA")
# Crear una distribución empírica discreta personalizada
custom_probs = [0.1, 0.2, 0.3, 0.25, 0.15]
custom_values = np.arange(len(custom_probs))

# Generar datos
custom_data = np.random.choice(custom_values, size=1000, p=custom_probs)

# Calcular frecuencias observadas
observed = np.bincount(custom_data, minlength=len(custom_probs))

# Las frecuencias esperadas son los valores teóricos
expected = np.array(custom_probs) * len(custom_data)

# Test Chi-cuadrado
custom_result = test_chi_square(observed, expected)
print(f"Resultados para distribución Empírica Discreta:")
for key, value in custom_result.items():
    print(f"  {key}: {value}")

# Visualizar
plot_discrete(observed, expected, 'empírica discreta')

# =====================================================================
# RESUMEN DE RESULTADOS
# =====================================================================

print("\n" + "=" * 80)
print("RESUMEN DE RESULTADOS")
print("=" * 80)

# Crear un DataFrame para los resultados
results_data = []

# Distribuciones continuas (Anderson-Darling)
continuas = [
    ("Uniforme", uniform_result),
    ("Exponencial", exponential_result),
    ("Gamma", {"Estadístico": ks_stat, "P-valor": ks_pvalue, "Conclusión": "No rechazar H₀" if ks_pvalue >= 0.05 else "Rechazar H₀"}),
    ("Normal", normal_result)
]

for dist_name, result in continuas:
    estadistico = result.get('Estadístico', 'N/A')
    if isinstance(estadistico, float):
        estadistico = f"{estadistico:.4f}"
    
    valor_critico = result.get('Valor Crítico', result.get('P-valor', 'N/A'))
    if isinstance(valor_critico, float):
        valor_critico = f"{valor_critico:.4f}"
    
    results_data.append({
        "Distribución": dist_name,
        "Test": "Anderson-Darling" if dist_name != "Gamma" else "Kolmogorov-Smirnov",
        "Estadístico": estadistico,
        "Valor Crítico/P-valor": valor_critico,
        "Conclusión": result.get('Conclusión', 'N/A')
    })

# Distribuciones discretas (Chi-cuadrado)
discretas = [
    ("Pascal", pascal_result),
    ("Binomial", binomial_result),
    ("Hipergeométrica", hypergeom_result),
    ("Poisson", poisson_result),
    ("Empírica Discreta", custom_result)
]

for dist_name, result in discretas:
    estadistico = result.get('Estadístico', 'N/A')
    if isinstance(estadistico, float):
        estadistico = f"{estadistico:.4f}"
    
    p_valor = result.get('P-valor', 'N/A')
    if isinstance(p_valor, float):
        p_valor = f"{p_valor:.4f}"
    
    results_data.append({
        "Distribución": dist_name,
        "Test": "Chi-cuadrado",
        "Estadístico": estadistico,
        "Valor Crítico/P-valor": p_valor,
        "Conclusión": result.get('Conclusión', 'N/A')
    })

# Crear y mostrar la tabla
df_results = pd.DataFrame(results_data)
print(tabulate(df_results, headers='keys', tablefmt='grid', showindex=False))

print("\nNota: Para los tests Anderson-Darling, se muestra el valor crítico.")
print("Para los tests Chi-cuadrado y Kolmogorov-Smirnov, se muestra el p-valor.")
print("En general, rechazamos H₀ si el estadístico > valor crítico o si p-valor < 0.05")
