# Generador de Números Aleatorios y Pruebas de Bondad de Ajuste

Este repositorio contiene un conjunto de scripts en Python para la generación de números pseudoaleatorios siguiendo diversas distribuciones de probabilidad, tanto continuas como discretas. Además, incluye un script principal que implementa el **Método de Rechazo** para la generación de variables aleatorias y realiza **Pruebas de Bondad de Ajuste** para validar los datos generados contra sus distribuciones teóricas.

Este proyecto fue desarrollado como una herramienta práctica para la materia de Simulación, demostrando la aplicación de conceptos estadísticos fundamentales en un entorno de programación.

## 📜 Descripción del Proyecto

El objetivo principal es doble:

1.  **Generación de Variables Aleatorias:** Implementar algoritmos para generar números que sigan distribuciones específicas.
2.  **Validación Estadística:** Utilizar pruebas estadísticas como Chi-cuadrado (para distribuciones discretas) y Anderson-Darling / Kolmogorov-Smirnov (para distribuciones continuas) para determinar si los números generados se ajustan correctamente a los modelos teóricos.

## 📂 Estructura del Repositorio

El proyecto está organizado en los siguientes módulos y scripts:

### Script Principal

* `test_rechazo.py`: Este es el corazón del proyecto.
    * Implementa el **Método de Rechazo** (`rejection_sampling`) como un método general para generar variables aleatorias a partir de una función de densidad o de masa de probabilidad (PDF/PMF).
    * Genera datos para 9 distribuciones diferentes: Uniforme, Exponencial, Gamma, Normal, Pascal, Binomial, Hipergeométrica, Poisson y una Empírica Discreta.
    * Realiza pruebas de bondad de ajuste para cada conjunto de datos:
        * **Chi-cuadrado** para las distribuciones discretas.
        * **Anderson-Darling** para Normal, Exponencial y Uniforme.
        * **Kolmogorov-Smirnov** como alternativa para las distribuciones Gamma y Uniforme.
    * Utiliza `matplotlib` para visualizar los resultados, comparando el histograma de los datos generados con la curva teórica de la distribución.
    * Finalmente, presenta una **tabla resumen** con los resultados de todas las pruebas utilizando `pandas` y `tabulate`.

### Módulos de Generación Individual

Estos scripts son implementaciones más simples y directas para generar y visualizar datos de una distribución específica. Son útiles para entender cada distribución de forma aislada.

* `Uniforme.py`: Genera y grafica una distribución Uniforme.
* `Exponencial.py`: Genera y grafica una distribución Exponencial.
* `Gamma.py`: Genera y grafica una distribución Gamma.
* `Normal.py`: Genera y grafica una distribución Normal.
* `Pascal.py`: Genera y grafica una distribución de Pascal (Binomial Negativa).
* `Binomial.py`: Genera y grafica una distribución Binomial.
* `Hipergeometrica.py`: Genera y grafica una distribución Hipergeométrica.
* `Poisson.py`: Genera y grafica una distribución de Poisson.
* `EmpiricaDisc.py`: Genera y grafica una distribución Empírica Discreta definida por el usuario.

## 🛠️ Tecnologías y Librerías Utilizadas

* **Python 3.x**
* **NumPy:** Para operaciones numéricas y generación de números aleatorios base.
* **SciPy:** Para el uso de funciones estadísticas avanzadas y distribuciones de probabilidad.
* **Matplotlib:** Para la visualización de datos y la creación de gráficos.
* **Pandas:** Para la estructuración y presentación de la tabla de resultados final.
* **Tabulate:** Para el formateo de la tabla de resumen en la consola.

## 🚀 Cómo Ejecutar

Para ver el análisis completo, simplemente ejecuta el script principal desde tu terminal:

```bash
python test_rechazo.py
```

Esto generará todos los gráficos y mostrará la tabla de resumen en la consola. Para ejecutar los generadores individuales, podés hacerlo de la misma manera:

```bash
python Normal.py
```
