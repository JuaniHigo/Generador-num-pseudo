# Generador de Números Aleatorios y Pruebas Estadísticas

Este repositorio contiene un conjunto de scripts en Python para la generación de números pseudoaleatorios siguiendo diversas distribuciones de probabilidad. El proyecto demuestra la implementación de diferentes métodos de generación (Método de Rechazo, Transformada Inversa) y la aplicación de pruebas de bondad de ajuste para la validación estadística de los datos generados.

Este proyecto fue desarrollado como una herramienta práctica para la materia de Simulación, demostrando la aplicación de conceptos estadísticos fundamentales en un entorno de programación.

## 📄 Informe del Proyecto

Para un análisis más detallado, la metodología y la descripción completa de los resultados, podés ver el [informe completo en PDF](TP_2_2_Generadores_de_números_pseudoaleatorios_de_distintas_Distribuciones_de_Probabilidad.pdf).

## 📂 Estructura del Repositorio

El proyecto está organizado en los siguientes scripts y módulos:

### Scripts Principales

Estos son los archivos que ejecutan los análisis completos.

* `test_inversa.py`: Implementa el **Método de la Transformada Inversa** para generar variables aleatorias para las distribuciones **Uniforme** y **Exponencial**. Para la distribución **Normal**, utiliza el método de **Box-Muller**, una técnica específica basada también en la transformada inversa. El script genera muestras, las grafica junto a su función de densidad teórica (PDF) y compara sus estadísticas (media y varianza).

* `test_rechazo.py`: Implementa el **Método de Rechazo** (`rejection_sampling`) para generar variables aleatorias de 9 distribuciones diferentes. Realiza pruebas de bondad de ajuste (**Chi-cuadrado** para discretas, **Anderson-Darling/Kolmogorov-Smirnov** para continuas) y presenta los resultados en gráficos comparativos y una tabla resumen final.

### Módulos de Generación Individual

Estos scripts son implementaciones más simples para generar y visualizar datos de una distribución específica.

* `Uniforme.py`, `Exponencial.py`, `Gamma.py`, `Normal.py`, `Pascal.py`, `Binomial.py`, `Hipergeometrica.py`, `Poisson.py`, `EmpiricaDisc.py`.

## 🛠️ Tecnologías y Librerías Utilizadas

* **Python 3.x**
* **NumPy:** Para operaciones numéricas y generación de números aleatorios base.
* **SciPy:** Para el uso de funciones estadísticas avanzadas y distribuciones de probabilidad.
* **Matplotlib:** Para la visualización de datos y la creación de gráficos.
* **Pandas:** (Usado en `test_rechazo.py`) Para la estructuración de la tabla de resultados.
* **Tabulate:** (Usado en `test_rechazo.py`) Para el formateo de la tabla de resumen.

## 🚀 Cómo Ejecutar

Para ver los análisis, simplemente ejecuta los scripts principales desde tu terminal:

```bash
# Para ejecutar la simulación con el método de la Transformada Inversa
python test_inversa.py

# Para ejecutar la simulación con el método de Rechazo y las pruebas de bondad de ajuste
python test_rechazo.py
```
