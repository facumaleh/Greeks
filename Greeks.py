import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import sympy as sp

# Configurar la aplicación en pantalla completa
st.set_page_config(layout="wide")

# Usar pestañas para la navegación
tab1, tab2 = st.tabs(["Black-Scholes", "Aproximación de Taylor"])

with tab1:
    # Código de la página de Black-Scholes
    st.title("📊 Visualizador de Letras Griegas en Black-Scholes")

    # Controles en dos filas
    st.header("⚙️ Parámetros de la Opción")

    col1, col2, col3 = st.columns(3)
    with col1:
        S = st.slider("Precio del Activo (S)", 1.0, 200.0, 100.0)
    with col2:
        K = st.slider("Precio de Ejercicio (K)", 1.0, 200.0, 100.0)
    with col3:
        T = st.slider("Tiempo hasta vencimiento (T)", 0.1, 5.0, 1.0)

    col4, col5 = st.columns(2)
    with col4:
        r = st.slider("Tasa libre de riesgo (r)", 0.0, 0.2, 0.05)
    with col5:
        sigma = st.slider("Volatilidad (σ)", 0.1, 1.0, 0.2)

    # Fórmula de Black-Scholes para una opción call
    def black_scholes_call(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price

    # Calcular el precio de la opción call
    call_price = black_scholes_call(S, K, T, r, sigma)

    # Mostrar el valor del call
    st.subheader("💵 Valor de la Opción Call")
    st.metric("Precio de la Opción Call", f"{call_price:.4f}")

    # Cálculo de las letras griegas
    def delta_call(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(d1)

    def gamma_call(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    def theta_call(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)

    def vega_call(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * norm.pdf(d1) * np.sqrt(T)

    def rho_call(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * T * np.exp(-r * T) * norm.cdf(d2)

    # Calcular las letras griegas
    delta = delta_call(S, K, T, r, sigma)
    gamma = gamma_call(S, K, T, r, sigma)
    theta = theta_call(S, K, T, r, sigma)
    vega = vega_call(S, K, T, r, sigma)
    rho = rho_call(S, K, T, r, sigma)

    # Mostrar los valores de las letras griegas en columnas
    st.subheader("📈 Valores de las Letras Griegas")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Δ Delta", f"{delta:.4f}")
    col2.metric("Γ Gamma", f"{gamma:.4f}")
    col3.metric("Θ Theta", f"{theta:.4f}")
    col4.metric("ν Vega", f"{vega:.4f}")
    col5.metric("ρ Rho", f"{rho:.4f}")

    # Graficar las letras griegas
    st.subheader("📊 Gráficas de las Letras Griegas")
    S_range = np.linspace(1, 200, 100)
    delta_values = delta_call(S_range, K, T, r, sigma)
    gamma_values = gamma_call(S_range, K, T, r, sigma)
    theta_values = theta_call(S_range, K, T, r, sigma)
    vega_values = vega_call(S_range, K, T, r, sigma)
    rho_values = rho_call(S_range, K, T, r, sigma)

    fig, ax = plt.subplots(3, 2, figsize=(16, 12))

    # Personalizar las gráficas
    ax[0, 0].plot(S_range, delta_values, label='Delta', color='blue')
    ax[0, 0].set_title('Δ Delta')
    ax[0, 0].set_xlabel('Precio del Activo (S)')
    ax[0, 0].set_ylabel('Delta')

    ax[0, 1].plot(S_range, gamma_values, label='Gamma', color='orange')
    ax[0, 1].set_title('Γ Gamma')
    ax[0, 1].set_xlabel('Precio del Activo (S)')
    ax[0, 1].set_ylabel('Gamma')

    ax[1, 0].plot(S_range, theta_values, label='Theta', color='green')
    ax[1, 0].set_title('Θ Theta')
    ax[1, 0].set_xlabel('Precio del Activo (S)')
    ax[1, 0].set_ylabel('Theta')

    ax[1, 1].plot(S_range, vega_values, label='Vega', color='red')
    ax[1, 1].set_title('ν Vega')
    ax[1, 1].set_xlabel('Precio del Activo (S)')
    ax[1, 1].set_ylabel('Vega')

    ax[2, 0].plot(S_range, rho_values, label='Rho', color='purple')
    ax[2, 0].set_title('ρ Rho')
    ax[2, 0].set_xlabel('Precio del Activo (S)')
    ax[2, 0].set_ylabel('Rho')

    # Ocultar la última gráfica (si no se usa)
    ax[2, 1].axis('off')

    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    # Código de la página de Aproximación de Taylor
    st.title("📈 Aproximación de Taylor")

    # Ejemplos de funciones
    st.markdown("""
    **Ejemplos de funciones que puedes usar:**
    - `sin(x)`
    - `cos(x)`
    - `exp(x)`
    - `log(x)`
    - `x**2 + 2*x + 1`
    """)

    # Entrada de la función
    st.header("⚙️ Ingresa una función")
    function_input = st.text_input("Ingresa una función de x (por ejemplo, sin(x), exp(x), x**2):", "sin(x)")

    # Configuración del gráfico
    st.header("⚙️ Configuración del gráfico")

    col1, col2, col3 = st.columns(3)
    with col1:
        x0 = st.slider(
            "Punto de expansión (x0)",
            min_value=-15.0,
            max_value=15.0,
            value=0.01,  # Valor predeterminado
            step=0.1,   # Paso más pequeño para mayor precisión
            help="Selecciona el punto alrededor del cual se calculará la expansión de Taylor."
        )
    with col2:
        x_min = st.slider(
            "Límite inferior de x",
            min_value=-10.0,
            max_value=10.0,
            value=-5.0,  # Valor predeterminado
            step=0.1,    # Paso más pequeño para mayor precisión
            help="Define el valor mínimo de x para el gráfico."
        )
    with col3:
        x_max = st.slider(
            "Límite superior de x",
            min_value=-10.0,
            max_value=10.0,
            value=5.0,   # Valor predeterminado
            step=0.1,    # Paso más pequeño para mayor precisión
            help="Define el valor máximo de x para el gráfico."
        )

    # Mostrar los valores seleccionados
    st.markdown(f"""
    - **Punto de expansión (x0):** `{x0}`
    - **Límite inferior de x:** `{x_min}`
    - **Límite superior de x:** `{x_max}`
    """)

    # Definir la variable simbólica
    x = sp.symbols('x')

    try:
        # Convertir la entrada del usuario en una función simbólica
        f = sp.sympify(function_input)

        # Calcular las derivadas
        f_prime = sp.diff(f, x)  # Primera derivada
        f_double_prime = sp.diff(f_prime, x)  # Segunda derivada

        # Expansión de Taylor de grado 1
        taylor_1 = f.subs(x, x0) + f_prime.subs(x, x0) * (x - x0)

        # Expansión de Taylor de grado 2
        taylor_2 = taylor_1 + (f_double_prime.subs(x, x0) / 2) * (x - x0)**2

        # Mostrar las expansiones de Taylor en formato matemático
        st.subheader("🔍 Expansiones de Taylor")
        st.latex(f"f(x) = {sp.latex(f)}")
        st.latex(f"T_1(x) = {sp.latex(taylor_1)}")
        st.latex(f"T_2(x) = {sp.latex(taylor_2)}")

        # Convertir las funciones simbólicas a funciones numéricas
        f_np = sp.lambdify(x, f, "numpy")
        taylor_1_np = sp.lambdify(x, taylor_1, "numpy")
        taylor_2_np = sp.lambdify(x, taylor_2, "numpy")

        # Crear un rango de valores para x
        x_vals = np.linspace(x_min, x_max, 500)

        # Graficar la función original y las aproximaciones de Taylor
        st.subheader("📊 Gráficas")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_vals, f_np(x_vals), label=f"Función: {function_input}", color='blue')
        ax.plot(x_vals, taylor_1_np(x_vals), label="Taylor Grado 1", color='green', linestyle='--')
        ax.plot(x_vals, taylor_2_np(x_vals), label="Taylor Grado 2", color='red', linestyle='--')
        ax.axvline(x=x0, color='gray', linestyle=':', label=f"Punto de expansión (x0 = {x0})")
        ax.set_title("Aproximación de Taylor")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error al procesar la función: {e}")

# Pie de página
st.markdown("---")
st.markdown("""
**Creado por:** Facundo Maleh  

**Nota:** Esta aplicación es solo para fines educativos.
""")
