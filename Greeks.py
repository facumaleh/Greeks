import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import sympy as sp
import plotly.graph_objects as go  # Para gráficos interactivos
import yfinance as yf  # Para obtener datos de precios reales
import pandas as pd

# Configuración de la página
st.set_page_config(
    layout="wide",
    page_title="Visualizador de Black-Scholes y Taylor",
    page_icon="📊"
)

# Función para cambiar entre modo claro y oscuro
def toggle_theme():
    if st.session_state.get("theme", "light") == "light":
        st.session_state.theme = "dark"
    else:
        st.session_state.theme = "light"

# Aplicar el tema seleccionado
def apply_theme():
    theme = st.session_state.get("theme", "light")
    if theme == "dark":
        st.markdown("""
        <style>
        .stApp {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
        .stSlider>div>div>div>div {
            background-color: #4CAF50;
        }
        .stTextInput>div>div>input {
            color: #FFFFFF;
        }
        .stSelectbox>div>div>div {
            color: #FFFFFF;
        }
        .stMarkdown {
            color: #FFFFFF;
        }
        .css-1d391kg {
            background-color: #1E1E1E;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp {
            background-color: #FFFFFF;
            color: #000000;
        }
        .stSlider>div>div>div>div {
            background-color: #4CAF50;
        }
        .stTextInput>div>div>input {
            color: #000000;
        }
        .stSelectbox>div>div>div {
            color: #000000;
        }
        .stMarkdown {
            color: #000000;
        }
        .css-1d391kg {
            background-color: #FFFFFF;
        }
        </style>
        """, unsafe_allow_html=True)

# Selección de tema en el cuerpo principal
st.title("Visualizador de Black-Scholes y Taylor")
theme = st.toggle("Modo Oscuro", value=st.session_state.get("theme", "light") == "dark", on_change=toggle_theme)
apply_theme()

# Menú de navegación con pestañas
tab1, tab2, tab3 = st.tabs(["📈 Black-Scholes", "📊 Aproximación de Taylor", "🆘 Ayuda"])

# Página de Black-Scholes
with tab1:
    st.title("📊 Visualizador de Letras Griegas en Black-Scholes")

    # Descripción de las letras griegas
    st.markdown("""
    **Letras Griegas:**
    - **Delta (Δ):** Sensibilidad del precio de la opción respecto al precio del activo subyacente.
    - **Gamma (Γ):** Sensibilidad de Delta respecto al precio del activo.
    - **Theta (Θ):** Sensibilidad del precio de la opción respecto al tiempo.
    - **Vega (ν):** Sensibilidad del precio de la opción respecto a la volatilidad.
    - **Rho (ρ):** Sensibilidad del precio de la opción respecto a la tasa de interés.
    """)

    # Controles en la parte superior
    st.header("⚙️ Parámetros de la Opción")
    col1, col2, col3 = st.columns(3)
    with col1:
        S = st.slider("Precio del Activo (S)", 1.0, 200.0, 100.0, help="Precio actual del activo subyacente.")
    with col2:
        K = st.slider("Precio de Ejercicio (K)", 1.0, 200.0, 100.0, help="Precio al que se ejerce la opción.")
    with col3:
        T = st.slider("Tiempo hasta vencimiento (T)", 0.1, 5.0, 1.0, help="Tiempo restante hasta el vencimiento de la opción.")

    col4, col5 = st.columns(2)
    with col4:
        r = st.slider("Tasa libre de riesgo (r)", 0.0, 0.2, 0.05, help="Tasa de interés libre de riesgo.")
    with col5:
        sigma = st.slider("Volatilidad (σ)", 0.1, 1.0, 0.2, help="Volatilidad del activo subyacente.")

    # Selección de tipo de opción (Call o Put)
    option_type = st.selectbox("Tipo de Opción", ["Call", "Put"], help="Selecciona si es una opción Call o Put.")

    # Función de Black-Scholes (Call y Put)
    @st.cache  # Almacenar en caché para mejorar el rendimiento
    def black_scholes(S, K, T, r, sigma, option_type):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "Call":
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return price

    # Calcular el precio de la opción
    option_price = black_scholes(S, K, T, r, sigma, option_type)

    # Cálculo de las letras griegas
    @st.cache
    def calculate_greeks(S, K, T, r, sigma, option_type):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "Call":
            delta = norm.cdf(d1)
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        else:
            delta = -norm.cdf(-d1)
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        vega = S * norm.pdf(d1) * np.sqrt(T)
        return delta, gamma, theta, vega, rho

    delta, gamma, theta, vega, rho = calculate_greeks(S, K, T, r, sigma, option_type)

    # Mostrar el valor de la opción y las letras griegas en una sola fila
    st.subheader("💵 Valor de la Opción y Letras Griegas")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric(f"Precio de la Opción {option_type}", f"{option_price:.4f}")
    with col2:
        st.metric("Δ Delta", f"{delta:.4f}")
    with col3:
        st.metric("Γ Gamma", f"{gamma:.4f}")
    with col4:
        st.metric("Θ Theta", f"{theta:.4f}")
    with col5:
        st.metric("ν Vega", f"{vega:.4f}")
    with col6:
        st.metric("ρ Rho", f"{rho:.4f}")

    # Gráficos de las letras griegas
    st.subheader("📊 Gráficas de las Letras Griegas")
    S_range = np.linspace(1, 200, 100)
    delta_values, gamma_values, theta_values, vega_values, rho_values = calculate_greeks(S_range, K, T, r, sigma, option_type)

    # Usar Plotly para gráficos interactivos
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S_range, y=delta_values, name="Δ Delta", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=S_range, y=gamma_values, name="Γ Gamma", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=S_range, y=theta_values, name="Θ Theta", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=S_range, y=vega_values, name="ν Vega", line=dict(color="red")))
    fig.add_trace(go.Scatter(x=S_range, y=rho_values, name="ρ Rho", line=dict(color="purple")))
    fig.update_layout(title="Letras Griegas", xaxis_title="Precio del Activo (S)", yaxis_title="Valor")
    st.plotly_chart(fig, use_container_width=True)

# Página de Aproximación de Taylor
with tab2:
    st.title("📈 Aproximación de Taylor")

    # Descripción de la expansión de Taylor
    st.markdown("""
    **Expansión de Taylor:**
    - La expansión de Taylor permite aproximar una función alrededor de un punto \( x_0 \).
    - Aquí puedes calcular las expansiones de Taylor de grado 1 y grado 2 para cualquier función.
    """)

    # Entrada de la función
    st.header("⚙️ Ingresa una función")
    function_input = st.text_input("Ingresa una función de x (por ejemplo, sin(x), exp(x), x**2):", "sin(x)")

    # Configuración del gráfico
    st.header("⚙️ Configuración del gráfico")
    col1, col2, col3 = st.columns(3)
    with col1:
        x0 = st.slider("Punto de expansión (x0)", -15.0, 15.0, 0.01, help="Punto alrededor del cual se calcula la expansión.")
    with col2:
        x_min = st.slider("Límite inferior de x", -15.0, 15.0, -5.0, help="Valor mínimo de x para el gráfico.")
    with col3:
        x_max = st.slider("Límite superior de x", -15.0, 15.0, 5.0, help="Valor máximo de x para el gráfico.")

    # Definir la variable simbólica
    x = sp.symbols('x')

    try:
        # Convertir la entrada del usuario en una función simbólica
        f = sp.sympify(function_input)

        # Calcular las derivadas
        f_prime = sp.diff(f, x)  # Primera derivada
        f_double_prime = sp.diff(f_prime, x)  # Segunda derivada

        # Expansión de Taylor de grado 1 y 2
        taylor_1 = f.subs(x, x0) + f_prime.subs(x, x0) * (x - x0)
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

        # Evaluar las funciones en el rango de x
        y_vals = f_np(x_vals)
        y_taylor_1 = taylor_1_np(x_vals)
        y_taylor_2 = taylor_2_np(x_vals)

        # Graficar la función original y las aproximaciones de Taylor
        st.subheader("📊 Gráficas")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, name=f"Función: {function_input}", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=x_vals, y=y_taylor_1, name="Taylor Grado 1", line=dict(color="green", dash="dash")))
        fig.add_trace(go.Scatter(x=x_vals, y=y_taylor_2, name="Taylor Grado 2", line=dict(color="red", dash="dash")))
        fig.update_layout(title="Aproximación de Taylor", xaxis_title="x", yaxis_title="f(x)")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error al procesar la función: {e}")

# Página de Ayuda
with tab3:
    st.title("🆘 Ayuda")
    st.markdown("""
    **Cómo usar esta aplicación:**
    - **Black-Scholes:** Calcula el precio de una opción y sus letras griegas.
    - **Aproximación de Taylor:** Aproxima una función usando expansiones de Taylor.
    - **Modo Oscuro:** Actívalo para cambiar el tema de la aplicación.
    """)

# Pie de página
st.markdown("---")
st.markdown("""
**Creado por:** Facundo Maleh  
**Nota:** Esta aplicación es solo para fines educativos.
""")
