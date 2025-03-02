import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import norm
import sympy as sp
import plotly.graph_objects as go
import pandas as pd

# Configuración de la página
st.set_page_config(
    layout="wide",
    page_title="Enjoy Finance",
    page_icon="📊"
)

# Aplicar el tema claro por defecto
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

# Título de la aplicación
st.title("Enjoy Finance")

# Menú de navegación con pestañas
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1️⃣ Aproximación de Taylor", 
    "2️⃣ Árbol Binomial", 
    "3️⃣ Black-Scholes", 
    "4️⃣ Expansión de Taylor para Call",
    "5️⃣ Optimización con Lagrange",
    "6️⃣ Paridad Put-Call"
])

# Función para calcular la expansión de Taylor
@st.cache_data  # Updated to use st.cache_data
def calcular_taylor(function_input, x0, x_min, x_max):
    """
    Calcula la expansión de Taylor de primer y segundo orden para una función dada.
    
    Parámetros:
    - function_input: La función a expandir.
    - x0: Punto de expansión.
    - x_min: Límite inferior del rango de x.
    - x_max: Límite superior del rango de x.
    
    Retorna:
    - taylor_1: Expansión de Taylor de primer orden.
    - taylor_2: Expansión de Taylor de segundo orden.
    """
    x = sp.symbols('x')
    try:
        f = sp.sympify(function_input)
        f_prime = sp.diff(f, x)
        f_double_prime = sp.diff(f_prime, x)
        taylor_1 = f.subs(x, x0) + f_prime.subs(x, x0) * (x - x0)
        taylor_2 = taylor_1 + (f_double_prime.subs(x, x0) / 2) * (x - x0)**2
        return taylor_1, taylor_2
    except sp.SympifyError:
        st.error("La función ingresada no es válida. Por favor, ingresa una función válida.")
        st.stop()

# Página de Aproximación de Taylor
with tab1:
    st.title("📊 Aproximación de Taylor")

    # Descripción de la expansión de Taylor
    with st.expander("📚 ¿Qué es la Expansión de Taylor?"):
        st.markdown("""
        **Expansión de Taylor:**
        - La expansión de Taylor permite aproximar una función alrededor de un punto \( x_0 \).
        - Aca podes calcular las expansiones de Taylor de grado 1 y grado 2 para cualquier función.
        """)

    # Entrada de la función
    st.header("⚙️ Ingresa una función")
    function_input = st.text_input("Ingresa una función de x (por ejemplo, sin(x), exp(x), x**2):", "sin(x)", key="taylor_function_input")

    # Configuración del gráfico
    with st.expander("⚙️ Configuración del gráfico"):
        col1, col2, col3 = st.columns(3)
        with col1:
            x0 = st.slider("Punto de expansión (x0)", -15.0, 15.0, 0.01, 0.1, key="taylor_x0")
        with col2:
            x_min = st.slider("Límite inferior de x", -15.0, 15.0, -5.0, 0.1, key="taylor_x_min")
        with col3:
            x_max = st.slider("Límite superior de x", -15.0, 15.0, 5.0, 0.1, key="taylor_x_max")

    # Calcular la expansión de Taylor
    taylor_1, taylor_2 = calcular_taylor(function_input, x0, x_min, x_max)

    # Mostrar las expansiones de Taylor en formato matemático
    st.subheader("🔍 Expansiones de Taylor")
    st.latex(f"f(x) = {sp.latex(sp.sympify(function_input))}")
    st.latex(f"T_1(x) = {sp.latex(taylor_1)}")
    st.latex(f"T_2(x) = {sp.latex(taylor_2)}")

    # Convertir las funciones simbólicas a funciones numéricas
    x = sp.symbols('x')
    f_np = sp.lambdify(x, sp.sympify(function_input), "numpy")
    taylor_1_np = sp.lambdify(x, taylor_1, "numpy")
    taylor_2_np = sp.lambdify(x, taylor_2, "numpy")

    # Crear un rango de valores para x
    x_vals = np.linspace(x_min, x_max, 500)

    # Evaluar las funciones en el rango de x
    try:
        y_vals = f_np(x_vals)
        y_taylor_1 = taylor_1_np(x_vals)
        y_taylor_2 = taylor_2_np(x_vals)
    except Exception as e:
        st.error(f"Error al evaluar la función: {e}")
        st.stop()

    # Graficar la función original y las aproximaciones de Taylor
    st.subheader("📊 Gráficas")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name=f"Función: {function_input}", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=x_vals, y=y_taylor_1, mode='lines', name="Taylor Grado 1", line=dict(color='green', dash='dash')))
    fig.add_trace(go.Scatter(x=x_vals, y=y_taylor_2, mode='lines', name="Taylor Grado 2", line=dict(color='red', dash='dash')))
    fig.add_vline(x=x0, line=dict(color='gray', dash='dot'), annotation_text=f"x0 = {x0}", annotation_position="top right")
    fig.update_layout(title="Aproximación de Taylor", xaxis_title="x", yaxis_title="f(x)", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Exportar datos
    st.subheader("📤 Exportar Datos")
    df = pd.DataFrame({
        "x": x_vals,
        "f(x)": y_vals,
        "Taylor 1": y_taylor_1,
        "Taylor 2": y_taylor_2
    })
    st.download_button("Descargar datos como CSV", df.to_csv(index=False), "datos_taylor.csv", "text/csv")

# Página de Árbol Binomial (ejemplo simplificado)
with tab2:
    st.title("🌳 Valuación de Opciones con Árbol Binomial")
    st.write("Implementación del árbol binomial...")

# Página de Black-Scholes (ejemplo simplificado)
with tab3:
    st.title("📈 Visualizador de Letras Griegas en Black-Scholes")
    st.write("Implementación de Black-Scholes...")

# Página de Expansión de Taylor para Call (ejemplo simplificado)
with tab4:
    st.title("📉 Expansión de Taylor para una Opción Call")
    st.write("Implementación de la expansión de Taylor para opciones call...")

# Página de Optimización con Lagrange (ejemplo simplificado)
with tab5:
    st.title("🔍 Optimización con Método de Lagrange")
    st.write("Implementación del método de Lagrange...")

# Página de Paridad Put-Call (ejemplo simplificado)
with tab6:
    st.title("📉 Valor de un Put usando Paridad Put-Call")
    st.write("Implementación de la paridad put-call...")

# Pie de página
st.markdown("---")
st.markdown("""
**Creado por:** Facundo Maleh  
**Nota:** Esta aplicación es solo para fines educativos.
""")
