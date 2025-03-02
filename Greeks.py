import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import norm
import sympy as sp
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(
    layout="wide",
    page_title="Enjoy Finance",
    page_icon="📊"
)

# Aplicar el tema claro por defecto
CSS_STYLES = """
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
"""
st.markdown(CSS_STYLES, unsafe_allow_html=True)

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

# Página de Aproximación de Taylor
def taylor_approximation():
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
    function_input = st.text_input("Ingresa una función de x (por ejemplo, sin(x), exp(x), x**2):", "sin(x)", key="taylor_function_input_unique")

    # Configuración del gráfico
    st.header("⚙️ Configuración del gráfico")
    col1, col2, col3 = st.columns(3)
    with col1:
        x0 = st.slider("Punto de expansión (x0)", -15.0, 15.0, 0.01, 0.1, help="Punto alrededor del cual se calculará la expansión de Taylor.", key="taylor_x0_unique")
    with col2:
        x_min = st.slider("Límite inferior de x", -15.0, 15.0, -5.0, 0.1, help="Valor mínimo de x para el gráfico.", key="taylor_x_min_unique")
    with col3:
        x_max = st.slider("Límite superior de x", -15.0, 15.0, 5.0, 0.1, help="Valor máximo de x para el gráfico.", key="taylor_x_max_unique")

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

        # Mostrar los polinomios genéricos y evaluar en un punto específico
        st.subheader("🔢 Evaluación de los Polinomios de Taylor")
        st.markdown("**Polinomios Genéricos:**")
        st.latex(f"T_1(x) = {sp.latex(taylor_1)}")
        st.latex(f"T_2(x) = {sp.latex(taylor_2)}")

        # Entrada para evaluar los polinomios en un punto específico
        eval_point = st.number_input("Ingresa un valor de x para evaluar los polinomios de Taylor:", value=x0, key="eval_point_unique")

        # Evaluar los polinomios en el punto especificado
        taylor_1_eval = taylor_1.subs(x, eval_point)
        taylor_2_eval = taylor_2.subs(x, eval_point)

        st.markdown("**Valores de los Polinomios en el punto especificado:**")
        st.latex(f"T_1({eval_point}) = {taylor_1_eval}")
        st.latex(f"T_2({eval_point}) = {taylor_2_eval}")

    except Exception as e:
        st.error(f"Error al procesar la función: {e}")

# Llamadas a las funciones de cada pestaña
with tab1:
    taylor_approximation()

# Pie de página
st.markdown("---")
st.markdown("""
**Creado por:** Facundo Maleh  
**Nota:** Esta aplicación es solo para fines educativos.
""")
