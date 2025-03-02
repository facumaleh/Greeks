import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import norm
import sympy as sp
import plotly.graph_objects as go

# Configuración de la página (DEBE SER LA PRIMERA LÍNEA DE STREAMLIT)
st.set_page_config(
    layout="wide",
    page_title="Visualizador de Opciones Financieras",
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
st.title("Visualizador de Opciones Financieras")
theme = st.toggle("Modo Oscuro", value=st.session_state.get("theme", "light") == "dark", on_change=toggle_theme)
apply_theme()

# Menú de navegación con pestañas
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Aproximación de Taylor", 
    "🌳 Árbol Binomial", 
    "📈 Black-Scholes", 
    "📉 Expansión de Taylor para Call"
])

# Página de Aproximación de Taylor
with tab1:
    st.title("📊 Aproximación de Taylor")

    # Descripción de la expansión de Taylor
    with st.expander("📚 ¿Qué es la Expansión de Taylor?"):
        st.markdown("""
        **Expansión de Taylor:**
        - La expansión de Taylor permite aproximar una función alrededor de un punto \( x_0 \).
        - Aquí puedes calcular las expansiones de Taylor de grado 1 y grado 2 para cualquier función.
        """)

    # Entrada de la función
    st.header("⚙️ Ingresa una función")
    function_input = st.text_input("Ingresa una función de x (por ejemplo, sin(x), exp(x), x**2):", "sin(x)", key="taylor_function_input")

    # Configuración del gráfico
    st.header("⚙️ Configuración del gráfico")
    col1, col2, col3 = st.columns(3)
    with col1:
        x0 = st.slider("Punto de expansión (x0)", -15.0, 15.0, 0.01, 0.1, help="Punto alrededor del cual se calculará la expansión de Taylor.", key="taylor_x0")
    with col2:
        x_min = st.slider("Límite inferior de x", -15.0, 15.0, -5.0, 0.1, help="Valor mínimo de x para el gráfico.", key="taylor_x_min")
    with col3:
        x_max = st.slider("Límite superior de x", -15.0, 15.0, 5.0, 0.1, help="Valor máximo de x para el gráfico.", key="taylor_x_max")

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
        fig.update_layout(title="Aproximación de Taylor", xaxis_title="x", yaxis_title="f(x)", template="plotly_dark" if theme == "dark" else "plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error al procesar la función: {e}")

# Página de Árbol Binomial
with tab2:
    st.title("🌳 Valuación de Opciones con Árbol Binomial")

    # Descripción del modelo de árbol binomial
    with st.expander("📚 ¿Qué es el Modelo de Árbol Binomial?"):
        st.markdown("""
        **Modelo de Árbol Binomial:**
        - Este modelo permite valuar una opción call utilizando un árbol binomial.
        - Se calcula el precio de la opción hacia atrás (backwards) y se muestra la proporción de delta y deuda en cada nodo.
        """)

    # Entrada de parámetros
    st.header("⚙️ Parámetros del Modelo")
    col1, col2 = st.columns(2)
    with col1:
        S = st.number_input("Precio del Activo (S)", value=100.0, min_value=0.01, key="binomial_S")
        K = st.number_input("Precio de Ejercicio (K)", value=100.0, min_value=0.01, key="binomial_K")
        U = st.number_input("Factor de Subida (U)", value=1.1, min_value=1.0, key="binomial_U")
    with col2:
        D = st.number_input("Factor de Bajada (D)", value=0.9, max_value=1.0, key="binomial_D")
        R = st.number_input("Factor de Capitalización (R = 1 + Rf)", value=1.05, min_value=1.0, key="binomial_R")
        periods = st.number_input("Número de Periodos", value=3, min_value=1, key="binomial_periods")

    # Función para calcular el precio de la opción call usando árbol binomial
    def binomial_tree_call(S, K, U, D, R, periods):
        # Probabilidad neutral al riesgo
        q = (R - D) / (U - D)

        # Inicializar el árbol de precios del activo
        asset_prices = np.zeros((periods + 1, periods + 1))
        asset_prices[0, 0] = S
        for i in range(1, periods + 1):
            asset_prices[i, 0] = asset_prices[i - 1, 0] * U  # Nodo superior (sube)
            for j in range(1, i + 1):
                asset_prices[i, j] = asset_prices[i - 1, j - 1] * D  # Nodo inferior (baja)

        # Inicializar el árbol de precios de la opción
        option_prices = np.zeros((periods + 1, periods + 1))
        for j in range(periods + 1):
            option_prices[periods, j] = max(0, asset_prices[periods, j] - K)

        # Valuación hacia atrás
        for i in range(periods - 1, -1, -1):
            for j in range(i + 1):
                option_prices[i, j] = (q * option_prices[i + 1, j] + (1 - q) * option_prices[i + 1, j + 1]) / R

        # Calcular delta y deuda en cada nodo
        deltas = np.zeros((periods, periods + 1))
        debts = np.zeros((periods, periods + 1))
        for i in range(periods):
            for j in range(i + 1):
                deltas[i, j] = (option_prices[i + 1, j] - option_prices[i + 1, j + 1]) / (asset_prices[i + 1, j] - asset_prices[i + 1, j + 1])
                debts[i, j] = (option_prices[i + 1, j + 1] * U - option_prices[i + 1, j] * D) / (R * (U - D))

        return asset_prices, option_prices, deltas, debts

    # Calcular el árbol binomial
    asset_prices, option_prices, deltas, debts = binomial_tree_call(S, K, U, D, R, periods)

    # Función para graficar un árbol binomial
    def plot_binomial_tree(values, title, ax):
        G = nx.Graph()
        pos = {}
        labels = {}
        for i in range(values.shape[0]):
            for j in range(i + 1):
                node = (i, j)
                G.add_node(node)
                pos[node] = (i, -j + i / 2)  # Ajustar la posición vertical
                labels[node] = f"{values[i, j]:.2f}"
                if i > 0:
                    parent = (i - 1, j) if j < i else (i - 1, j - 1)
                    G.add_edge(parent, node)

        nx.draw(G, pos, labels=labels, with_labels=True, node_size=2000, node_color="lightblue", font_size=10, font_weight="bold", ax=ax)
        ax.set_title(title)

    # Mostrar los árboles binomiales uno al lado del otro
    st.subheader("📊 Árboles Binomiales")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        plot_binomial_tree(asset_prices, "Árbol de Precios del Activo", ax1)
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        plot_binomial_tree(option_prices, "Árbol de Precios de la Opción Call", ax2)
        st.pyplot(fig2)

    with col3:
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        plot_binomial_tree(deltas, "Árbol de Deltas (Δ)", ax3)
        st.pyplot(fig3)

    with col4:
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        plot_binomial_tree(debts, "Árbol de Deudas (B)", ax4)
        st.pyplot(fig4)

    # Mostrar el precio final de la opción
    st.markdown(f"**Precio de la Opción Call:** `{option_prices[0, 0]:.4f}`")

# Página de Black-Scholes
with tab3:
    st.title("📈 Visualizador de Letras Griegas en Black-Scholes")

    # Descripción de las letras griegas
    with st.expander("📚 ¿Qué son las Letras Griegas?"):
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
        S = st.slider("Precio del Activo (S)", 1.0, 200.0, 100.0, help="Precio actual del activo subyacente.", key="black_scholes_S")
    with col2:
        K = st.slider("Precio de Ejercicio (K)", 1.0, 200.0, 100.0, help="Precio al que se puede ejercer la opción.", key="black_scholes_K")
    with col3:
        T = st.slider("Tiempo hasta vencimiento (T)", 0.1, 5.0, 1.0, help="Tiempo restante hasta el vencimiento de la opción.", key="black_scholes_T")

    col4, col5 = st.columns(2)
    with col4:
        r = st.slider("Tasa libre de riesgo (r)", 0.0, 0.2, 0.05, help="Tasa de interés libre de riesgo.", key="black_scholes_r")
    with col5:
        sigma = st.slider("Volatilidad (σ)", 0.1, 1.0, 0.2, help="Volatilidad del activo subyacente.", key="black_scholes_sigma")

    # Fórmula de Black-Scholes para una opción call
    def black_scholes_call(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price

    # Calcular el precio de la opción call
    call_price = black_scholes_call(S, K, T, r, sigma)

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

    # Gráficos de las letras griegas con Plotly
    st.subheader("📊 Gráficas de las Letras Griegas")
    S_range = np.linspace(1, 200, 100)
    delta_values = delta_call(S_range, K, T, r, sigma)
    gamma_values = gamma_call(S_range, K, T, r, sigma)
    theta_values = theta_call(S_range, K, T, r, sigma)
    vega_values = vega_call(S_range, K, T, r, sigma)
    rho_values = rho_call(S_range, K, T, r, sigma)

    # Crear gráficos interactivos
    fig_delta = go.Figure()
    fig_delta.add_trace(go.Scatter(x=S_range, y=delta_values, mode='lines', name='Delta', line=dict(color='blue')))
    fig_delta.update_layout(title="Δ Delta", xaxis_title="Precio del Activo (S)", yaxis_title="Delta", template="plotly_dark" if theme == "dark" else "plotly_white")

    fig_gamma = go.Figure()
    fig_gamma.add_trace(go.Scatter(x=S_range, y=gamma_values, mode='lines', name='Gamma', line=dict(color='orange')))
    fig_gamma.update_layout(title="Γ Gamma", xaxis_title="Precio del Activo (S)", yaxis_title="Gamma", template="plotly_dark" if theme == "dark" else "plotly_white")

    fig_theta = go.Figure()
    fig_theta.add_trace(go.Scatter(x=S_range, y=theta_values, mode='lines', name='Theta', line=dict(color='green')))
    fig_theta.update_layout(title="Θ Theta", xaxis_title="Precio del Activo (S)", yaxis_title="Theta", template="plotly_dark" if theme == "dark" else "plotly_white")

    fig_vega = go.Figure()
    fig_vega.add_trace(go.Scatter(x=S_range, y=vega_values, mode='lines', name='Vega', line=dict(color='red')))
    fig_vega.update_layout(title="ν Vega", xaxis_title="Precio del Activo (S)", yaxis_title="Vega", template="plotly_dark" if theme == "dark" else "plotly_white")

    fig_rho = go.Figure()
    fig_rho.add_trace(go.Scatter(x=S_range, y=rho_values, mode='lines', name='Rho', line=dict(color='purple')))
    fig_rho.update_layout(title="ρ Rho", xaxis_title="Precio del Activo (S)", yaxis_title="Rho", template="plotly_dark" if theme == "dark" else "plotly_white")

    # Mostrar gráficos en columnas
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.plotly_chart(fig_delta, use_container_width=True)
    with col2:
        st.plotly_chart(fig_gamma, use_container_width=True)
    with col3:
        st.plotly_chart(fig_theta, use_container_width=True)
    with col4:
        st.plotly_chart(fig_vega, use_container_width=True)
    with col5:
        st.plotly_chart(fig_rho, use_container_width=True)

    # Mostrar el valor de la opción y las letras griegas en una sola fila
    st.subheader("💵 Valor de la Opción Call y Letras Griegas")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Precio de la Opción Call", f"{call_price:.4f}")
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

# Página de Expansión de Taylor para Call
with tab4:
    st.title("📉 Expansión de Taylor para una Opción Call")

    # Descripción de la expansión de Taylor aplicada a una opción call
    with st.expander("📚 ¿Qué es la Expansión de Taylor para una Opción Call?"):
        st.markdown("""
        **Expansión de Taylor para una Opción Call:**
        - La expansión de Taylor permite aproximar el precio de una opción call alrededor de un precio del activo subyacente \( S_0 \).
        - Se utiliza para estimar cómo cambia el precio de la opción cuando el precio del activo subyacente varía ligeramente.
        - Aquí se calcula la expansión de Taylor de primer y segundo orden.
        """)

    # Controles para los parámetros de la opción
    with st.expander("⚙️ Parámetros de la Opción"):
        col1, col2, col3 = st.columns(3)
        with col1:
            S0 = st.slider("Precio Actual del Activo (S₀)", 1.0, 200.0, 100.0, help="Precio actual del activo subyacente.", key="taylor_S0")
        with col2:
            K = st.slider("Precio de Ejercicio (K)", 1.0, 200.0, 100.0, help="Precio al que se puede ejercer la opción.", key="taylor_K")
        with col3:
            T = st.slider("Tiempo hasta Vencimiento (T)", 0.1, 5.0, 1.0, help="Tiempo restante hasta el vencimiento de la opción.", key="taylor_T")

        col4, col5 = st.columns(2)
        with col4:
            r = st.slider("Tasa Libre de Riesgo (r)", 0.0, 0.2, 0.05, help="Tasa de interés libre de riesgo.", key="taylor_r")
        with col5:
            sigma = st.slider("Volatilidad (σ)", 0.1, 1.0, 0.2, help="Volatilidad del activo subyacente.", key="taylor_sigma")

    # Calcular el precio de la opción call usando Black-Scholes
    def black_scholes_call(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price

    # Calcular las derivadas (Griegas) necesarias para la expansión de Taylor
    def delta_call(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(d1)

    def gamma_call(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Precio de la opción call en S₀
    call_price_S0 = black_scholes_call(S0, K, T, r, sigma)

    # Expansión de Taylor de primer y segundo orden
    def taylor_expansion_call(S, S0, call_price_S0, delta, gamma):
        taylor_1 = call_price_S0 + delta * (S - S0)  # Primer orden
        taylor_2 = taylor_1 + 0.5 * gamma * (S - S0)**2  # Segundo orden
        return taylor_1, taylor_2

    # Calcular Delta y Gamma en S₀
    delta_S0 = delta_call(S0, K, T, r, sigma)
    gamma_S0 = gamma_call(S0, K, T, r, sigma)

    # Rango de precios del activo para graficar
    S_range = np.linspace(S0 - 20, S0 + 20, 100)  # Rango alrededor de S₀

    # Calcular la expansión de Taylor para el rango de precios
    taylor_1_values, taylor_2_values = taylor_expansion_call(S_range, S0, call_price_S0, delta_S0, gamma_S0)

    # Calcular el precio real de la opción call para el rango de precios
    call_prices = [black_scholes_call(S, K, T, r, sigma) for S in S_range]

    # Mostrar las ecuaciones de la expansión de Taylor
    st.subheader("📝 Ecuaciones de la Expansión de Taylor")
    
    # Aproximación de Primer Grado (Lineal)
    st.markdown("**Aproximación de Primer Grado (Lineal):**")
    st.latex(r"""
    C(S) \approx C(S_0) + \Delta(S_0) \cdot (S - S_0)
    """)
  
    # Aproximación de Segundo Grado (Cuadrática)
    st.markdown("**Aproximación de Segundo Grado (Cuadrática):**")
    st.latex(r"""
    C(S) \approx C(S_0) + \Delta(S_0) \cdot (S - S_0) + \frac{1}{2} \Gamma(S_0) \cdot (S - S_0)^2
    """)
  

    # Graficar la expansión de Taylor y el precio real de la opción
    st.subheader("📊 Gráfica de la Expansión de Taylor")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S_range, y=call_prices, mode='lines', name='Precio Real de la Opción', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=S_range, y=taylor_1_values, mode='lines', name='Taylor Primer Orden', line=dict(color='green', dash='dash')))
    fig.add_trace(go.Scatter(x=S_range, y=taylor_2_values, mode='lines', name='Taylor Segundo Orden', line=dict(color='red', dash='dash')))
    fig.add_vline(x=S0, line=dict(color='gray', dash='dot'), annotation_text=f"S₀ = {S0}", annotation_position="top right")
    fig.update_layout(
        title="Expansión de Taylor para una Opción Call",
        xaxis_title="Precio del Activo (S)",
        yaxis_title="Precio de la Opción",
        template="plotly_dark" if theme == "dark" else "plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Mostrar los valores de Delta y Gamma en S₀
    st.subheader("📝 Valores de Delta y Gamma en S₀")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Δ Delta en S₀", f"{delta_S0:.4f}")
    with col2:
        st.metric("Γ Gamma en S₀", f"{gamma_S0:.4f}")

# Pie de página
st.markdown("---")
st.markdown("""
**Creado por:** Facundo Maleh  
**Nota:** Esta aplicación es solo para fines educativos.
""")
