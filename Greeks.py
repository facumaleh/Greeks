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
st.title("Enjoy Finance 📊")

# Menú de navegación con pestañas
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9  = st.tabs([
    "1️⃣ Aproximación de Taylor", 
    "2️⃣ Árbol Binomial", 
    "3️⃣ Black-Scholes", 
    "4️⃣ Expansión de Taylor para Call",
    "5️⃣ Optimización con Lagrange",
    "6️⃣ Paridad Put-Call",
    "7️⃣ Simulación de Monte Carlo para Opciones",
    "8️⃣ Explicación Gráfica de Taylor",
    "9️⃣ uncertainty y prima"
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
    function_input = st.text_input(
        "Ingresa una función de x (por ejemplo, sin(x), exp(x), x**2):", 
        "x*x", 
        key="taylor_function_input",
        help="Ingresa una función válida de x."
    )

    # Configuración del gráfico
    st.header("⚙️ Configuración del gráfico")
    col1, col2, col3 = st.columns(3)
    with col1:
        x0 = st.number_input(
            "Punto de expansión (x0)", 
            value=0.01, 
            format="%.4f", 
            key="taylor_x0_input",
            help="Punto alrededor del cual se calculará la expansión de Taylor."
        )
    with col2:
        x_min = st.number_input(
            "Límite inferior de x", 
            value=-5.0, 
            format="%.4f", 
            key="taylor_x_min_input",
            help="Valor mínimo de x para el gráfico."
        )
    with col3:
        x_max = st.number_input(
            "Límite superior de x", 
            value=5.0, 
            format="%.4f", 
            key="taylor_x_max_input",
            help="Valor máximo de x para el gráfico."
        )

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
        fig.add_trace(go.Scatter(
            x=x_vals, 
            y=y_vals, 
            mode='lines', 
            name=f"Función: {function_input}", 
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=x_vals, 
            y=y_taylor_1, 
            mode='lines', 
            name="Taylor Grado 1", 
            line=dict(color='green', dash='dash', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=x_vals, 
            y=y_taylor_2, 
            mode='lines', 
            name="Taylor Grado 2", 
            line=dict(color='red', dash='dash', width=2)
        ))
        fig.add_vline(
            x=x0, 
            line=dict(color='gray', dash='dot'), 
            annotation_text=f"x0 = {x0}", 
            annotation_position="top right"
        )
        fig.update_layout(
            title="Aproximación de Taylor",
            xaxis_title="x",
            yaxis_title="f(x)",
            template="plotly_white",
            legend=dict(x=0.02, y=0.98)
        )
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
        S = st.number_input(
            "Precio del Activo (S)", 
            value=100.0, 
            min_value=0.01, 
            format="%.4f", 
            key="binomial_S",
            help="Precio actual del activo subyacente."
        )
        K = st.number_input(
            "Precio de Ejercicio (K)", 
            value=100.0, 
            min_value=0.01, 
            format="%.4f", 
            key="binomial_K",
            help="Precio al que se puede ejercer la opción."
        )
        U = st.number_input(
            "Factor de Subida (U)", 
            value=1.1, 
            min_value=1.0, 
            format="%.4f", 
            key="binomial_U",
            help="Factor de subida del precio del activo."
        )
    with col2:
        D = st.number_input(
            "Factor de Bajada (D)", 
            value=0.9, 
            max_value=1.0, 
            format="%.4f", 
            key="binomial_D",
            help="Factor de bajada del precio del activo."
        )
        R = st.number_input(
            "Factor de Capitalización (R = 1 + Rf)", 
            value=1.05, 
            min_value=1.0, 
            format="%.4f", 
            key="binomial_R",
            help="Factor de capitalización libre de riesgo."
        )
        periods = st.number_input(
            "Número de Periodos", 
            value=3, 
            min_value=1, 
            key="binomial_periods",
            help="Número de periodos en el árbol binomial."
        )

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

    # Función para graficar un árbol binomial (diseño original)
    def plot_binomial_tree(values, title, ax):
        G = nx.Graph()
        pos = {}
        labels = {}
        for i in range(values.shape[0]):
            for j in range(i + 1):
                node = (i, j)
                G.add_node(node)
                pos[node] = (i, -j + i / 2)  # Ajustar la posición vertical
                labels[node] = f"{values[i, j]:.4f}"  # Mostrar valores con 4 decimales
                if i > 0:
                    parent = (i - 1, j) if j < i else (i - 1, j - 1)
                    G.add_edge(parent, node)

        # Dibujar el árbol
        nx.draw(G, pos, labels=labels, with_labels=True, node_size=1000, node_color="red", font_size=10, font_weight="bold", ax=ax)
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
        S = st.slider(
            "Precio del Activo (S)", 
            1.0, 200.0, 100.0, 
            help="Precio actual del activo subyacente.", 
            key="black_scholes_S"
        )
    with col2:
        K = st.slider(
            "Precio de Ejercicio (K)", 
            1.0, 200.0, 100.0, 
            help="Precio al que se puede ejercer la opción.", 
            key="black_scholes_K"
        )
    with col3:
        T = st.slider(
            "Tiempo hasta vencimiento (T)", 
            0.1, 5.0, 1.0, 
            help="Tiempo restante hasta el vencimiento de la opción.", 
            key="black_scholes_T"
        )

    col4, col5 = st.columns(2)
    with col4:
        r = st.slider(
            "Tasa libre de riesgo (r)", 
            0.0, 0.2, 0.05, 
            help="Tasa de interés libre de riesgo.", 
            key="black_scholes_r"
        )
    with col5:
        sigma = st.slider(
            "Volatilidad (σ)", 
            0.1, 1.0, 0.2, 
            help="Volatilidad del activo subyacente.", 
            key="black_scholes_sigma"
        )

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
    fig_delta.update_layout(title="Δ Delta", xaxis_title="Precio del Activo (S)", yaxis_title="Delta", template="plotly_white")

    fig_gamma = go.Figure()
    fig_gamma.add_trace(go.Scatter(x=S_range, y=gamma_values, mode='lines', name='Gamma', line=dict(color='orange')))
    fig_gamma.update_layout(title="Γ Gamma", xaxis_title="Precio del Activo (S)", yaxis_title="Gamma", template="plotly_white")

    fig_theta = go.Figure()
    fig_theta.add_trace(go.Scatter(x=S_range, y=theta_values, mode='lines', name='Theta', line=dict(color='green')))
    fig_theta.update_layout(title="Θ Theta", xaxis_title="Precio del Activo (S)", yaxis_title="Theta", template="plotly_white")

    fig_vega = go.Figure()
    fig_vega.add_trace(go.Scatter(x=S_range, y=vega_values, mode='lines', name='Vega', line=dict(color='red')))
    fig_vega.update_layout(title="ν Vega", xaxis_title="Precio del Activo (S)", yaxis_title="Vega", template="plotly_white")

    fig_rho = go.Figure()
    fig_rho.add_trace(go.Scatter(x=S_range, y=rho_values, mode='lines', name='Rho', line=dict(color='purple')))
    fig_rho.update_layout(title="ρ Rho", xaxis_title="Precio del Activo (S)", yaxis_title="Rho", template="plotly_white")

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

    # Controles para los parámetros de la opción (siempre visibles)
    st.header("⚙️ Parámetros de la Opción")
    col1, col2, col3 = st.columns(3)
    with col1:
        S0 = st.slider(
            "Precio Actual del Activo (S₀)", 
            1.0, 200.0, 100.0, 
            help="Precio actual del activo subyacente.", 
            key="taylor_S0"
        )
    with col2:
        K = st.slider(
            "Precio de Ejercicio (K)", 
            1.0, 200.0, 100.0, 
            help="Precio al que se puede ejercer la opción.", 
            key="taylor_K"
        )
    with col3:
        T = st.slider(
            "Tiempo hasta Vencimiento (T)", 
            0.1, 5.0, 1.0, 
            help="Tiempo restante hasta el vencimiento de la opción.", 
            key="taylor_T"
        )

    col4, col5 = st.columns(2)
    with col4:
        r = st.slider(
            "Tasa Libre de Riesgo (r)", 
            0.0, 0.2, 0.05, 
            help="Tasa de interés libre de riesgo.", 
            key="taylor_r"
        )
    with col5:
        sigma = st.slider(
            "Volatilidad (σ)", 
            0.1, 1.0, 0.2, 
            help="Volatilidad del activo subyacente.", 
            key="taylor_sigma"
        )

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

    # Mostrar las ecuaciones de la expansión de Taylor dentro de un expander
    with st.expander("📝 Ecuaciones de la Expansión de Taylor"):
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

    # Graficar la expansión de Taylor y el precio real de la opción (siempre visible)
    st.subheader("📊 Gráfica de la Expansión de Taylor")

    # Crear la figura con Plotly
    fig = go.Figure()

    # Precio real de la opción
    fig.add_trace(go.Scatter(
        x=S_range,
        y=call_prices,
        mode='lines',
        name='Precio Real de la Opción',
        line=dict(color='blue', width=2)
    ))

    # Aproximación de Taylor de primer grado
    fig.add_trace(go.Scatter(
        x=S_range,
        y=taylor_1_values,
        mode='lines',
        name='Taylor Primer Orden',
        line=dict(color='green', dash='dash', width=2)
    ))

    # Aproximación de Taylor de segundo grado
    fig.add_trace(go.Scatter(
        x=S_range,
        y=taylor_2_values,
        mode='lines',
        name='Taylor Segundo Orden',
        line=dict(color='red', dash='dash', width=2)
    ))

    # Resaltar áreas donde el polinomio subestima o sobreestima
    fig.add_trace(go.Scatter(
        x=S_range,
        y=np.minimum(call_prices, taylor_1_values),  # Área donde Taylor 1 subestima
        fill='tonexty',
        mode='none',
        name='Taylor 1 Subestima',
        fillcolor='rgba(255, 0, 0, 0.1)',  # Rojo claro
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=S_range,
        y=np.maximum(call_prices, taylor_1_values),  # Área donde Taylor 1 sobreestima
        fill='tonexty',
        mode='none',
        name='Taylor 1 Sobrestima',
        fillcolor='rgba(0, 255, 0, 0.1)',  # Verde claro
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=S_range,
        y=np.minimum(call_prices, taylor_2_values),  # Área donde Taylor 2 subestima
        fill='tonexty',
        mode='none',
        name='Taylor 2 Subestima',
        fillcolor='rgba(255, 0, 0, 0.1)',  # Rojo claro
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=S_range,
        y=np.maximum(call_prices, taylor_2_values),  # Área donde Taylor 2 sobreestima
        fill='tonexty',
        mode='none',
        name='Taylor 2 Sobrestima',
        fillcolor='rgba(0, 255, 0, 0.1)',  # Verde claro
        showlegend=False
    ))

    # Línea vertical en el punto de expansión (S₀)
    fig.add_vline(
        x=S0,
        line=dict(color='gray', dash='dot'),
        annotation_text=f"S₀ = {S0}",
        annotation_position="top right"
    )

    # Configuración del layout
    fig.update_layout(
        title="Expansión de Taylor para una Opción Call",
        xaxis_title="Precio del Activo (S)",
        yaxis_title="Precio de la Opción",
        template="plotly_white",
        legend=dict(x=0.02, y=0.98),  # Posición de la leyenda
        margin=dict(l=20, r=20, t=40, b=20)
    )

    # Mostrar la gráfica
    st.plotly_chart(fig, use_container_width=True)

    # Explicación de las áreas
    st.markdown("""
    ### 🎯 Áreas de Subestimación y Sobrestimación
    - **Áreas en rojo claro:** Indican donde el polinomio de Taylor **subestima** el precio real de la opción.
    - **Áreas en verde claro:** Indican donde el polinomio de Taylor **sobrestima** el precio real de la opción.
    """)

    # Página de Optimización con Lagrange
    with tab5:
        st.title("🔍 Optimización con Método de Lagrange")
    
        # Descripción del método de Lagrange
        with st.expander("📚 ¿Qué es el Método de Lagrange?"):
            st.markdown("""
            **Método de Lagrange:**
            - El método de Lagrange se utiliza para encontrar los extremos de una función sujeta a restricciones.
            - Se introduce un multiplicador de Lagrange (\(\lambda\)) para incorporar la restricción en la función objetivo.
            - El sistema de ecuaciones se resuelve para encontrar los valores óptimos de \(x\), \(y\) y \(\lambda\).
            """)
    
        # Entrada de la función objetivo y la restricción
        st.header("⚙️ Ingresa la Función Objetivo y la Restricción")
        col1, col2 = st.columns(2)
        with col1:
            funcion_objetivo = st.text_input(
                "Función Objetivo (f(x, y)):", 
                "x**2 + y**2", 
                key="lagrange_funcion_objetivo",
                help="Ingresa una función válida de x e y."
            )
        with col2:
            restriccion = st.text_input(
                "Restricción (g(x, y) = 0):", 
                "x + y - 1", 
                key="lagrange_restriccion",
                help="Ingresa una restricción válida de x e y."
            )
    
        # Definir las variables simbólicas
        x, y, lambda_ = sp.symbols('x y lambda')
    
        try:
            # Convertir las entradas del usuario en funciones simbólicas
            f = sp.sympify(funcion_objetivo)
            g = sp.sympify(restriccion)
    
            # Construir la función de Lagrange
            L = f - lambda_ * g
    
            # Calcular las derivadas parciales
            dL_dx = sp.diff(L, x)
            dL_dy = sp.diff(L, y)
            dL_dlambda = sp.diff(L, lambda_)
    
            # Mostrar las derivadas parciales
            st.subheader("📝 Derivadas Parciales")
            st.latex(f"\\frac{{\\partial L}}{{\\partial x}} = {sp.latex(dL_dx)}")
            st.latex(f"\\frac{{\\partial L}}{{\\partial y}} = {sp.latex(dL_dy)}")
            st.latex(f"\\frac{{\\partial L}}{{\\partial \\lambda}} = {sp.latex(dL_dlambda)}")
    
            # Resolver el sistema de ecuaciones
            st.subheader("🔍 Solución del Sistema de Ecuaciones")
            soluciones = sp.solve([dL_dx, dL_dy, dL_dlambda], (x, y, lambda_), dict=True)
    
            if soluciones:
                for i, sol in enumerate(soluciones):
                    st.markdown(f"**Solución {i + 1}:**")
                    st.latex(f"x = {sp.latex(sol[x])}")
                    st.latex(f"y = {sp.latex(sol[y])}")
                    st.latex(f"\\lambda = {sp.latex(sol[lambda_])}")
    
                    # Evaluar la función objetivo en la solución
                    valor_optimo = f.subs({x: sol[x], y: sol[y]})
                    st.markdown(f"**Valor Óptimo de la Función Objetivo:** `{valor_optimo:.4f}`")
            else:
                st.error("No se encontraron soluciones para el sistema de ecuaciones.")
    
        except Exception as e:
            st.error(f"Error al procesar la función o la restricción: {e}")

    # Página de Paridad Put-Call
    with tab6:
        st.title("📉 Valor de un Put usando Paridad Put-Call")

        # Descripción de la Paridad Put-Call
        with st.expander("📚 ¿Qué es la Paridad Put-Call?"):
            st.markdown("""
            **Paridad Put-Call:**
            - La paridad put-call es una relación entre el precio de una opción call y una opción put con el mismo precio de ejercicio y fecha de vencimiento.
            - La fórmula de paridad put-call es:
              \[
              C + K e^{-rT} = P + S
              \]
              Donde:
              - \(C\): Precio de la opción call.
              - \(P\): Precio de la opción put.
              - \(S\): Precio del activo subyacente.
              - \(K\): Precio de ejercicio.
              - \(r\): Tasa libre de riesgo.
              - \(T\): Tiempo hasta el vencimiento.
            - Esta relación se utiliza para calcular el precio de una opción put si se conoce el precio de la opción call, o viceversa.
            """)

        # Entrada de parámetros
        st.header("⚙️ Parámetros de la Opción")
        col1, col2 = st.columns(2)
        with col1:
            S = st.number_input(
                "Precio del Activo (S)", 
                value=100.0, 
                min_value=0.01, 
                key="put_call_parity_S",
                help="Precio actual del activo subyacente."
            )
            K = st.number_input(
                "Precio de Ejercicio (K)", 
                value=100.0, 
                min_value=0.01, 
                key="put_call_parity_K",
                help="Precio al que se puede ejercer la opción."
            )
            T = st.number_input(
                "Tiempo hasta Vencimiento (T)", 
                value=1.0, 
                min_value=0.01, 
                key="put_call_parity_T",
                help="Tiempo restante hasta el vencimiento de la opción."
            )
        with col2:
            r = st.number_input(
                "Tasa Libre de Riesgo (r)", 
                value=0.05, 
                min_value=0.0, 
                key="put_call_parity_r",
                help="Tasa de interés libre de riesgo."
            )
            C = st.number_input(
                "Precio de la Opción Call (C)", 
                value=10.0, 
                min_value=0.0, 
                key="put_call_parity_C",
                help="Precio de la opción call."
            )

        # Calcular el precio de la opción put usando la paridad put-call
        def calcular_put_call_parity(S, K, T, r, C):
            P = C + K * np.exp(-r * T) - S
            return P

        # Calcular el precio de la opción put
        P = calcular_put_call_parity(S, K, T, r, C)

        # Mostrar el resultado
        st.subheader("💵 Precio de la Opción Put")
        st.markdown(f"**Precio de la Opción Put (P):** `{P:.4f}`")

        # Mostrar la fórmula de paridad put-call
        st.subheader("📝 Fórmula de Paridad Put-Call")
        st.latex(r"""
        P = C + K e^{-rT} - S
        """)

        # Explicación del cálculo
        st.markdown("""
        ### 🎯 Explicación del Cálculo
        - **Precio de la Opción Call (C):** Precio de la opción call proporcionado por el usuario.
        - **Precio del Activo (S):** Precio actual del activo subyacente.
        - **Precio de Ejercicio (K):** Precio al que se puede ejercer la opción.
        - **Tasa Libre de Riesgo (r):** Tasa de interés libre de riesgo.
        - **Tiempo hasta Vencimiento (T):** Tiempo restante hasta el vencimiento de la opción.
        - **Precio de la Opción Put (P):** Precio calculado de la opción put usando la fórmula de paridad put-call.
        """)
# Agregar una nueva pestaña para Monte Carlo
with tab7:
    st.title("🎲 Simulación de Monte Carlo para Opciones")

    # Descripción de la simulación de Monte Carlo
    with st.expander("📚 ¿Qué es la Simulación de Monte Carlo?"):
        st.markdown("""
        **Simulación de Monte Carlo:**
        - La simulación de Monte Carlo es un método numérico que utiliza muestreo aleatorio para estimar el valor de una opción.
        - Se generan múltiples trayectorias de precios del activo subyacente y se calcula el valor de la opción como el valor esperado de los pagos descontados.
        """)

    # Entrada de parámetros
    st.header("⚙️ Parámetros de la Simulación")
    col1, col2 = st.columns(2)
    with col1:
        S_mc = st.number_input(
            "Precio del Activo (S)", 
            value=100.0, 
            min_value=0.01, 
            key="mc_S",
            help="Precio actual del activo subyacente."
        )
        K_mc = st.number_input(
            "Precio de Ejercicio (K)", 
            value=100.0, 
            min_value=0.01, 
            key="mc_K",
            help="Precio al que se puede ejercer la opción."
        )
        T_mc = st.number_input(
            "Tiempo hasta Vencimiento (T)", 
            value=1.0, 
            min_value=0.01, 
            key="mc_T",
            help="Tiempo restante hasta el vencimiento de la opción."
        )
    with col2:
        r_mc = st.number_input(
            "Tasa Libre de Riesgo (r)", 
            value=0.05, 
            min_value=0.0, 
            key="mc_r",
            help="Tasa de interés libre de riesgo."
        )
        sigma_mc = st.number_input(
            "Volatilidad (σ)", 
            value=0.2, 
            min_value=0.01, 
            key="mc_sigma",
            help="Volatilidad del activo subyacente."
        )
        simulations = st.number_input(
            "Número de Simulaciones", 
            value=1000, 
            min_value=100, 
            key="mc_simulations",
            help="Número de trayectorias de precios a generar."
        )
        steps = st.number_input(
            "Número de Pasos de Tiempo", 
            value=100, 
            min_value=10, 
            key="mc_steps",
            help="Número de pasos de tiempo para cada trayectoria."
        )

    # Función para simular trayectorias de precios
    def monte_carlo_simulation(S, K, T, r, sigma, simulations, steps):
        dt = T / steps
        paths = np.zeros((steps + 1, simulations))
        paths[0] = S

        for t in range(1, steps + 1):
            z = np.random.standard_normal(simulations)
            paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)

        return paths

    # Función para calcular el precio de la opción call
    def monte_carlo_call_price(paths, K, r, T):
        payoffs = np.maximum(paths[-1] - K, 0)
        call_price = np.exp(-r * T) * np.mean(payoffs)
        return call_price, payoffs

    # Ejecutar la simulación
    if st.button("Ejecutar Simulación"):
        paths = monte_carlo_simulation(S_mc, K_mc, T_mc, r_mc, sigma_mc, simulations, steps)
        call_price, payoffs = monte_carlo_call_price(paths, K_mc, r_mc, T_mc)

        # Mostrar resultados
        st.subheader("📊 Resultados de la Simulación")
        st.markdown(f"**Precio de la Opción Call:** `{call_price:.4f}`")

        # Gráfico de trayectorias de precios
        st.subheader("📈 Trayectorias de Precios Simuladas")
        fig_paths = go.Figure()
        for i in range(min(100, simulations)):  # Mostrar solo 100 trayectorias para claridad
            fig_paths.add_trace(go.Scatter(
                x=np.arange(steps + 1),
                y=paths[:, i],
                mode='lines',
                line=dict(width=1),
                name=f"Trayectoria {i + 1}"
            ))
        fig_paths.update_layout(
            title="Trayectorias de Precios Simuladas",
            xaxis_title="Pasos de Tiempo",
            yaxis_title="Precio del Activo",
            template="plotly_white"
        )
        st.plotly_chart(fig_paths, use_container_width=True)

        # Histograma de pagos
        st.subheader("📊 Histograma de Pagos")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=payoffs,
            nbinsx=50,
            marker=dict(color='blue'),
            name="Pagos"
        ))
        fig_hist.update_layout(
            title="Distribución de Pagos",
            xaxis_title="Pago",
            yaxis_title="Frecuencia",
            template="plotly_white"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
with tab8:
    # Título y descripción
    st.title("📊 Explicación Gráfica de la Aproximación de Taylor")
    st.markdown("""
    Esta herramienta te permite visualizar cómo el polinomio de Taylor de primer y segundo grado aproxima una función alrededor de un punto \( x_0 \).
    Explora cómo la aproximación subestima o sobreestima la función dependiendo de la concavidad y el valor de \( \Dx \).
    """)

    # Entrada de la función
    st.header("⚙️ Ingresa una función")
    function_input = st.text_input(
        "Ingresa una función de \( x \) (por ejemplo, `x**2`, `sin(x)`, `exp(x)`):", 
        "ln(x)", 
        key="taylor_explanation_function_input",
        help="Ingresa una función válida de \( x \). Usa '**' para exponenciación (por ejemplo, 'x**2')."
    )

    # Reemplazar '^' por '**' en la función ingresada
    function_input = function_input.replace("^", "**")

    # Configuración del gráfico en columnas
    st.header("⚙️ Configuración del gráfico")
    col1, col2, col3 = st.columns(3)
    with col1:
        x0 = st.number_input(
            "Punto de expansión \( x_0 \):", 
            value=10.0, 
            format="%.4f", 
            key="taylor_explanation_x0_input",
            help="Punto alrededor del cual se calculará la expansión de Taylor."
        )
    with col2:
        x_min = st.number_input(
            "Límite inferior de \( x \):", 
            value=5.0, 
            format="%.4f", 
            key="taylor_explanation_x_min_input",
            help="Valor mínimo de \( x \) para el gráfico."
        )
    with col3:
        x_max = st.number_input(
            "Límite superior de \( x \):", 
            value=15.0, 
            format="%.4f", 
            key="taylor_explanation_x_max_input",
            help="Valor máximo de \( x \) para el gráfico."
        )

    # Definir la variable simbólica
    x = sp.symbols('x')

    try:
        # Convertir la entrada del usuario en una función simbólica
        f = sp.sympify(function_input)

        # Verificar si la función depende de x
        if not f.has(x):
            st.error("La función ingresada no depende de \( x \). Ingresa una función válida de \( x \).")
            st.stop()

        # Calcular las derivadas
        f_prime = sp.diff(f, x)  # Primera derivada
        f_double_prime = sp.diff(f_prime, x)  # Segunda derivada

        # Expansión de Taylor de grado 1
        taylor_1 = f.subs(x, x0) + f_prime.subs(x, x0) * (x - x0)

        # Expansión de Taylor de grado 2
        taylor_2 = taylor_1 + (f_double_prime.subs(x, x0) / 2) * (x - x0)**2

        # Convertir las funciones simbólicas a funciones numéricas
        f_np = sp.lambdify(x, f, "numpy")
        taylor_1_np = sp.lambdify(x, taylor_1, "numpy")
        taylor_2_np = sp.lambdify(x, taylor_2, "numpy")
        f_prime_np = sp.lambdify(x, f_prime, "numpy")
        f_double_prime_np = sp.lambdify(x, f_double_prime, "numpy")

        # Crear un rango de valores para x
        x_vals = np.linspace(x_min, x_max, 500)

        # Evaluar las funciones en el rango de x
        try:
            y_vals = f_np(x_vals)
            y_taylor_1 = taylor_1_np(x_vals)
            y_taylor_2 = taylor_2_np(x_vals)
            y_prime = f_prime_np(x_vals)
            y_double_prime = f_double_prime_np(x_vals)
        except Exception as e:
            st.error(f"Error al evaluar la función: {e}")
            st.stop()

        # Graficar la función original y las aproximaciones de Taylor
        st.subheader("📊 Aproximación de Taylor")
        fig_taylor = go.Figure()

        # Función original
        fig_taylor.add_trace(go.Scatter(
            x=x_vals, 
            y=y_vals, 
            mode='lines', 
            name=f"Función: {function_input}", 
            line=dict(color='blue', width=2)
        ))

        # Taylor de primer grado
        fig_taylor.add_trace(go.Scatter(
            x=x_vals, 
            y=y_taylor_1, 
            mode='lines', 
            name="Taylor Grado 1", 
            line=dict(color='green', dash='dash', width=2)
        ))

        # Taylor de segundo grado
        fig_taylor.add_trace(go.Scatter(
            x=x_vals, 
            y=y_taylor_2, 
            mode='lines', 
            name="Taylor Grado 2", 
            line=dict(color='red', dash='dash', width=2)
        ))

        # Línea vertical en el punto de expansión
        fig_taylor.add_vline(
            x=x0, 
            line=dict(color='gray', dash='dot'), 
            annotation_text=f"x₀ = {x0}", 
            annotation_position="top right"
        )

        # Resaltar áreas de subestimación y sobreestimación
        fig_taylor.add_trace(go.Scatter(
            x=x_vals, 
            y=np.minimum(y_vals, y_taylor_1),  # Área donde Taylor 1 subestima
            fill='tonexty',
            mode='none',
            name='Taylor 1 Subestima',
            fillcolor='rgba(255, 0, 0, 0.1)',  # Rojo claro
            showlegend=False
        ))

        fig_taylor.add_trace(go.Scatter(
            x=x_vals, 
            y=np.maximum(y_vals, y_taylor_1),  # Área donde Taylor 1 sobreestima
            fill='tonexty',
            mode='none',
            name='Taylor 1 Sobrestima',
            fillcolor='rgba(0, 255, 0, 0.1)',  # Verde claro
            showlegend=False
        ))

        fig_taylor.update_layout(
            title="Aproximación de Taylor",
            xaxis_title="x",
            yaxis_title="f(x)",
            template="plotly_white",
            legend=dict(x=0.02, y=0.98),
            hovermode="x unified"  # Tooltip unificado
        )
        st.plotly_chart(fig_taylor, use_container_width=True)

        # Gráficas de la función original, primera derivada y segunda derivada en una fila
        st.subheader("📊 Función y Derivadas")
        col1, col2, col3 = st.columns(3)

        # Gráfica de la función original
        with col1:
            fig_original = go.Figure()
            fig_original.add_trace(go.Scatter(
                x=x_vals, 
                y=y_vals, 
                mode='lines', 
                name=f"Función: {function_input}", 
                line=dict(color='blue', width=2)
            ))
            fig_original.update_layout(
                title=f"Función Original: {function_input}",
                xaxis_title="x",
                yaxis_title="f(x)",
                template="plotly_white"
            )
            st.plotly_chart(fig_original, use_container_width=True)

        # Gráfica de la primera derivada
        with col2:
            fig_prime = go.Figure()
            fig_prime.add_trace(go.Scatter(
                x=x_vals, 
                y=y_prime, 
                mode='lines', 
                name=f"Primera Derivada: {sp.latex(f_prime)}", 
                line=dict(color='purple', width=2)
            ))
            fig_prime.update_layout(
                title=f"Primera Derivada: {sp.latex(f_prime)}",
                xaxis_title="x",
                yaxis_title="f'(x)",
                template="plotly_white"
            )
            st.plotly_chart(fig_prime, use_container_width=True)

        # Gráfica de la segunda derivada
        with col3:
            fig_double_prime = go.Figure()
            fig_double_prime.add_trace(go.Scatter(
                x=x_vals, 
                y=y_double_prime, 
                mode='lines', 
                name=f"Segunda Derivada: {sp.latex(f_double_prime)}", 
                line=dict(color='orange', width=2)
            ))
            fig_double_prime.update_layout(
                title=f"Segunda Derivada: {sp.latex(f_double_prime)}",
                xaxis_title="x",
                yaxis_title="f''(x)",
                template="plotly_white"
            )
            st.plotly_chart(fig_double_prime, use_container_width=True)

        # Explicación de subestimación y sobreestimación
        with st.expander("📚 ¿Por qué el polinomio de Taylor subestima o sobreestima?"):
            st.markdown("""
            ### Subestimación y Sobrestimación
            - **Subestimación:** Cuando \( \Dx > 0 \) y la función es cóncava hacia arriba (\( f''(x_0) > 0 \)), el polinomio de Taylor de primer grado subestima la función.
            - **Sobrestimación:** Cuando \( \Dx < 0 \) y la función es cóncava hacia arriba (\( f''(x_0) > 0 \)), el polinomio de Taylor de primer grado sobreestima la función.
            - **Corrección cuadrática:** El polinomio de segundo grado corrige esta subestimación o sobreestimación al incluir la curvatura de la función.
            """)

            # Tabla resumen
            st.markdown("""
            ### Resumen
            | Condición               | Comportamiento del Polinomio de Taylor |
            |-------------------------|----------------------------------------|
            | \( \Dx > 0 \) y \( f''(x_0) > 0 \) | Subestima la función |
            | \( \Dx < 0 \) y \( f''(x_0) > 0 \) | Sobrestima la función |
            """)

        # Feedback al usuario
        st.success("¡Gráfica generada con éxito! Explora cómo el polinomio de Taylor aproxima la función.")

    except Exception as e:
        st.error(f"Error al procesar la función: {e}")

# Agregar una nueva pestaña para el cálculo de la prima
with tab9:
    st.title("💰 Cálculo de Prima")

    # Descripción del cálculo de la prima
    with st.expander("📚 ¿Qué es el Cálculo de Prima?"):
        st.markdown("""
        **Cálculo de Prima:**
        - La prima es el costo asociado a un seguro o cobertura.
        - En este caso, se utiliza la siguiente ecuación para calcular la prima:
          \[
          \pi \cdot \ln(\text{Riqueza}_{\text{bueno}}) + (1 - \pi) \cdot \ln(\text{Riqueza}_{\text{malo}}) = \ln(\text{Riqueza} - \text{Prima})
          \]
          Donde:
          - \(\pi\): Probabilidad del caso bueno.
          - \(\text{Riqueza}_{\text{bueno}}\): Riqueza en el caso bueno.
          - \(\text{Riqueza}_{\text{malo}}\): Riqueza en el caso malo.
          - \(\text{Riqueza}\): Riqueza inicial.
          - \(\text{Prima}\): Prima a calcular.
        """)

    # Entrada de parámetros
    st.header("⚙️ Parámetros del Cálculo")
    col1, col2 = st.columns(2)
    with col1:
        riqueza_inicial = st.number_input(
            "Riqueza Inicial", 
            value=100.0, 
            min_value=0.01, 
            key="prima_riqueza_inicial",
            help="Riqueza inicial del individuo."
        )
        ganancia_bueno = st.number_input(
            "Ganancia en el Caso Bueno", 
            value=20.0, 
            min_value=0.0, 
            key="prima_ganancia_bueno",
            help="Ganancia en el caso bueno."
        )
    with col2:
        perdida_malo = st.number_input(
            "Pérdida en el Caso Malo", 
            value=30.0, 
            min_value=0.0, 
            key="prima_perdida_malo",
            help="Pérdida en el caso malo."
        )
        pi = st.number_input(
            "Probabilidad del Caso Bueno (\(\pi\))", 
            value=0.6, 
            min_value=0.0, 
            max_value=1.0, 
            key="prima_pi",
            help="Probabilidad del caso bueno."
        )

    # Función para calcular la prima
    def calcular_prima(riqueza_inicial, ganancia_bueno, perdida_malo, pi):
        # Calcular riqueza en el caso bueno y malo
        riqueza_bueno = riqueza_inicial + ganancia_bueno
        riqueza_malo = riqueza_inicial - perdida_malo

        # Calcular el lado izquierdo de la ecuación
        lado_izquierdo = pi * np.log(riqueza_bueno) + (1 - pi) * np.log(riqueza_malo)

        # Resolver para la prima
        prima = riqueza_inicial - np.exp(lado_izquierdo)

        return prima

    # Calcular la prima
    prima = calcular_prima(riqueza_inicial, ganancia_bueno, perdida_malo, pi)

    # Mostrar el resultado
    st.subheader("💵 Resultado del Cálculo")
    st.markdown(f"**Valor de la Prima:** `{prima:.4f}`")

    # Explicación del cálculo
    st.markdown("""
    ### 🎯 Explicación del Cálculo
    
    - **Riqueza en el Caso Bueno:**  
      Riqueza_bueno = Riqueza Inicial + Ganancia en el Caso Bueno
    
    - **Riqueza en el Caso Malo:**  
      Riqueza_malo = Riqueza Inicial - Pérdida en el Caso Malo
    
    - **Ecuación de Prima:**  
      π * ln(Riqueza_bueno) + (1 - π) * ln(Riqueza_malo) = ln(Riqueza - Prima)
    
    - **Prima Calculada:**  
      Prima = Riqueza Inicial - e^(Lado Izquierdo)
    """)


st.markdown("---")
st.markdown("""
**Creado por:** Facundo Maleh  
**Nota:** Esta aplicación es solo para fines educativos.
""")
