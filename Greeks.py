import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import norm  # Importar norm desde scipy.stats
import sympy as sp


# Configuración de la página (DEBE SER LA PRIMERA LÍNEA DE STREAMLIT)
st.set_page_config(
    layout="wide",
    page_title="Visualizador de Black-Scholes, Taylor y Binomial",
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
st.title("Visualizador de Black-Scholes, Taylor y Binomial")
theme = st.toggle("Modo Oscuro", value=st.session_state.get("theme", "light") == "dark", on_change=toggle_theme)
apply_theme()

# Menú de navegación con pestañas
tab1, tab2, tab3 = st.tabs(["📈 Black-Scholes", "📊 Aproximación de Taylor", "🌳 Árbol Binomial"])

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

    # Gráficos de las letras griegas
    st.subheader("📊 Gráficas de las Letras Griegas")
    S_range = np.linspace(1, 200, 100)
    delta_values = delta_call(S_range, K, T, r, sigma)
    gamma_values = gamma_call(S_range, K, T, r, sigma)
    theta_values = theta_call(S_range, K, T, r, sigma)
    vega_values = vega_call(S_range, K, T, r, sigma)
    rho_values = rho_call(S_range, K, T, r, sigma)

    # Organizar los gráficos en 5 columnas (una para cada gráfico)
    cols = st.columns(5)  # Cambia a 5 columnas

    with cols[0]:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(S_range, delta_values, label='Delta', color='blue')
        ax1.set_title('Δ Delta')
        ax1.set_xlabel('Precio del Activo (S)')
        ax1.set_ylabel('Delta')
        ax1.grid(True)
        st.pyplot(fig1)

    with cols[1]:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(S_range, gamma_values, label='Gamma', color='orange')
        ax2.set_title('Γ Gamma')
        ax2.set_xlabel('Precio del Activo (S)')
        ax2.set_ylabel('Gamma')
        ax2.grid(True)
        st.pyplot(fig2)

    with cols[2]:
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.plot(S_range, theta_values, label='Theta', color='green')
        ax3.set_title('Θ Theta')
        ax3.set_xlabel('Precio del Activo (S)')
        ax3.set_ylabel('Theta')
        ax3.grid(True)
        st.pyplot(fig3)

    with cols[3]:
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        ax4.plot(S_range, vega_values, label='Vega', color='red')
        ax4.set_title('ν Vega')
        ax4.set_xlabel('Precio del Activo (S)')
        ax4.set_ylabel('Vega')
        ax4.grid(True)
        st.pyplot(fig4)

    with cols[4]:
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        ax5.plot(S_range, rho_values, label='Rho', color='purple')
        ax5.set_title('ρ Rho')
        ax5.set_xlabel('Precio del Activo (S)')
        ax5.set_ylabel('Rho')
        ax5.grid(True)
        st.pyplot(fig5)

    # Mostrar el valor de la opción y las letras griegas en una sola fila
    st.subheader("💵 Valor de la Opción Call y Letras Griegas")

    # Crear 6 columnas: 1 para el precio de la opción y 5 para las letras griegas
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    # Precio de la opción Call
    with col1:
        st.metric("Precio de la Opción Call", f"{call_price:.4f}")

    # Valores de las letras griegas
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
        x0 = st.slider(
            "Punto de expansión (x0)",
            min_value=-15.0,
            max_value=15.0,
            value=0.01,
            step=0.1,
            help="Selecciona el punto alrededor del cual se calculará la expansión de Taylor."
        )
    with col2:
        x_min = st.slider(
            "Límite inferior de x",
            min_value=-15.0,
            max_value=15.0,
            value=-5.0,
            step=0.1,
            help="Define el valor mínimo de x para el gráfico."
        )
    with col3:
        x_max = st.slider(
            "Límite superior de x",
            min_value=-15.0,
            max_value=15.0,
            value=5.0,
            step=0.1,
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
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_vals, y_vals, label=f"Función: {function_input}", color='blue')
        ax.plot(x_vals, y_taylor_1, label="Taylor Grado 1", color='green', linestyle='--')
        ax.plot(x_vals, y_taylor_2, label="Taylor Grado 2", color='red', linestyle='--')
        ax.axvline(x=x0, color='gray', linestyle=':', label=f"Punto de expansión (x0 = {x0})")
        ax.set_title("Aproximación de Taylor")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error al procesar la función: {e}")

# Página de Árbol Binomial
with tab3:
    st.title("🌳 Valuación de Opciones con Árbol Binomial")

    # Descripción del modelo de árbol binomial
    st.markdown("""
    **Modelo de Árbol Binomial:**
    - Este modelo permite valuar una opción call utilizando un árbol binomial.
    - Se calcula el precio de la opción hacia atrás (backwards) y se muestra la proporción de delta y deuda en cada nodo.
    """)

    # Entrada de parámetros
    st.header("⚙️ Parámetros del Modelo")
    col1, col2 = st.columns(2)
    with col1:
        S = st.number_input("Precio del Activo (S)", value=10.0, min_value=0.01)
        K = st.number_input("Precio de Ejercicio (K)", value=10.0, min_value=0.01)
        U = st.number_input("Factor de Subida (U)", value=2.0, min_value=1.0)
    with col2:
        D = st.number_input("Factor de Bajada (D)", value=0.5, max_value=1.0)
        R = st.number_input("Factor de Capitalización (R = 1 + Rf)", value=1.0, min_value=1.0)
        periods = st.number_input("Número de Periodos", value=2, min_value=1)

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
                pos[node] = (i, j - i / 2)
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

# Pie de página
st.markdown("---")
st.markdown("""
**Creado por:** Facundo Maleh  
**Nota:** Esta aplicación es solo para fines educativos.
""")
