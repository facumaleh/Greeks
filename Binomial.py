import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Configuración de la página
st.set_page_config(
    layout="wide",
    page_title="Enjoy Finance - Árbol Binomial",
    page_icon="🌳"
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

# Función para graficar un árbol binomial con mejor diseño
def plot_binomial_tree(values, title, ax, color="blue"):
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

    # Dibujar el árbol
    nx.draw(G, pos, labels=labels, with_labels=True, node_size=2000, node_color=color, font_size=10, font_weight="bold", ax=ax, edge_color="gray", width=1.5)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_facecolor("#f7f7f7")  # Fondo gris claro
    ax.grid(False)  # Desactivar la cuadrícula

# Mostrar los árboles binomiales en una cuadrícula
st.subheader("📊 Árboles Binomiales")
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Árboles Binomiales", fontsize=16, fontweight="bold")

# Árbol de Precios del Activo
plot_binomial_tree(asset_prices, "Árbol de Precios del Activo", axs[0, 0], color="#1f77b4")

# Árbol de Precios de la Opción Call
plot_binomial_tree(option_prices, "Árbol de Precios de la Opción Call", axs[0, 1], color="#ff7f0e")

# Árbol de Deltas (Δ)
plot_binomial_tree(deltas, "Árbol de Deltas (Δ)", axs[1, 0], color="#2ca02c")

# Árbol de Deudas (B)
plot_binomial_tree(debts, "Árbol de Deudas (B)", axs[1, 1], color="#d62728")

# Ajustar el layout
plt.tight_layout()
st.pyplot(fig)

# Mostrar el precio final de la opción
st.markdown(f"**Precio de la Opción Call:** `{option_prices[0, 0]:.4f}`")

# Pie de página
st.markdown("---")
st.markdown("""
**Creado por:** Facundo Maleh  
**Nota:** Esta aplicación es solo para fines educativos.
""")
