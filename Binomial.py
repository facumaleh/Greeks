import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Page Configuration
st.set_page_config(
    layout="wide",
    page_title="Binomial Option Pricing",
    page_icon="🌳"
)

# Custom CSS for a modern look
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

# Title
st.title("🌳 Binomial Option Pricing Model")

# Explanation of the Binomial Model
with st.expander("📚 What is the Binomial Model?"):
    st.markdown("""
    **The Binomial Model** is a discrete-time model used to price options. It assumes that the price of the underlying asset can move up or down by a certain factor in each time step. The model is particularly useful for valuing **American options**, which can be exercised at any time before expiration.

    **Key Concepts:**
    - **Up Factor (U):** The factor by which the asset price increases.
    - **Down Factor (D):** The factor by which the asset price decreases.
    - **Risk-Neutral Probability (q):** The probability of an up move in a risk-neutral world.
    - **Backward Induction:** The process of calculating option prices by working backward from the final nodes to the initial node.

    **Connection to Black-Scholes:**
    - The Binomial Model converges to the Black-Scholes formula as the number of time steps increases.
    - Both models rely on the concept of **risk-neutral valuation** and **no-arbitrage pricing**.
    """)

# User Inputs
st.header("⚙️ Model Parameters")
col1, col2 = st.columns(2)
with col1:
    S = st.number_input("Asset Price (S)", value=100.0, min_value=0.01, key="binomial_S")
    K = st.number_input("Strike Price (K)", value=100.0, min_value=0.01, key="binomial_K")
    U = st.number_input("Up Factor (U)", value=1.1, min_value=1.0, key="binomial_U")
with col2:
    D = st.number_input("Down Factor (D)", value=0.9, max_value=1.0, key="binomial_D")
    R = st.number_input("Risk-Free Rate Factor (R = 1 + r)", value=1.05, min_value=1.0, key="binomial_R")
    periods = st.number_input("Number of Periods", value=3, min_value=1, key="binomial_periods")

# Binomial Tree Calculation
def binomial_tree_call(S, K, U, D, R, periods):
    q = (R - D) / (U - D)  # Risk-neutral probability
    asset_prices = np.zeros((periods + 1, periods + 1))
    asset_prices[0, 0] = S
    for i in range(1, periods + 1):
        asset_prices[i, 0] = asset_prices[i - 1, 0] * U
        for j in range(1, i + 1):
            asset_prices[i, j] = asset_prices[i - 1, j - 1] * D

    option_prices = np.zeros((periods + 1, periods + 1))
    for j in range(periods + 1):
        option_prices[periods, j] = max(0, asset_prices[periods, j] - K)

    for i in range(periods - 1, -1, -1):
        for j in range(i + 1):
            option_prices[i, j] = (q * option_prices[i + 1, j] + (1 - q) * option_prices[i + 1, j + 1]) / R

    deltas = np.zeros((periods, periods + 1))
    debts = np.zeros((periods, periods + 1))
    for i in range(periods):
        for j in range(i + 1):
            deltas[i, j] = (option_prices[i + 1, j] - option_prices[i + 1, j + 1]) / (asset_prices[i + 1, j] - asset_prices[i + 1, j + 1])
            debts[i, j] = (option_prices[i + 1, j + 1] * U - option_prices[i + 1, j] * D) / (R * (U - D))

    return asset_prices, option_prices, deltas, debts

# Calculate Binomial Tree
asset_prices, option_prices, deltas, debts = binomial_tree_call(S, K, U, D, R, periods)

# Improved Visualization Function
def plot_binomial_tree(values, title, ax, color="blue"):
    G = nx.Graph()
    pos = {}
    labels = {}
    for i in range(values.shape[0]):
        for j in range(i + 1):
            node = (i, j)
            G.add_node(node)
            pos[node] = (i, -j + i / 2)  # Adjust vertical position
            labels[node] = f"{values[i, j]:.2f}"
            if i > 0:
                parent = (i - 1, j) if j < i else (i - 1, j - 1)
                G.add_edge(parent, node)

    nx.draw(G, pos, labels=labels, with_labels=True, node_size=2000, node_color=color, font_size=10, font_weight="bold", ax=ax, edge_color="gray", width=1.5)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_facecolor("#f7f7f7")  # Light gray background
    ax.grid(False)  # Disable grid

# Display Binomial Trees
st.subheader("📊 Binomial Trees")
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Binomial Trees", fontsize=16, fontweight="bold")

plot_binomial_tree(asset_prices, "Asset Price Tree", axs[0, 0], color="#1f77b4")
plot_binomial_tree(option_prices, "Option Price Tree", axs[0, 1], color="#ff7f0e")
plot_binomial_tree(deltas, "Delta (Δ) Tree", axs[1, 0], color="#2ca02c")
plot_binomial_tree(debts, "Debt (B) Tree", axs[1, 1], color="#d62728")

plt.tight_layout()
st.pyplot(fig)

# Display Final Option Price
st.markdown(f"**Call Option Price:** `{option_prices[0, 0]:.4f}`")

# Footer
st.markdown("---")
st.markdown("""
**Created by:** Facundo Maleh  
**Note:** This app is for educational purposes only.
""")
