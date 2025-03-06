import streamlit as st
import numpy as np
import plotly.graph_objects as go

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

# Function to create a Plotly tree
def create_plotly_tree(values, title, color="blue"):
    fig = go.Figure()
    for i in range(values.shape[0]):
        for j in range(i + 1):
            # Add nodes
            fig.add_trace(go.Scatter(
                x=[i], y=[j - i / 2],
                mode="markers+text",
                marker=dict(size=20, color=color),
                text=[f"{values[i, j]:.2f}"],
                textposition="middle center",
                name=f"Node ({i}, {j})"
            ))
            # Add edges
            if i > 0:
                parent_j = j if j < i else j - 1
                fig.add_trace(go.Scatter(
                    x=[i - 1, i], y=[parent_j - (i - 1) / 2, j - i / 2],
                    mode="lines",
                    line=dict(color="gray", width=2),
                    hoverinfo="none"
                ))
    fig.update_layout(
        title=title,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        plot_bgcolor="#f7f7f7",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# Display Binomial Trees
st.subheader("📊 Binomial Trees")
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(create_plotly_tree(asset_prices, "Asset Price Tree", color="#1f77b4"), use_container_width=True)
    st.plotly_chart(create_plotly_tree(option_prices, "Option Price Tree", color="#ff7f0e"), use_container_width=True)
with col2:
    st.plotly_chart(create_plotly_tree(deltas, "Delta (Δ) Tree", color="#2ca02c"), use_container_width=True)
    st.plotly_chart(create_plotly_tree(debts, "Debt (B) Tree", color="#d62728"), use_container_width=True)

# Display Final Option Price
st.markdown(f"**Call Option Price:** `{option_prices[0, 0]:.4f}`")

# Footer
st.markdown("---")
st.markdown("""
**Created by:** Facundo Maleh  
**Note:** This app is for educational purposes only.
""")
