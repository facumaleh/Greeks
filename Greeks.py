# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Título de la aplicación
st.title("Visualizador de Letras Griegas en Black-Scholes")
st.markdown("""
Esta aplicación te permite visualizar cómo cambian las letras griegas (Delta, Gamma, Theta, Vega, Rho) 
en la fórmula de Black-Scholes para una opción call.
""")

# Sidebar para los parámetros
st.sidebar.header("Parámetros de la Opción")
S = st.sidebar.slider("Precio del Activo (S)", 1.0, 200.0, 100.0)
moneyness = st.sidebar.selectbox("Moneyness", ["ATM", "ITM", "OTM"])
T = st.sidebar.slider("Tiempo hasta vencimiento (T)", 0.1, 5.0, 1.0)
r = st.sidebar.slider("Tasa libre de riesgo (r)", 0.0, 0.2, 0.05)
sigma = st.sidebar.slider("Volatilidad (σ)", 
