# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm
# 
# # Título de la aplicación
# st.title("Visualizador de Letras Griegas en Black-Scholes")
# st.markdown("""
# Esta aplicación te permite visualizar cómo cambian las letras griegas (Delta, Gamma, Theta, Vega, Rho)
# en la fórmula de Black-Scholes para una opción call.
# """)
# 
# # Sidebar para los parámetros
# st.sidebar.header("Parámetros de la Opción")
# S = st.sidebar.slider("Precio del Activo (S)", 1.0, 200.0, 100.0)
# moneyness = st.sidebar.selectbox("Moneyness", ["ATM", "ITM", "OTM"])
# T = st.sidebar.slider("Tiempo hasta vencimiento (T)", 0.1, 5.0, 1.0)
# r = st.sidebar.slider("Tasa libre de riesgo (r)", 0.0, 0.2, 0.05)
# sigma = st.sidebar.slider("Volatilidad (σ)", 0.1, 1.0, 0.2)
# 
# # Ajustar K en función de la selección (ATM, ITM, OTM)
# def adjust_K(S, moneyness):
#     if moneyness == "ATM":
#         return S  # At the Money: K = S
#     elif moneyness == "ITM":
#         return S * 0.9  # In the Money: K < S
#     elif moneyness == "OTM":
#         return S * 1.1  # Out of the Money: K > S
#     else:
#         return S  # Por defecto, ATM
# 
# K = adjust_K(S, moneyness)
# 
# # Fórmula de Black-Scholes para una opción call
# def black_scholes_call(S, K, T, r, sigma):
#     d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
#     d2 = d1 - sigma * np.sqrt(T)
#     call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
#     return call_price
# 
# # Cálculo de las letras griegas
# def delta_call(S, K, T, r, sigma):
#     d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
#     return norm.cdf(d1)
# 
# def gamma_call(S, K, T, r, sigma):
#     d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
#     return norm.pdf(d1) / (S * sigma * np.sqrt(T))
# 
# def theta_call(S, K, T, r, sigma):
#     d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
#     d2 = d1 - sigma * np.sqrt(T)
#     return - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
# 
# def vega_call(S, K, T, r, sigma):
#     d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
#     return S * norm.pdf(d1) * np.sqrt(T)
# 
# def rho_call(S, K, T, r, sigma):
#     d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
#     d2 = d1 - sigma * np.sqrt(T)
#     return K * T * np.exp(-r * T) * norm.cdf(d2)
# 
# # Calcular las letras griegas
# delta = delta_call(S, K, T, r, sigma)
# gamma = gamma_call(S, K, T, r, sigma)
# theta = theta_call(S, K, T, r, sigma)
# vega = vega_call(S, K, T, r, sigma)
# rho = rho_call(S, K, T, r, sigma)
# 
# # Mostrar los valores de las letras griegas
# st.subheader("Valores de las Letras Griegas")
# col1, col2, col3, col4, col5 = st.columns(5)
# col1.metric("Delta", f"{delta:.4f}")
# col2.metric("Gamma", f"{gamma:.4f}")
# col3.metric("Theta", f"{theta:.4f}")
# col4.metric("Vega", f"{vega:.4f}")
# col5.metric("Rho", f"{rho:.4f}")
# 
# # Graficar las letras griegas
# st.subheader("Gráficas de las Letras Griegas")
# S_range = np.linspace(1, 200, 100)
# delta_values = delta_call(S_range, K, T, r, sigma)
# gamma_values = gamma_call(S_range, K, T, r, sigma)
# theta_values = theta_call(S_range, K, T, r, sigma)
# vega_values = vega_call(S_range, K, T, r, sigma)
# rho_values = rho_call(S_range, K, T, r, sigma)
# 
# fig, ax = plt.subplots(3, 2, figsize=(14, 10))
# ax[0, 0].plot(S_range, delta_values, label='Delta')
# ax[0, 0].set_title('Delta')
# ax[0, 1].plot(S_range, gamma_values, label='Gamma', color='orange')
# ax[0, 1].set_title('Gamma')
# ax[1, 0].plot(S_range, theta_values, label='Theta', color='green')
# ax[1, 0].set_title('Theta')
# ax[1, 1].plot(S_range, vega_values, label='Vega', color='red')
# ax[1, 1].set_title('Vega')
# ax[2, 0].plot(S_range, rho_values, label='Rho', color='purple')
# ax[2, 0].set_title('Rho')
# plt.tight_layout()
# st.pyplot(fig)

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
sigma = st.sidebar.slider("Volatilidad (σ)", 0.1, 1.0, 0.2)

# Ajustar K en función de la selección (ATM, ITM, OTM)
def adjust_K(S, moneyness):
    if moneyness == "ATM":
        return S  # At the Money: K = S
    elif moneyness == "ITM":
        return S * 0.9  # In the Money: K < S
    elif moneyness == "OTM":
        return S * 1.1  # Out of the Money: K > S
    else:
        return S  # Por defecto, ATM

K = adjust_K(S, moneyness)

# Fórmula de Black-Scholes para una opción call
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

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

# Mostrar los valores de las letras griegas
st.subheader("Valores de las Letras Griegas")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Delta", f"{delta:.4f}")
col2.metric("Gamma", f"{gamma:.4f}")
col3.metric("Theta", f"{theta:.4f}")
col4.metric("Vega", f"{vega:.4f}")
col5.metric("Rho", f"{rho:.4f}")

# Graficar las letras griegas
st.subheader("Gráficas de las Letras Griegas")
S_range = np.linspace(1, 200, 100)
delta_values = delta_call(S_range, K, T, r, sigma)
gamma_values = gamma_call(S_range, K, T, r, sigma)
theta_values = theta_call(S_range, K, T, r, sigma)
vega_values = vega_call(S_range, K, T, r, sigma)
rho_values = rho_call(S_range, K, T, r, sigma)

fig, ax = plt.subplots(3, 2, figsize=(14, 10))
ax[0, 0].plot(S_range, delta_values, label='Delta')
ax[0, 0].set_title('Delta')
ax[0, 1].plot(S_range, gamma_values, label='Gamma', color='orange')
ax[0, 1].set_title('Gamma')
ax[1, 0].plot(S_range, theta_values, label='Theta', color='green')
ax[1, 0].set_title('Theta')
ax[1, 1].plot(S_range, vega_values, label='Vega', color='red')
ax[1, 1].set_title('Vega')
ax[2, 0].plot(S_range, rho_values, label='Rho', color='purple')
ax[2, 0].set_title('Rho')
plt.tight_layout()
st.pyplot(fig)
