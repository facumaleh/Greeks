import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Título y descripción
st.title("📊 Visualizador de Letras Griegas en Black-Scholes")
st.markdown("""
**Esta aplicación** te permite visualizar cómo cambian las letras griegas (Delta, Gamma, Theta, Vega, Rho) 
en la fórmula de Black-Scholes para una opción call. 

👉 Usa los controles debajo para ajustar los parámetros.
""")

# Controles debajo del título
st.header("⚙️ Parámetros de la Opción")

# Sliders y selectores
col1, col2 = st.columns(2)
with col1:
    S = st.slider("Precio del Activo (S)", 1.0, 200.0, 100.0)
    moneyness = st.selectbox("Moneyness", ["ATM", "ITM", "OTM"])
with col2:
    T = st.slider("Tiempo hasta vencimiento (T)", 0.1, 5.0, 1.0)
    r = st.slider("Tasa libre de riesgo (r)", 0.0, 0.2, 0.05)
    sigma = st.slider("Volatilidad (σ)", 0.1, 1.0, 0.2)

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

# Calcular el precio de la opción call
call_price = black_scholes_call(S, K, T, r, sigma)

# Mostrar el valor del call
st.subheader("💵 Valor de la Opción Call")
st.metric("Precio de la Opción Call", f"{call_price:.4f}")

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

# Mostrar los valores de las letras griegas en columnas
st.subheader("📈 Valores de las Letras Griegas")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Δ Delta", f"{delta:.4f}")
col2.metric("Γ Gamma", f"{gamma:.4f}")
col3.metric("Θ Theta", f"{theta:.4f}")
col4.metric("ν Vega", f"{vega:.4f}")
col5.metric("ρ Rho", f"{rho:.4f}")

# Graficar las letras griegas
st.subheader("📊 Gráficas de las Letras Griegas")
S_range = np.linspace(1, 200, 100)
delta_values = delta_call(S_range, K, T, r, sigma)
gamma_values = gamma_call(S_range, K, T, r, sigma)
theta_values = theta_call(S_range, K, T, r, sigma)
vega_values = vega_call(S_range, K, T, r, sigma)
rho_values = rho_call(S_range, K, T, r, sigma)

fig, ax = plt.subplots(3, 2, figsize=(14, 10))

# Personalizar las gráficas
ax[0, 0].plot(S_range, delta_values, label='Delta', color='blue')
ax[0, 0].set_title('Δ Delta')
ax[0, 0].set_xlabel('Precio del Activo (S)')
ax[0, 0].set_ylabel('Delta')

ax[0, 1].plot(S_range, gamma_values, label='Gamma', color='orange')
ax[0, 1].set_title('Γ Gamma')
ax[0, 1].set_xlabel('Precio del Activo (S)')
ax[0, 1].set_ylabel('Gamma')

ax[1, 0].plot(S_range, theta_values, label='Theta', color='green')
ax[1, 0].set_title('Θ Theta')
ax[1, 0].set_xlabel('Precio del Activo (S)')
ax[1, 0].set_ylabel('Theta')

ax[1, 1].plot(S_range, vega_values, label='Vega', color='red')
ax[1, 1].set_title('ν Vega')
ax[1, 1].set_xlabel('Precio del Activo (S)')
ax[1, 1].set_ylabel('Vega')

ax[2, 0].plot(S_range, rho_values, label='Rho', color='purple')
ax[2, 0].set_title('ρ Rho')
ax[2, 0].set_xlabel('Precio del Activo (S)')
ax[2, 0].set_ylabel('Rho')

# Ocultar la última gráfica (si no se usa)
ax[2, 1].axis('off')

plt.tight_layout()
st.pyplot(fig)

# Pie de página
st.markdown("---")
st.markdown("""
**Nota:** Esta aplicación es solo para fines educativos.
""")
