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
    
    # Crear la figura con Plotly
    fig = go.Figure()
    
    # Precio real de la opción
    fig.add_trace(go.Scatter(
        x=S_range,
        y=call_prices,
        mode='lines',
        name='Precio Real de la Opción',
        line=dict(color='blue', width=2)
    )
    
    # Aproximación de Taylor de primer grado
    fig.add_trace(go.Scatter(
        x=S_range,
        y=taylor_1_values,
        mode='lines',
        name='Taylor Primer Orden',
        line=dict(color='green', dash='dash', width=2)
    )
    
    # Aproximación de Taylor de segundo grado
    fig.add_trace(go.Scatter(
        x=S_range,
        y=taylor_2_values,
        mode='lines',
        name='Taylor Segundo Orden',
        line=dict(color='red', dash='dash', width=2)
    )
    
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
        template="plotly_dark" if theme == "dark" else "plotly_white",
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

# Pie de página
st.markdown("---")
st.markdown("""
**Creado por:** Facundo Maleh  
**Nota:** Esta aplicación es solo para fines educativos.
""")
