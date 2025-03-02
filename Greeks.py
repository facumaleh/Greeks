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
