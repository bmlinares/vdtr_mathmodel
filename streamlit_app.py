import streamlit as st

from models.vdtr import run

st.set_page_config(
     page_title="Disease Informed Neural Networks",
     page_icon="🧬",
     layout="wide",
)

st.title("COVID-19 and Disease Informed Neural Networks")

tab_sird, tab_help = st.tabs(["VDTR", "Help"])

with tab_sird:

    st.latex(r"""
        \begin{align*}
        \frac{dS}{dt} &= - \frac{\beta}{N}  S I \\
        \frac{dI}{dt} &= \frac{\beta}{N} S I - \omega  I - \gamma I \\
        \frac{dR}{dt} &= \omega I \\
        \frac{dD}{dt} &= \gamma I \\
        \end{align*}
    """
    )

    col11, col12, col13, col14 = st.columns(4)

    N = col11.number_input(
        "N",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        format=None,
        help="Population",
    )

    beta = col12.number_input(
        "Beta",
        min_value=0.001,
        max_value=1.0,
        value=0.5,
        step=0.001,
        format=None,
        help="Transmission Rate",
    )

    gamma = col13.number_input(
        "gamma",
        min_value=0.001,
        max_value=1.0,
        value=1 / 14,
        step=0.001,
        format=None,
        help="Rate at which Infected individuals become Recovered",
    )

    alpha = col14.number_input(
        "Alpha",
        min_value=0.001,
        max_value=1.0,
        value=1 / 5,
        step=0.001,
        format=None,
        help="Rate at which Infected individuals become Dead",
    )

    
    col21, col22, col23 = st.columns(3)

    iterations = col21.number_input(
        "Iterations",
        min_value=1000,
        max_value=50000,
        value=10000,
        step=1000,
        format=None,
        help="Training iterations",
    )

    layers = col22.number_input(
        "Layers",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        format=None,
        help="Neural Network hidden layers",
    )

    neurons = col23.number_input(
        "Neurons",
        min_value=8,
        max_value=256,
        value=64,
        step=8,
        format=None,
        help="Neurons for each hidden layer",
    )

    if st.button("Run DINN"):
        with st.spinner('Wait for it...'):
            p = 0.1
            delta = 0.1
            eta1 = 0.1
            eta2 = 0.1
            mu = 0.1
            error_df, fig = run(N, alpha, beta, gamma, p, delta, eta1, eta2, mu, iterations, layers, neurons)
            st.pyplot(fig)
            st.table(error_df)


with tab_help:
    st.header("Work in progress... Here is kitty.")
    st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

