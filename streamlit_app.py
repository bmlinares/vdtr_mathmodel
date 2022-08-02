import streamlit as st

from models.vdtr import run

st.set_page_config(
     page_title="Disease Informed Neural Networks",
     page_icon="🧬",
     layout="wide",
)

st.title("A Mathematical Model for Understanding and Predicting Dynamics of Depression as an Epidemic")

tab_sird, tab_help = st.tabs(["VDTR", "Help"])

with tab_sird:

    st.latex(r"""
        \begin{align*}
        \frac{dV}{dt} &= \alpha N - \frac{\beta}{N}  V D + \omega T + \eta R - mu V \\
        \frac{dD}{dt} &= \frac{\beta}{N} V D - \gamma  D - \mu D \\
        \frac{dT}{dt} &= \p gamma D - \delta T - \omega T - \mu T \\
        \frac{dR}{dt} &= \(1-p) gamma D + \delta T - \eta R - \mu R \\
        \end{align*}
    """
    )

    col11, col12, col13, col14, col15, col16, col17, col18 = st.columns(8)

    N = col11.number_input(
        "N",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        format=None,
        help="Population",
    )

    alpha = col12.number_input(
        "Alpha",
        min_value=0.001,
        max_value=1.0,
        value=0.1,
        step=0.001,
        format=None,
        help="Recruitment Rate",
    )

    beta = col13.number_input(
        "Beta",
        min_value=0.001,
        max_value=1.0,
        value=0.9,
        step=0.001,
        format=None,
        help="Transmission Rate",
    )

    p = col14.number_input(
        "p",
        min_value=0.001,
        max_value=1.0,
        value=0.3,
        step=0.001,
        format=None,
        help="Probability Rate",
    )

    delta = col15.number_input(
        "Delta",
        min_value=0.001,
        max_value=1.0,
        value=0,1,
        step=0.001,
        format=None,
        help="Treated to Recovery Rate",
    )

    eta1 = col16.number_input(
        "Omega",
        min_value=0.001,
        max_value=1.0,
        value=0.1,
        step=0.001,
        format=None,
        help="Recidivism from Treatment Rate",
    )

    eta2 = col17.number_input(
        "Eta",
        min_value=0.001,
        max_value=1.0,
        value=0.1,
        step=0.001,
        format=None,
        help="Recidivism from Recovery Rate",
    )

    mu = col18.number_input(
        "Mu",
        min_value=0.001,
        max_value=1.0,
        value=0.1,
        step=0.001,
        format=None,
        help="Withdraw Rate",
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
            error_df, fig = run(N, alpha, beta, gamma, p, delta, omega, eta, mu, iterations, layers, neurons)
            st.pyplot(fig)
            st.table(error_df)


with tab_help:
    st.header("Work in progress... Here is kitty.")
    st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

