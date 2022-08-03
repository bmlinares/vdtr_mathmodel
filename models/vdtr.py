import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import deepxde as dde
import seaborn as sns

from deepxde.backend import tf
from scipy.integrate import odeint

sns.set_theme(style="darkgrid")

# dde.config.real.set_float64()


def vdtr_model(
    t,
    N,
    alpha,
    beta,
    gamma,
    p,
    delta,
    omega,
    eta,
    mu,
):
    def func(y, t):
        V, D, T, R = y
        dV = alpha * N - (beta * V * D) / N + omega * T + eta * R - mu * V
        dD = (beta * V * D) / N - gamma * D - mu * D
        dT = p * gamma * D - delta * T - omega * T - mu * T
        dR = (1 - p) * gamma * D + delta * T - eta * R - mu * R
        return [dV, dD, dT, dR]
    
    V_0= N - 1
    D_0 = 1
    T_0 = 0
    R_0 = 0

    y0 = [V_0, D_0, T_0, R_0]
    return odeint(func, y0, t)


def dinn(data_t, data_y, N, iterations, layers, neurons):    
    # Variables
    alpha = tf.math.sigmoid(dde.Variable(0.1))
    beta = tf.math.sigmoid(dde.Variable(0.1))
    gamma = tf.math.sigmoid(dde.Variable(0.1))
    p = tf.math.sigmoid(dde.Variable(0.1))
    delta = tf.math.sigmoid(dde.Variable(0.1))
    omega = tf.math.sigmoid(dde.Variable(0.1))
    eta = tf.math.sigmoid(dde.Variable(0.1))
    mu = tf.math.sigmoid(dde.Variable(0.1))
    variable_list = [alpha, beta, gamma, p, delta, omega,  eta, mu]
    
    # ODE model
    def ODE(t, y):
        V = y[:, 0:1]
        D = y[:, 1:2]
        T = y[:, 2:3]
        R = y[:, 3:4]
        
        dV_t = dde.grad.jacobian(y, t, i=0)
        dD_t = dde.grad.jacobian(y, t, i=1)
        dT_t = dde.grad.jacobian(y, t, i=2)
        dR_t = dde.grad.jacobian(y, t, i=3)

        return [
            dV_t - (alpha * N - (beta * V * D) / N + omega * T + eta * R - mu * V),
            dD_t - ((beta * V * D) / N - gamma * D - mu * D),
            dT_t - (p * gamma * D - delta * T - omega * T - mu * T),
            dR_t - ((1 - p) * gamma * D + delta * T - eta * R - mu * R)
        ]
    
    # Geometry
    geom = dde.geometry.TimeDomain(data_t[0, 0], data_t[-1, 0])
    
    # Boundaries
    def boundary(_, on_initial):
        return on_initial
    
    # Initial conditions
    ic_V = dde.icbc.IC(geom, lambda x: N - 1, boundary, component=0)
    ic_D = dde.icbc.IC(geom, lambda x: 1, boundary, component=1)
    ic_T = dde.icbc.IC(geom, lambda x: 0, boundary, component=2)
    ic_R = dde.icbc.IC(geom, lambda x: 0, boundary, component=3)
    
    # Train data
    observe_V = dde.icbc.PointSetBC(data_t, data_y[:, 0:1], component=0)
    observe_D = dde.icbc.PointSetBC(data_t, data_y[:, 1:2], component=1)
    observe_T = dde.icbc.PointSetBC(data_t, data_y[:, 2:3], component=2)
    observe_R = dde.icbc.PointSetBC(data_t, data_y[:, 3:4], component=3)
    
    # Model
    data = dde.data.PDE(
        geom,
        ODE,
        [
            ic_V,
            ic_D,
            ic_T,
            ic_R,
            observe_V,
            observe_D,
            observe_T,
            observe_R
        ],
        num_domain=400,
        num_boundary=2,
        anchors=data_t,
    )
    
    net = dde.nn.FNN([1] + [neurons] * layers + [4], "relu", "Glorot uniform")
    
    def feature_transform(t):
        t = t / data_t[-1, 0]
        return t

    net.apply_feature_transform(feature_transform)
    
    model = dde.Model(data, net)
    model.compile(
        "adam",
        lr=1e-3,
        loss_weights=4 * [1] + 4 * [1] + 4 * [1],
        external_trainable_variables=variable_list
    )
    variable = dde.callbacks.VariableValue(
        variable_list,
        period=5000
    )
    _, _ = model.train(
        iterations=iterations,
        display_every=1000,
        callbacks=[variable]
      )
    return model, variable

    
def error(parameters_real, parameters_pred):
    parameter_names = [
    "alpha",
    "beta",
    "gamma",
    "p",
    "delta",
    "omega",
    "eta",
    "mu",
]
    errors = (
        pd.DataFrame(
            {
                "Real": parameters_real,
                "Predicted": parameters_pred
            },
            index=parameter_names
        )
        .assign(
            **{"Relative Error": lambda x: (x["Real"] - x["Predicted"]).abs() / x["Real"]}
        )
    )
    return errors


def plot(data_pred, data_real):

    g = sns.relplot(
        data=data_pred,
        x="time",
        y="population",
        hue="status",
        kind="line",
        aspect=2,
    )

    sns.scatterplot(
        data=data_real,
        x="time",
        y="population",
        hue="status",
        ax=g.ax,
        legend=False
    )

    (
        g.set_axis_labels("Time", "Population")
        .tight_layout(w_pad=1)
    )

    g._legend.set_title("Status")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(f"VDTR model estimation")
    return g


def run_dinn(N, alpha, beta, gamma, p, delta, omega, eta, mu, iterations, layers, neurons):

    names = list("VDTR")
    t = np.arange(0, 366, 3)[:, np.newaxis]
    y = vdtr_model(np.ravel(t), N, alpha, beta, gamma, p, delta, omega, eta, mu)
    data_real = (
        pd.DataFrame(y, columns=names, index=t.ravel())
        .rename_axis("time")
        .reset_index()
        .melt(id_vars="time", var_name="status", value_name="population")
    )

    model, variable = dinn(t, y, N, iterations, layers, neurons)
    
    full_t = np.arange(0, 366)[:, np.newaxis]
    y_pred = model.predict(full_t)
    data_pred = (
        pd.DataFrame(y_pred, columns=names, index=full_t.ravel())
        .rename_axis("time")
        .reset_index()
        .melt(id_vars="time", var_name="status", value_name="population")
    )

    parameters_real = [alpha, beta, gamma, p, delta, omega, eta, mu]
    parameters_pred = variable.value
    error_df = error(parameters_real, parameters_pred)
    fig = plot(data_pred, data_real)

    return error_df, fig


def run_forward(N, alpha, beta, gamma, p, delta, omega, eta, mu):
    names = list("VDTR")
    t = np.arange(0, 366, 7)[:, np.newaxis]
    y = vdtr_model(np.ravel(t), N, alpha, beta, gamma, p, delta, omega, eta, mu)
    data_real = (
        pd.DataFrame(y, columns=names, index=t.ravel())
        .rename_axis("time")
        .reset_index()
        .melt(id_vars="time", var_name="status", value_name="population")
    )

    g = sns.relplot(
        data=data_real,
        x="time",
        y="population",
        hue="status",
        kind="line",
        aspect=2,
    )

    (
        g.set_axis_labels("Time", "Population")
        .tight_layout(w_pad=1)
    )

    g._legend.set_title("Status")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(f"VDTR forward solution")
    return g

if __name__ == "__main__":
    N = 1001
    alpha = 0.8
    beta = 0.8
    gamma = 0.3
    p = 0.1
    delta = 0.1
    omega = 0.1
    eta = 0.1
    mu = 0.1
    iterations = 50000
    layers = 3
    neurons = 64
    error_df, fig = run_dinn(N, alpha, beta, gamma, p, delta, omega, eta, mu, iterations, layers, neurons)
    plt.show()
    print(error_df)
