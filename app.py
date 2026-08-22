import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.black_scholes import black_scholes_price
from src.monte_carlo import simulate_gbm_paths

st.set_page_config(page_title="Quant Portfolio: Option Pricing", layout="wide")

st.title("QUANT PORTFOLIO: EUROPEAN OPTION PRICING MODEL")

with st.sidebar:
    st.header("MODEL PARAMETERS")
    S0 = st.slider("Asset Price (S0)", 50.0, 200.0, 100.0)
    K = st.slider("Strike Price (K)", 50.0, 200.0, 100.0)
    r = st.slider("Risk-Free Rate (r)", 0.0, 0.2, 0.05)
    sigma = st.slider("Volatility (σ)", 0.05, 0.8, 0.20)
    T = st.slider("Time to Maturity (T)", 0.1, 3.0, 1.0)
    N = st.slider("Number of Simulations (N)", 1000, 100000, 50000)
    option_type = st.selectbox("Option Type", ["call", "put"])

mc_price, se, ci, exact_ST = simulate_gbm_paths(S0, K, r, sigma, T, N, 100, True, option_type)
bs_price = black_scholes_price(S0, K, r, sigma, T, option_type)

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Model Summary")
    st.metric("MC Price", f"${mc_price:.2f}")
    st.metric("BS Analytical Price", f"${bs_price:.4f}")
    st.metric("Standard Error", f"{se:.4f}")
    st.write(f"**95% CI:** [${ci[0]:.2f},${ci[1]:.2f}]")

with col2:
    # Build FEA style surface grid
    S_range = np.linspace(S0 * 0.7, S0 * 1.3, 30)
    K_range = np.linspace(K * 0.7, K * 1.3, 30)
    S_grid, K_grid = np.meshgrid(S_range, K_range)

    if option_type == "call":
        Z_grid = np.maximum(S_grid - K_grid, 0)
    else:
        Z_grid = np.maximum(K_grid - S_grid, 0)

    fig = go.Figure(data=[go.Surface(
        z=Z_grid, x=K_grid, y=S_grid,
        colorscale='Jet',
        contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
    )])

    fig.update_layout(
        title=f"MONTE CARLO TERMINAL PAYOFF SURFACE (N={N:,})",
        scene=dict(
            xaxis_title='Strike Price (K)',
            yaxis_title='Spot Price (S0)',
            zaxis_title='Payoff ($)'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    st.plotly_chart(fig, use_container_width=True)
