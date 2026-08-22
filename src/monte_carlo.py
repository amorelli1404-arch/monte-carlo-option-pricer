import numpy as np

def simulate_gbm_paths(S0, K, r, sigma, T, N, M, use_antithetic=True, option_type="call"):
    """Vectorized GBM simulation using Antithetic Variates for variance reduction."""
    dt = T / M

    if use_antithetic:
        half_N = N // 2
        Z = np.random.normal(0, 1, size=(half_N, M))
        Z = np.vstack((Z, -Z))
    else:
        Z = np.random.normal(0, 1, size=(N, M))

    # Vectorized terminal price using exact analytical solution
    exact_ST = S0 * np.exp(np.sum((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z, axis=1))

    # Calculate payoffs
    if option_type.lower() == "call":
        payoffs = np.maximum(exact_ST - K, 0)
    else:
        payoffs = np.maximum(K - exact_ST, 0)

    discounted_payoffs = np.exp(-r * T) * payoffs

    mc_price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs) / np.sqrt(len(discounted_payoffs))
    ci_95 = (mc_price - 1.96 * std_error, mc_price + 1.96 * std_error)

    return mc_price, std_error, ci_95, exact_ST
