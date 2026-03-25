import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import beta


def fit_beta_from_mean_and_interval(mean, lower, upper, interval_mass=0.95):
    tail = (1 - interval_mass) / 2

    def objective(log_k):
        k = np.exp(log_k)
        a = mean * k
        b = (1 - mean) * k

        q_low = beta.ppf(tail, a, b)
        q_high = beta.ppf(1 - tail, a, b)

        return (q_low - lower) ** 2 + (q_high - upper) ** 2

    result = minimize_scalar(objective, bounds=(-6, 10), method="bounded")
    k = np.exp(result.x)
    return mean * k, (1 - mean) * k


def beta_from_mean_and_ess(mean, ess):
    return mean * ess, (1 - mean) * ess


def validate_mean_ess(mean, ess):
    return (0 < mean < 1) and (ess > 0)


def validate_mean_range(mean, lower, upper):
    if lower is None or upper is None:
        return False
    return (0 < lower < mean < upper < 1)


def safe_beta_quantiles(a, b):
    return beta.ppf([0.025, 0.975], a, b)


def derive_prior_from_dict(d, mode):
    if mode == "Mean + effective sample size":
        return beta_from_mean_and_ess(d["mean_click"], d["prior_ess"])
    return fit_beta_from_mean_and_interval(d["mean_click"], d["lower_click"], d["upper_click"])


def validate_prior_dict(d, mode):
    if mode == "Mean + effective sample size":
        return validate_mean_ess(d["mean_click"], d["prior_ess"])
    return validate_mean_range(d["mean_click"], d["lower_click"], d["upper_click"])


def prior_signature(a, b):
    return round(a, 6), round(b, 6)
