import numpy as np
from scipy.stats import beta


def posterior_params(prior_a, prior_b, clicks, no_clicks):
    return prior_a + clicks, prior_b + no_clicks


def posterior_draws(post_a, post_b, n_draws=20000):
    return np.random.beta(post_a, post_b, size=n_draws)


def expected_loss_vs_control(candidate_draws, control_draws, optimize):
    if optimize == "Maximize click rate (conversion)":
        loss = np.maximum(0, control_draws - candidate_draws)
    else:
        loss = np.maximum(0, candidate_draws - control_draws)
    return float(loss.mean())


def pairwise_probability_vs_control(candidate_draws, control_draws, optimize):
    if optimize == "Maximize click rate (conversion)":
        return float((candidate_draws > control_draws).mean()), "Pr(beat control)"
    return float((candidate_draws < control_draws).mean()), "Pr(lower than control)"


def best_probability_matrix(draw_matrix, optimize):
    if optimize == "Maximize click rate (conversion)":
        winners = np.argmax(draw_matrix, axis=0)
        label = "Pr(best)"
    else:
        winners = np.argmin(draw_matrix, axis=0)
        label = "Pr(lowest click rate)"

    probs = [(winners == i).mean() for i in range(draw_matrix.shape[0])]
    return probs, label


def posterior_mean(a, b):
    return float(a / (a + b))


def safe_beta_pdf(x, a, b):
    y = beta.pdf(x, a, b)
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


def safe_beta_mean_height(mean, a, b):
    h = beta.pdf(mean, a, b)
    if np.isfinite(h):
        return float(h)
    return None