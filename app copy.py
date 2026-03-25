import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.optimize import minimize_scalar

st.set_page_config(page_title="Bayesian Phishing Risk Estimator", layout="wide")


# -----------------------------
# Helper functions
# -----------------------------
def fit_beta_from_mean_and_interval(mean, lower, upper, interval_mass=0.95):
    tail = (1 - interval_mass) / 2

    def objective(log_k):
        k = np.exp(log_k)
        a = mean * k
        b = (1 - mean) * k

        if a <= 0 or b <= 0:
            return 1e9

        q_low = beta.ppf(tail, a, b)
        q_high = beta.ppf(1 - tail, a, b)
        return (q_low - lower) ** 2 + (q_high - upper) ** 2

    result = minimize_scalar(objective, bounds=(-6, 10), method="bounded")
    k_opt = float(np.exp(result.x))
    return mean * k_opt, (1 - mean) * k_opt


def beta_from_mean_and_ess(mean, ess):
    return mean * ess, (1 - mean) * ess


def validate_mean_ess(mean, ess):
    return (0 < mean < 1) and (ess > 0)


def validate_mean_range(mean, lower, upper):
    if lower is None or upper is None:
        return False
    return (0 < lower < 1) and (0 < mean < 1) and (0 < upper < 1) and (lower < mean < upper)


def derive_prior(mean, prior_mode, ess=None, lower=None, upper=None):
    if prior_mode == "Mean + effective sample size":
        return beta_from_mean_and_ess(mean, ess)
    return fit_beta_from_mean_and_interval(mean, lower, upper, interval_mass=0.95)


def safe_beta_pdf(x, a, b):
    y = beta.pdf(x, a, b)
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


def safe_beta_quantiles(a, b, mass=0.95):
    tail = (1 - mass) / 2
    low, high = beta.ppf([tail, 1 - tail], a, b)
    return float(low), float(high)


def safe_beta_mean_height(mean, a, b):
    h = beta.pdf(mean, a, b)
    if np.isfinite(h):
        return float(h)
    return None


def add_mean_label(ax, mean_x, mean_y, color, text):
    ax.text(
        mean_x,
        mean_y * 1.03,
        text,
        color=color,
        fontsize=9,
        ha="center",
        va="bottom"
    )


def shade_interval_under_curve(ax, x, y, low, high, color, alpha=0.12):
    mask = (x >= low) & (x <= high)
    ax.fill_between(x, 0, y, where=mask, interpolate=True, color=color, alpha=alpha)


def plot_density(
    ax,
    x,
    a,
    b,
    color,
    label=None,
    linestyle="-",
    linewidth=2.0,
    alpha=1.0,
    interval=None,
    interval_alpha=None,
    mean_label=False,
    mean_line_style=":"
):
    y = safe_beta_pdf(x, a, b)

    if interval is not None and interval_alpha is not None:
        shade_interval_under_curve(ax, x, y, interval[0], interval[1], color=color, alpha=interval_alpha)

    ax.plot(
        x,
        y,
        linestyle=linestyle,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        label=label
    )

    if mean_label:
        mean = a / (a + b)
        mean_height = safe_beta_mean_height(mean, a, b)
        if mean_height is not None:
            ax.vlines(
                mean,
                0,
                mean_height,
                color=color,
                linestyle=mean_line_style,
                linewidth=1.8 if mean_line_style == ":" else 1.6,
                alpha=0.9
            )
            add_mean_label(ax, mean, mean_height, color, f"{mean:.3f}")

    return y


def compute_dynamic_xmax(high_values, padding=0.20):
    if len(high_values) == 0:
        return 1.0
    xmax = max(high_values) + padding
    xmax = min(1.0, xmax)
    xmax = max(0.2, xmax)
    return xmax


def posterior_overlap_coefficient(a1, b1, a2, b2, grid_max, n_grid=2000):
    x = np.linspace(0.0001, grid_max, n_grid)
    y1 = safe_beta_pdf(x, a1, b1)
    y2 = safe_beta_pdf(x, a2, b2)
    overlap = np.trapezoid(np.minimum(y1, y2), x)
    return float(overlap)


def build_overlap_table(posterior_param_rows, grid_max):
    rows = []
    for i in range(len(posterior_param_rows)):
        for j in range(i + 1, len(posterior_param_rows)):
            r1 = posterior_param_rows[i]
            r2 = posterior_param_rows[j]
            overlap = posterior_overlap_coefficient(
                r1["post_a"], r1["post_b"],
                r2["post_a"], r2["post_b"],
                grid_max=grid_max
            )
            rows.append({
                "Variant A": r1["name"],
                "Variant B": r2["name"],
                "Posterior overlap": round(overlap, 4),
                "Percent overlap": round(overlap * 100, 1)
            })
    return pd.DataFrame(rows)


# -----------------------------
# App header
# -----------------------------
st.title("Bayesian Phishing Risk Estimator")
st.write(
    "Set priors for phishing click rates, inspect those priors, then update them with observed clicks and no-clicks."
)

st.info(
    "This tool estimates click rates using Bayesian updating. It does not choose a winner automatically. "
    "Use probabilities, uncertainty, and expected loss together."
)

colors = ["C0", "C1", "C2", "C3", "C4", "C5"]


# -----------------------------
# Inputs
# -----------------------------
st.subheader("Step 1. Number of variants")
k = st.number_input("Number of variants", min_value=1, max_value=6, value=2, step=1)

st.subheader("Step 2. Prior structure")
same_prior_for_all = st.toggle("Use the same prior for all variants", value=True)

st.subheader("Step 3. Prior input method")
prior_mode = st.radio(
    "Choose prior input method",
    options=["Mean + effective sample size", "Mean + plausible range"]
)

if prior_mode == "Mean + plausible range":
    st.info(
        "In range mode, the lower and upper values are treated as approximate central 95% prior bounds. "
        "Because a Beta prior has only two parameters, the fitted interval may not match your entered bounds exactly."
    )

st.subheader("Step 4. Name the variants")
variant_names = []
for i in range(k):
    name = st.text_input(f"Variant {i+1} name", value=f"Variant {i+1}", key=f"name_{i}")
    variant_names.append(name)

st.subheader("Step 5. Enter priors")

prior_inputs = []

if same_prior_for_all:
    st.markdown("### Common prior for all variants")
    mean = st.number_input("Expected click rate", min_value=0.001, max_value=0.999, value=0.10, step=0.01, key="common_mean")

    if prior_mode == "Mean + effective sample size":
        ess = st.number_input("Prior effective sample size", min_value=0.1, value=10.0, step=1.0, key="common_ess")
        lower = None
        upper = None
    else:
        lower = st.number_input("Approximate lower 95% prior bound", min_value=0.001, max_value=0.999, value=0.05, step=0.01, key="common_low")
        upper = st.number_input("Approximate upper 95% prior bound", min_value=0.001, max_value=0.999, value=0.20, step=0.01, key="common_high")
        ess = None

    for i in range(k):
        prior_inputs.append({
            "name": variant_names[i],
            "mean": mean,
            "ess": ess,
            "low": lower,
            "high": upper,
            "color": colors[i]
        })
else:
    for i in range(k):
        st.markdown(f"### Prior for {variant_names[i]}")
        mean = st.number_input(
            f"Expected click rate for {variant_names[i]}",
            min_value=0.001,
            max_value=0.999,
            value=0.10,
            step=0.01,
            key=f"mean_{i}"
        )

        if prior_mode == "Mean + effective sample size":
            ess = st.number_input(
                f"Prior effective sample size for {variant_names[i]}",
                min_value=0.1,
                value=10.0,
                step=1.0,
                key=f"ess_{i}"
            )
            lower = None
            upper = None
        else:
            lower = st.number_input(
                f"Approximate lower 95% prior bound for {variant_names[i]}",
                min_value=0.001,
                max_value=0.999,
                value=0.05,
                step=0.01,
                key=f"low_{i}"
            )
            upper = st.number_input(
                f"Approximate upper 95% prior bound for {variant_names[i]}",
                min_value=0.001,
                max_value=0.999,
                value=0.20,
                step=0.01,
                key=f"high_{i}"
            )
            ess = None

        prior_inputs.append({
            "name": variant_names[i],
            "mean": mean,
            "ess": ess,
            "low": lower,
            "high": upper,
            "color": colors[i]
        })


# -----------------------------
# Build priors
# -----------------------------
prior_valid = True
prior_records = []
prior_summary_rows = []
prior_highs = []
mismatch_notes = []

for p in prior_inputs:
    if prior_mode == "Mean + effective sample size":
        is_valid = validate_mean_ess(p["mean"], p["ess"])
    else:
        is_valid = validate_mean_range(p["mean"], p["low"], p["high"])

    if not is_valid:
        st.error(f"{p['name']}: invalid prior inputs.")
        prior_valid = False
        continue

    a, b = derive_prior(
        mean=p["mean"],
        prior_mode=prior_mode,
        ess=p["ess"],
        lower=p["low"],
        upper=p["high"]
    )

    fitted_low, fitted_high = safe_beta_quantiles(a, b)
    ess_value = a + b

    if prior_mode == "Mean + plausible range":
        diff = abs(fitted_low - p["low"]) + abs(fitted_high - p["high"])
        if diff > 0.05:
            mismatch_notes.append(
                f"{p['name']}: the fitted 95% prior interval differs noticeably from the entered bounds."
            )

    prior_records.append({
        "name": p["name"],
        "a": a,
        "b": b,
        "mean": a / (a + b),
        "low": fitted_low,
        "high": fitted_high,
        "ess": ess_value,
        "color": p["color"]
    })

    row = {
        "Variant": p["name"],
        "Prior mean": round(a / (a + b), 4),
        "Prior alpha": round(a, 3),
        "Prior beta": round(b, 3),
        "Prior effective sample size": round(ess_value, 3),
        "Fitted 95% low": round(fitted_low, 4),
        "Fitted 95% high": round(fitted_high, 4),
        "Fitted 95% CI": f"[{fitted_low:.3f}, {fitted_high:.3f}]"
    }

    if prior_mode == "Mean + plausible range":
        row["Input lower"] = round(p["low"], 4)
        row["Input upper"] = round(p["high"], 4)

    if prior_mode == "Mean + effective sample size":
        row["User-entered prior ESS"] = round(p["ess"], 3)

    prior_summary_rows.append(row)
    prior_highs.append(fitted_high)


# -----------------------------
# Prior plot
# -----------------------------
st.subheader("Step 6. Inspect priors")

if prior_valid:
    for note in mismatch_notes:
        st.warning(note)

    x_prior_max = compute_dynamic_xmax(prior_highs, padding=0.20)
    x_prior = np.linspace(0.0001, x_prior_max, 1200)

    fig_prior, ax_prior = plt.subplots(figsize=(10, 5))
    ax_prior.set_title("Implied prior distributions for click rate")
    ax_prior.set_xlabel("Click rate")
    ax_prior.set_ylabel("Density")

    all_same = len({(round(r["a"], 6), round(r["b"], 6)) for r in prior_records}) == 1

    if all_same:
        r = prior_records[0]
        plot_density(
            ax=ax_prior,
            x=x_prior,
            a=r["a"],
            b=r["b"],
            color="black",
            label="Common prior",
            linestyle="--",
            linewidth=2.2,
            alpha=0.95,
            interval=(r["low"], r["high"]),
            interval_alpha=0.12,
            mean_label=True,
            mean_line_style=":"
        )
    else:
        for r in prior_records:
            plot_density(
                ax=ax_prior,
                x=x_prior,
                a=r["a"],
                b=r["b"],
                color=r["color"],
                label=r["name"],
                linestyle="--",
                linewidth=2.0,
                alpha=0.95,
                interval=(r["low"], r["high"]),
                interval_alpha=0.12,
                mean_label=True,
                mean_line_style=":"
            )

    ax_prior.set_xlim(0, x_prior_max)
    ax_prior.legend()
    st.pyplot(fig_prior)

    prior_df = pd.DataFrame(prior_summary_rows)
    st.dataframe(prior_df, use_container_width=True)


# -----------------------------
# Observed data
# -----------------------------
st.subheader("Step 7. Enter observed data")

variant_data = []
for i, name in enumerate(variant_names):
    st.markdown(f"### {name}")
    col1, col2 = st.columns(2)

    with col1:
        clicks = st.number_input(
            f"Observed clicks for {name}",
            min_value=0,
            value=10,
            step=1,
            key=f"clicks_{i}"
        )
    with col2:
        no_clicks = st.number_input(
            f"Observed no-clicks for {name}",
            min_value=0,
            value=90,
            step=1,
            key=f"noclicks_{i}"
        )

    if prior_valid:
        variant_data.append({
            "name": name,
            "clicks": clicks,
            "no_clicks": no_clicks,
            "prior": prior_records[i],
            "color": colors[i]
        })


# -----------------------------
# Decision framing
# -----------------------------
st.subheader("Step 8. Decision framing")

decision_direction = st.radio(
    "What outcome are you trying to optimize?",
    ["Minimize click rate (phishing risk)", "Maximize click rate (conversion)"]
)

reference_arm = st.selectbox("Reference (control) variant", variant_names)
show_priors = st.toggle("Show priors on posterior plot", value=True)
n_draws = st.number_input("Posterior draws", min_value=5000, max_value=50000, value=20000, step=5000)

st.caption(
    "Posterior draws are simulated values from each variant's posterior distribution. "
    "They are used to estimate probabilities such as which variant is lowest-risk or beats the control. "
    "The default is usually fine and can usually be left unchanged."
)

run = st.button("Run Bayesian update", disabled=not prior_valid)


# -----------------------------
# Posterior analysis
# -----------------------------
if run and prior_valid:
    st.subheader("Step 9. Posterior analysis")

    posterior_rows = []
    updating_rows = []
    draw_list = []
    posterior_highs = []
    safeguard_notes = []
    posterior_param_rows = []

    for v in variant_data:
        pa = v["prior"]["a"]
        pb = v["prior"]["b"]
        post_a = pa + v["clicks"]
        post_b = pb + v["no_clicks"]

        post_mean = post_a / (post_a + post_b)
        post_low, post_high = safe_beta_quantiles(post_a, post_b)
        posterior_highs.append(post_high)

        draws = np.random.beta(post_a, post_b, size=int(n_draws))
        draw_list.append(draws)

        posterior_param_rows.append({
            "name": v["name"],
            "post_a": post_a,
            "post_b": post_b
        })

        observed_n = v["clicks"] + v["no_clicks"]
        prior_ess = v["prior"]["ess"]

        if prior_ess < 1:
            safeguard_notes.append(f"{v['name']}: prior is extremely weak (ESS < 1).")
        if prior_ess > 1000:
            safeguard_notes.append(f"{v['name']}: prior is extremely strong (ESS > 1000).")
        if observed_n > 0 and prior_ess > observed_n * 2:
            safeguard_notes.append(f"{v['name']}: prior is stronger than the observed data.")
        if observed_n > 0 and prior_ess < observed_n * 0.01:
            safeguard_notes.append(f"{v['name']}: prior is negligible relative to the observed data.")

        posterior_rows.append({
            "Variant": v["name"],
            "Prior mean": round(v["prior"]["mean"], 4),
            "Prior 95% low": round(v["prior"]["low"], 4),
            "Prior 95% high": round(v["prior"]["high"], 4),
            "Observed clicks": v["clicks"],
            "Observed no-clicks": v["no_clicks"],
            "Observed sample size": observed_n,
            "Posterior mean": round(post_mean, 4),
            "Posterior 95% low": round(post_low, 4),
            "Posterior 95% high": round(post_high, 4),
            "Posterior 95% CI": f"[{post_low:.3f}, {post_high:.3f}]",
            "Prior effective sample size": round(prior_ess, 3)
        })

        updating_rows.append({
            "Variant": v["name"],
            "Prior alpha": round(pa, 3),
            "Prior beta": round(pb, 3),
            "Observed clicks": v["clicks"],
            "Observed no-clicks": v["no_clicks"],
            "Posterior alpha": round(post_a, 3),
            "Posterior beta": round(post_b, 3)
        })

    for note in safeguard_notes:
        st.warning(note)

    draw_matrix = np.array(draw_list)
    control_idx = variant_names.index(reference_arm)

    if decision_direction == "Maximize click rate (conversion)":
        winners = np.argmax(draw_matrix, axis=0)
        best_label = "Pr(best)"
        pair_label = "Pr(beat control)"
    else:
        winners = np.argmin(draw_matrix, axis=0)
        best_label = "Pr(lowest click rate)"
        pair_label = "Pr(lower than control)"

    best_probs = [(winners == i).mean() for i in range(draw_matrix.shape[0])]

    for i in range(len(posterior_rows)):
        posterior_rows[i][best_label] = round(best_probs[i], 4)

    for i in range(len(posterior_rows)):
        if i == control_idx:
            posterior_rows[i][pair_label] = None
            posterior_rows[i]["Expected loss vs control"] = 0.0
            continue

        if decision_direction == "Maximize click rate (conversion)":
            pair_prob = (draw_matrix[i] > draw_matrix[control_idx]).mean()
            loss = np.maximum(0, draw_matrix[control_idx] - draw_matrix[i]).mean()
        else:
            pair_prob = (draw_matrix[i] < draw_matrix[control_idx]).mean()
            loss = np.maximum(0, draw_matrix[i] - draw_matrix[control_idx]).mean()

        posterior_rows[i][pair_label] = round(float(pair_prob), 4)
        posterior_rows[i]["Expected loss vs control"] = round(float(loss), 5)

    # Posterior plot
    x_post_max = compute_dynamic_xmax(posterior_highs, padding=0.20)
    x_post = np.linspace(0.0001, x_post_max, 1200)

    fig_post, ax_post = plt.subplots(figsize=(10, 5))
    ax_post.set_title("Posterior distributions for click rate")
    ax_post.set_xlabel("Click rate")
    ax_post.set_ylabel("Density")

    all_same = len({(round(r["a"], 6), round(r["b"], 6)) for r in prior_records}) == 1

    for v in variant_data:
        pa = v["prior"]["a"]
        pb = v["prior"]["b"]
        post_a = pa + v["clicks"]
        post_b = pb + v["no_clicks"]
        post_low, post_high = safe_beta_quantiles(post_a, post_b)

        prior_color = "black" if all_same else v["color"]

        if show_priors:
            plot_density(
                ax=ax_post,
                x=x_post,
                a=pa,
                b=pb,
                color=prior_color,
                label=None,
                linestyle="--",
                linewidth=2.0,
                alpha=0.9,
                interval=(v["prior"]["low"], v["prior"]["high"]),
                interval_alpha=0.06,
                mean_label=False
            )

        plot_density(
            ax=ax_post,
            x=x_post,
            a=post_a,
            b=post_b,
            color=v["color"],
            label=v["name"],
            linestyle="-",
            linewidth=2.5,
            alpha=0.55,
            interval=(post_low, post_high),
            interval_alpha=0.12,
            mean_label=True,
            mean_line_style="-"
        )

    ax_post.set_xlim(0, x_post_max)
    ax_post.legend()
    st.pyplot(fig_post)

    posterior_df = pd.DataFrame(posterior_rows)
    st.subheader("Posterior summary")
    st.dataframe(posterior_df, use_container_width=True)

    st.subheader("Posterior overlap table")
    overlap_df = build_overlap_table(posterior_param_rows, grid_max=x_post_max)
    if len(overlap_df) > 0:
        st.dataframe(overlap_df, use_container_width=True)
        st.caption(
            "Posterior overlap measures how similar two posterior distributions are. "
            "Higher values mean the variants are less distinguishable. Lower values mean clearer separation."
        )
    else:
        st.write("Overlap requires at least two variants.")

    st.subheader("Bayesian updating table")
    st.dataframe(pd.DataFrame(updating_rows), use_container_width=True)

    st.subheader("Interpretation note")
    st.write(
        "Lower posterior means indicate lower estimated click risk. "
        "Use the posterior probability columns, the overlap table, and expected loss versus control together when comparing variants."
    )