import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import beta
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bayesian Phishing Risk Estimator", layout="wide")

st.title("Bayesian Phishing Risk Estimator")
st.write(
    "Set priors for phishing click rates and inspect them before entering observed data. "
    "Then update those priors with observed clicks and no-clicks."
)

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
    k_opt = np.exp(result.x)

    alpha = mean * k_opt
    beta_param = (1 - mean) * k_opt
    return alpha, beta_param

def beta_from_mean_and_ess(mean, ess):
    alpha = mean * ess
    beta_param = (1 - mean) * ess
    return alpha, beta_param

def validate_mean_ess(mean, ess):
    return (0 < mean < 1) and (ess > 0)

def validate_mean_range(mean, lower, upper):
    if lower is None or upper is None:
        return False
    return (0 < lower < 1) and (0 < mean < 1) and (0 < upper < 1) and (lower < mean < upper)

def safe_beta_pdf(x, a, b):
    y = beta.pdf(x, a, b)
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

def safe_beta_quantiles(a, b):
    low, high = beta.ppf([0.025, 0.975], a, b)
    return low, high

def safe_beta_mean_height(mean, a, b):
    h = beta.pdf(mean, a, b)
    if np.isfinite(h):
        return h
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

def compute_dynamic_xmax(high_values, padding=0.20):
    if len(high_values) == 0:
        return 1.0
    xmax = max(high_values) + padding
    xmax = min(1.0, xmax)
    xmax = max(0.2, xmax)
    return xmax

def prior_signature(a, b, digits=6):
    return (round(float(a), digits), round(float(b), digits))

def shade_interval_under_curve(ax, x, y, low, high, color, alpha=0.12):
    mask = (x >= low) & (x <= high)
    ax.fill_between(x, 0, y, where=mask, color=color, alpha=alpha, interpolate=True)

def get_prior_inputs(prefix, prior_mode, label_suffix=""):
    mean_click = st.number_input(
        f"Expected click rate{label_suffix}",
        min_value=0.001,
        max_value=0.999,
        value=0.10,
        step=0.01,
        key=f"{prefix}_mean"
    )

    if prior_mode == "Mean + effective sample size":
        prior_ess = st.number_input(
            f"Prior effective sample size{label_suffix}",
            min_value=0.1,
            value=10.0,
            step=1.0,
            key=f"{prefix}_ess",
            help="For example, a mean of 0.10 and an ESS of 10 gives Beta(1, 9)."
        )
        lower_click = None
        upper_click = None
    else:
        lower_click = st.number_input(
            f"Plausible low click rate{label_suffix}",
            min_value=0.001,
            max_value=0.999,
            value=0.05,
            step=0.01,
            key=f"{prefix}_low"
        )
        upper_click = st.number_input(
            f"Plausible high click rate{label_suffix}",
            min_value=0.001,
            max_value=0.999,
            value=0.20,
            step=0.01,
            key=f"{prefix}_high"
        )
        prior_ess = None

    return {
        "mean_click": mean_click,
        "prior_ess": prior_ess,
        "lower_click": lower_click,
        "upper_click": upper_click
    }

def validate_prior_dict(prior_dict, prior_mode):
    if prior_mode == "Mean + effective sample size":
        return validate_mean_ess(prior_dict["mean_click"], prior_dict["prior_ess"])
    return validate_mean_range(
        prior_dict["mean_click"],
        prior_dict["lower_click"],
        prior_dict["upper_click"]
    )

def derive_prior_from_dict(prior_dict, prior_mode):
    if prior_mode == "Mean + effective sample size":
        return beta_from_mean_and_ess(prior_dict["mean_click"], prior_dict["prior_ess"])
    return fit_beta_from_mean_and_interval(
        mean=prior_dict["mean_click"],
        lower=prior_dict["lower_click"],
        upper=prior_dict["upper_click"],
        interval_mass=0.95
    )

colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

st.subheader("Step 1. Choose the number of phishing conditions")
k = st.number_input("Number of arms", min_value=1, max_value=6, value=1, step=1)

st.subheader("Step 2. Choose prior structure")
same_prior_for_all = st.toggle("Use the same prior for all arms", value=True)

st.subheader("Step 3. Choose how to specify priors")
prior_mode = st.radio(
    "Prior input method",
    options=["Mean + effective sample size", "Mean + plausible range"],
    help=(
        "Mean + effective sample size is often easiest when you know roughly how much prior weight "
        "you want. Mean + plausible range treats the low and high values as an approximate central 95% prior interval."
    )
)

arm_names = []
st.subheader("Step 4. Name the arms")
for i in range(k):
    arm_name = st.text_input(
        f"Arm name {i+1}",
        value=f"Arm {i+1}",
        key=f"name_{i}"
    )
    arm_names.append(arm_name)

common_prior = None
arm_specific_priors = {}

st.subheader("Step 5. Enter priors")

if same_prior_for_all:
    st.markdown("### Common prior for all arms")
    common_prior = get_prior_inputs("common_prior", prior_mode, " for all arms")
else:
    for i in range(k):
        st.markdown(f"### Prior for {arm_names[i]}")
        arm_specific_priors[i] = get_prior_inputs(
            f"armprior_{i}",
            prior_mode,
            f" for {arm_names[i]}"
        )

# Build prior-only records first
prior_records = []
prior_valid = True
prior_errors = []
prior_highs = []
prior_param_list = []
prior_summary = []

for i in range(k):
    if same_prior_for_all:
        prior_dict = common_prior.copy()
    else:
        prior_dict = arm_specific_priors[i].copy()

    arm_name = arm_names[i]

    if not validate_prior_dict(prior_dict, prior_mode):
        prior_valid = False
        if prior_mode == "Mean + effective sample size":
            prior_errors.append(
                f'{arm_name}: expected click rate must be between 0 and 1, and prior effective sample size must be greater than 0.'
            )
        else:
            prior_errors.append(
                f'{arm_name}: the plausible low click rate must be below the expected click rate, and the expected click rate must be below the plausible high click rate.'
            )
        continue

    prior_a, prior_b = derive_prior_from_dict(prior_dict, prior_mode)
    prior_mean = prior_a / (prior_a + prior_b)
    prior_low, prior_high = safe_beta_quantiles(prior_a, prior_b)
    prior_ess = prior_a + prior_b

    prior_records.append({
        "name": arm_name,
        "color": colors[i],
        "prior_dict": prior_dict,
        "prior_a": prior_a,
        "prior_b": prior_b,
        "prior_mean": prior_mean,
        "prior_low": prior_low,
        "prior_high": prior_high,
        "prior_ess": prior_ess
    })

    prior_param_list.append(prior_signature(prior_a, prior_b))
    prior_highs.append(prior_high)

    row = {
        "Arm": arm_name,
        "Prior mean": round(prior_mean, 4),
        "Prior alpha": round(prior_a, 3),
        "Prior beta": round(prior_b, 3),
        "Prior effective sample size": round(prior_ess, 3),
        "Prior 95% low": round(prior_low, 4),
        "Prior 95% high": round(prior_high, 4),
        "Prior 95% CI": f"[{prior_low:.3f}, {prior_high:.3f}]"
    }

    if prior_mode == "Mean + effective sample size":
        row["User-entered prior ESS"] = round(prior_dict["prior_ess"], 3)

    prior_summary.append(row)

st.subheader("Step 6. Inspect the implied prior distributions")

if prior_valid:
    all_priors_identical = len(set(prior_param_list)) == 1
    prior_xmax = compute_dynamic_xmax(prior_highs, padding=0.20)
    x_prior = np.linspace(0.0001, prior_xmax, 1200)

    fig_prior, ax_prior = plt.subplots(figsize=(10, 5))

    if all_priors_identical:
        p0 = prior_records[0]
        y_prior = safe_beta_pdf(x_prior, p0["prior_a"], p0["prior_b"])

        shade_interval_under_curve(
            ax_prior, x_prior, y_prior, p0["prior_low"], p0["prior_high"], "black", alpha=0.12
        )

        ax_prior.plot(
            x_prior,
            y_prior,
            linestyle="--",
            color="black",
            linewidth=2.2,
            alpha=0.95,
            label="Common prior"
        )

        mean_height = safe_beta_mean_height(p0["prior_mean"], p0["prior_a"], p0["prior_b"])
        if mean_height is not None:
            ax_prior.vlines(
                p0["prior_mean"],
                0,
                mean_height,
                color="black",
                linestyle=":",
                linewidth=1.8,
                alpha=0.9
            )
            add_mean_label(
                ax_prior,
                p0["prior_mean"],
                mean_height,
                "black",
                f"{p0['prior_mean']:.3f}"
            )
    else:
        for rec in prior_records:
            y_prior = safe_beta_pdf(x_prior, rec["prior_a"], rec["prior_b"])

            shade_interval_under_curve(
                ax_prior, x_prior, y_prior, rec["prior_low"], rec["prior_high"], rec["color"], alpha=0.12
            )

            ax_prior.plot(
                x_prior,
                y_prior,
                linestyle="--",
                color=rec["color"],
                linewidth=2,
                alpha=0.95,
                label=rec["name"]
            )

            mean_height = safe_beta_mean_height(rec["prior_mean"], rec["prior_a"], rec["prior_b"])
            if mean_height is not None:
                ax_prior.vlines(
                    rec["prior_mean"],
                    0,
                    mean_height,
                    color=rec["color"],
                    linestyle=":",
                    linewidth=1.8,
                    alpha=0.9
                )
                add_mean_label(
                    ax_prior,
                    rec["prior_mean"],
                    mean_height,
                    rec["color"],
                    f"{rec['prior_mean']:.3f}"
                )

    ax_prior.set_xlim(0, prior_xmax)
    ax_prior.set_title("Implied prior distributions for click rate")
    ax_prior.set_xlabel("Click rate")
    ax_prior.set_ylabel("Density")
    ax_prior.legend()
    st.pyplot(fig_prior)

    st.dataframe(pd.DataFrame(prior_summary), use_container_width=True)

    if all_priors_identical:
        st.caption(
            "The shaded area under the curve shows the central 95% prior interval. "
            "The dashed line shows the prior density. "
            "The dotted line and label mark the prior mean."
        )
    else:
        st.caption(
            "Shaded areas under the curves show the central 95% prior intervals. "
            "Dashed lines show the prior densities. "
            "Dotted lines and labels mark the prior means."
        )
else:
    for msg in prior_errors:
        st.error(msg)

st.subheader("Step 7. Enter observed data for each arm")

variants = []
for i in range(k):
    st.markdown(f"### {arm_names[i]}")

    col1, col2 = st.columns(2)

    with col1:
        clicks = st.number_input(
            f"Observed clicks for {arm_names[i]}",
            min_value=0,
            value=10,
            step=1,
            key=f"clicks_{i}"
        )

    with col2:
        no_clicks = st.number_input(
            f"Observed no-clicks for {arm_names[i]}",
            min_value=0,
            value=90,
            step=1,
            key=f"noclicks_{i}"
        )

    prior_rec = prior_records[i] if prior_valid else None

    variants.append({
        "name": arm_names[i],
        "clicks": clicks,
        "no_clicks": no_clicks,
        "color": colors[i],
        "prior_rec": prior_rec
    })

run_analysis = st.button("Run Bayesian update", disabled=not prior_valid)

if run_analysis and prior_valid:
    st.subheader("Step 8. Priors and posteriors")

    results = []
    updating_rows = []
    posterior_draws = []
    posterior_highs = []

    all_priors_identical = len(set(prior_param_list)) == 1

    for v in variants:
        prior_a = v["prior_rec"]["prior_a"]
        prior_b = v["prior_rec"]["prior_b"]
        prior_mean = v["prior_rec"]["prior_mean"]
        prior_low = v["prior_rec"]["prior_low"]
        prior_high = v["prior_rec"]["prior_high"]
        prior_ess = v["prior_rec"]["prior_ess"]

        post_a = prior_a + v["clicks"]
        post_b = prior_b + v["no_clicks"]

        post_mean = post_a / (post_a + post_b)
        post_low, post_high = safe_beta_quantiles(post_a, post_b)

        posterior_highs.append(post_high)

        draws = np.random.beta(post_a, post_b, size=20000)
        posterior_draws.append(draws)

        observed_n = v["clicks"] + v["no_clicks"]

        results.append({
            "Arm": v["name"],
            "Prior mean": round(prior_mean, 4),
            "Prior 95% low": round(prior_low, 4),
            "Prior 95% high": round(prior_high, 4),
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
            "Arm": v["name"],
            "Prior alpha": round(prior_a, 3),
            "Prior beta": round(prior_b, 3),
            "Observed clicks": v["clicks"],
            "Observed no-clicks": v["no_clicks"],
            "Posterior alpha = prior alpha + clicks": round(post_a, 3),
            "Posterior beta = prior beta + no-clicks": round(post_b, 3)
        })

    post_xmax = compute_dynamic_xmax(posterior_highs, padding=0.20)
    x_post = np.linspace(0.0001, post_xmax, 1200)

    fig_post, ax_post = plt.subplots(figsize=(10, 5))

    for v in variants:
        prior_a = v["prior_rec"]["prior_a"]
        prior_b = v["prior_rec"]["prior_b"]
        prior_low = v["prior_rec"]["prior_low"]
        prior_high = v["prior_rec"]["prior_high"]

        post_a = prior_a + v["clicks"]
        post_b = prior_b + v["no_clicks"]
        post_mean = post_a / (post_a + post_b)
        post_low, post_high = safe_beta_quantiles(post_a, post_b)

        prior_line_color = "black" if all_priors_identical else v["color"]

        y_prior = safe_beta_pdf(x_post, prior_a, prior_b)
        y_post = safe_beta_pdf(x_post, post_a, post_b)

        shade_interval_under_curve(
            ax_post, x_post, y_prior, prior_low, prior_high, prior_line_color, alpha=0.06
        )
        shade_interval_under_curve(
            ax_post, x_post, y_post, post_low, post_high, v["color"], alpha=0.12
        )

        ax_post.plot(
            x_post,
            y_prior,
            linestyle="--",
            color=prior_line_color,
            linewidth=2,
            alpha=0.9
        )

        ax_post.plot(
            x_post,
            y_post,
            linestyle="-",
            color=v["color"],
            linewidth=2.5,
            alpha=0.55,
            label=v["name"]
        )

        post_mean_height = safe_beta_mean_height(post_mean, post_a, post_b)
        if post_mean_height is not None:
            ax_post.vlines(
                post_mean,
                0,
                post_mean_height,
                color=v["color"],
                linestyle="-",
                linewidth=1.6,
                alpha=0.9
            )
            add_mean_label(
                ax_post,
                post_mean,
                post_mean_height,
                v["color"],
                f"{post_mean:.3f}"
            )

    if len(variants) > 1:
        posterior_draws = np.array(posterior_draws)
        winners = np.argmax(posterior_draws, axis=0)
        prob_highest = [(winners == i).mean() for i in range(len(variants))]

        for i in range(len(results)):
            results[i]["Pr(highest click rate)"] = round(prob_highest[i], 4)

    ax_post.set_xlim(0, post_xmax)
    ax_post.set_title("Dashed priors and solid posteriors for click rate")
    ax_post.set_xlabel("Click rate")
    ax_post.set_ylabel("Density")
    ax_post.legend()
    st.pyplot(fig_post)

    st.dataframe(pd.DataFrame(results), use_container_width=True)

    st.subheader("Bayesian updating table")
    st.dataframe(pd.DataFrame(updating_rows), use_container_width=True)

    with st.expander("How the update works"):
        st.write(
            "For each arm, the prior is a Beta distribution over the click rate. "
            "Observed clicks update alpha, and observed no-clicks update beta."
        )
        st.code(
            "Posterior alpha = Prior alpha + Clicks\n"
            "Posterior beta  = Prior beta + No-clicks"
        )
        st.write(
            "On both plots, the shaded area stays under the curve and marks the central 95% interval. "
            "Dashed curves are priors. Solid curves are posteriors. "
            "Posterior means are labeled on the posterior plot."
        )