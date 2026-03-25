import matplotlib.pyplot as plt
from posteriors import safe_beta_pdf, safe_beta_mean_height


def compute_dynamic_xmax(high_values, padding=0.20):
    if len(high_values) == 0:
        return 1.0
    xmax = max(high_values) + padding
    xmax = min(1.0, xmax)
    xmax = max(0.2, xmax)
    return xmax


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
    mean_line_style=":",
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
        label=label,
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
                alpha=0.9,
            )
            add_mean_label(ax, mean, mean_height, color, f"{mean:.3f}")

    return y


def make_figure(title, xlabel="Click rate", ylabel="Density"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax
