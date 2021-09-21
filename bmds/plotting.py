from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt  # noqa

mpl.use("Agg")  # prevent matplotlib framework issues

__all__ = []

PLOT_FIGSIZE = (8, 5)
DPI = 100
PLOT_MARGINS = 0.05
DATASET_POINT_FORMAT = dict(ms=7, fmt="o", c="k", capsize=3, lw=1)
DATASET_INDIVIDUAL_FORMAT = dict(s=35, alpha=0.60, c="k")
LEGEND_OPTS = dict(loc="best", fontsize=8, fancybox=True, frameon=True)
LINE_FORMAT = dict(c="#6470C0", lw=3)
INDIVIDUAL_MODEL_COLORS = ["#6e40aa", "#e7298a", "#1b9e77", "#cc7939", "#666666"]
INDIVIDUAL_LINE_STYLES = ["solid", "dotted", "dashed", "dashdot"]
BMD_LINE_FORMAT = dict(c="#BFC05D", lw=2)
BMD_LABEL_FORMAT = dict(size=9)
FAILURE_MESSAGE_FORMAT = dict(
    style="italic",
    weight="bold",
    bbox={"facecolor": "red", "alpha": 0.35, "pad": 10},
    horizontalalignment="center",
    verticalalignment="center",
)
CDF_X_LABEL = "Dose"
CDF_Y_LABEL = "Percentile"
CDF_TITLE = "BMD Cumulative distribution function"


def create_empty_figure():
    plt.style.use("seaborn-darkgrid")
    mpl.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=DPI)
    return fig


def close_figure(fig):
    plt.close(fig)


def add_bmr_lines(
    ax, bmd: Optional[float] = None, bmdl: Optional[float] = None, bmd_y: Optional[float] = None
):
    xdomain = ax.xaxis.get_view_interval()
    xrng = xdomain[1] - xdomain[0]

    if bmd and bmd > 0:
        ax.plot([0, bmd], [bmd_y, bmd_y], **BMD_LINE_FORMAT)
        ax.plot([bmd, bmd], [0, bmd_y], **BMD_LINE_FORMAT)
        ax.text(
            bmd + xrng * 0.01,
            0,
            "BMD",
            label="BMR, BMD, BMDL",
            horizontalalignment="left",
            verticalalignment="center",
            **BMD_LABEL_FORMAT,
        )

    if bmdl and bmdl > 0:
        ax.plot([bmdl, bmdl], [0, bmd_y], **BMD_LINE_FORMAT)
        ax.text(
            bmdl - xrng * 0.01,
            0,
            "BMDL",
            horizontalalignment="right",
            verticalalignment="center",
            **BMD_LABEL_FORMAT,
        )
