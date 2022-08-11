import matplotlib as mpl
import matplotlib.pyplot as plt  # noqa

mpl.use("Agg")  # prevent matplotlib framework issues

__all__ = []

PLOT_FIGSIZE = (8, 5)
DPI = 100
PLOT_MARGINS = 0.05
DATASET_POINT_FORMAT = dict(ms=7, fmt="o", c="k", capsize=3, lw=1, zorder=100)
DATASET_INDIVIDUAL_FORMAT = dict(s=35, alpha=0.60, c="k")
LEGEND_OPTS = dict(loc="best", fontsize=9, frameon=True, facecolor="white", markerscale=0.6)
LINE_FORMAT = dict(c="#6470C0", lw=3, zorder=50)
INDIVIDUAL_MODEL_COLORS = ["#6e40aa", "#e7298a", "#1b9e77", "#b8a800", "#666666"]
INDIVIDUAL_LINE_STYLES = ["solid", "dotted", "dashed", "dashdot"]
BMD_LINE_FORMAT = dict(c="#cc7939", zorder=120, ms=9, fmt="o", elinewidth=2, capthick=2, capsize=6)
FAILURE_MESSAGE_FORMAT = dict(
    style="italic",
    weight="bold",
    bbox={"facecolor": "red", "alpha": 0.35, "pad": 10},
    horizontalalignment="center",
    verticalalignment="center",
)


def create_empty_figure():
    plt.style.use("seaborn-darkgrid")
    mpl.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=DPI)
    return fig


def close_figure(fig):
    plt.close(fig)


def add_bmr_lines(ax, bmd: float, bmd_y: float, bmdl: float, bmdu: float):
    if bmd <= 0:
        return

    lower = 0 if bmdl < 0 else bmd - bmdl
    upper = 0 if bmdu < 0 else bmdu - bmd

    ax.errorbar(
        bmd,
        bmd_y,
        xerr=[[lower], [upper]],
        label="BMD (BMDL, BMDU, and BMR)",
        **BMD_LINE_FORMAT,
    )
