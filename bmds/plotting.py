import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

mpl.use("Agg")  # prevent matplotlib framework issues

__all__ = []

PLOT_FIGSIZE = (8, 5)
DPI = 100
PLOT_MARGINS = 0.05
DATASET_POINT_FORMAT = dict(ms=7, fmt="o", c="k", capsize=3, lw=1, zorder=50)
DATASET_INDIVIDUAL_FORMAT = dict(s=35, color=to_rgba("#ffffff", 0.5), edgecolors="black")
LEGEND_OPTS = dict(loc="best", fontsize=9, frameon=True, facecolor="white", markerscale=0.5)
LINE_FORMAT = dict(c="#6470C0", lw=3, zorder=100)
AXLINE_FORMAT = dict(c="#b8a800", lw=3, zorder=100)
INDIVIDUAL_MODEL_COLORS = ["#6e40aa", "#e7298a", "#1b9e77", "#b8a800", "#666666"]
INDIVIDUAL_LINE_STYLES = ["solid", "dotted", "dashed", "dashdot"]
BMD_LABEL_FORMAT = dict(size=9)
BMD_LINE_FORMAT = dict(
    c="#6470C0",
    markeredgecolor="white",
    markeredgewidth=2,
    fmt="d",
    ecolor=to_rgba("#6470C0", 0.7),
    ms=12,
    elinewidth=7,
    zorder=150,
)
FAILURE_MESSAGE_FORMAT = dict(
    style="italic",
    weight="bold",
    bbox={"facecolor": "red", "alpha": 0.35, "pad": 10},
    horizontalalignment="center",
    verticalalignment="center",
)


def create_empty_figure(figsize: tuple[float, float] | None = None):
    plt.style.use("seaborn-v0_8-darkgrid")
    mpl.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=figsize or PLOT_FIGSIZE, dpi=DPI)
    return fig


def close_figure(fig):
    plt.close(fig)


def add_bmr_lines(
    ax, bmd: float, bmd_y: float, bmdl: float, bmdu: float, axlines: bool = False, **kw
):
    if bmd <= 0:
        return
    if axlines:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.axhline(
            y=bmd_y,
            xmin=0,
            xmax=(bmd + abs(min(xmin, 0))) / abs(xmin - xmax),
            # label="BMD/BMDL",
            **AXLINE_FORMAT,
        )
        ax.axvline(
            x=bmd,
            ymin=0,
            ymax=(bmd_y + abs(min(ymin, 0))) / abs(ymin - ymax),
            # label="BMD/BMDL",
            **AXLINE_FORMAT,
        )
        if bmdl > 0:
            ax.axvline(
                x=bmdl,
                ymin=0,
                ymax=(bmd_y + abs(min(ymin, 0))) / abs(ymin - ymax),
                label="BMD/BMDL",
                **AXLINE_FORMAT,
            )
    else:
        styles = {**BMD_LINE_FORMAT, **kw}
        lower = 0 if bmdl < 0 else bmd - bmdl
        upper = 0 if bmdu < 0 else bmdu - bmd
        ax.errorbar(bmd, bmd_y, xerr=[[lower], [upper]], **styles)
