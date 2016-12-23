import matplotlib as mpl
import matplotlib.pyplot as plt

__all__ = []

PLOT_FIGSIZE = (8, 5)
PLOT_MARGINS = 0.05
DATASET_POINT_FORMAT = dict(
    ms=7,
    fmt='o',
    c='k'
)
DATASET_INDIVIDUAL_FORMAT = dict(
    s=35,
    alpha=0.60,
    c='k'
)
LINE_FORMAT = dict(
    c='#6470C0',
    lw=3
)
BMD_LINE_FORMAT = dict(
    c='#BFC05D',
    lw=2
)
BMD_LABEL_FORMAT = dict(
    size=9
)
FAILURE_MESSAGE_FORMAT = dict(
    style='italic',
    weight='bold',
    bbox={'facecolor': 'red', 'alpha': 0.35, 'pad': 10},
    horizontalalignment='center',
    verticalalignment='center'
)


def create_empty_figure():
    plt.style.use('seaborn-darkgrid')
    mpl.rcParams.update({'font.size': 10})
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    return fig
