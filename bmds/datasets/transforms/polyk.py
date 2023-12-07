"""
Poly K adjustment test, original citation:

Bailer AJ, Portier CJ. Effects of treatment-induced mortality and tumor-induced mortality on tests
for carcinogenicity in small samples. Biometrics. 1988 Jun;44(2):417-31.
PMID: 3390507. DOI: 10.2307/2531856
"""

from io import BytesIO
from itertools import cycle

import matplotlib.ticker as mtick
import pandas as pd

from ... import plotting
from ...reporting.styling import Report, add_mpl_figure, write_cell


def adjust_n(df: pd.DataFrame, k: float | None = 3, max_day: int | None = None) -> pd.DataFrame:
    """Adjust the n for individual observations in a dataset.

    Args:
        df (pd.DataFrame): a DataFrame of dataset used to manipulate. Three columns:
            - dose (float >=0)
            - day (integer >=0)
            - has_tumor (integer, 0 or 1)
        k (Optional[float], optional, default 3): The adjustment term to apply
        max_day (Optional[int], optional): The maximum data. If specific, the value is used,
            otherwise, it is calculated from the maximum reported day in the dataset

    Returns:
        pd.DataFrame: A copy of the original dataframe, with an new column `adj_n`
    """
    columns = ["dose", "day", "has_tumor"]
    if df.columns.tolist() != columns:
        raise ValueError(f"Unexpected column names; expecting {columns}")
    if set(df.has_tumor.unique()) != {0, 1}:
        raise ValueError("Expected `has_tumor` values must be 0 and 1")
    df = df.copy()
    if max_day is None:
        max_day = df.day.max()
    df.loc[:, "adj_n"] = (df.query("has_tumor==0").day / max_day) ** k
    df.loc[:, "adj_n"] = df.loc[:, "adj_n"].fillna(1).clip(upper=1)
    return df


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate summary statistics by group for poly k adjusted data, which can be used for
    dichotomous dose-response modeling.

    Args:
        df (pd.DataFrame): The input dataframe of individual response data with adjustment
            calculated from the `adjust_n` method above.

    Returns:
        pd.DataFrame: A dataframe of group-level values, both adjusted and unadjusted
    """
    columns = ["dose", "day", "has_tumor", "adj_n"]
    if df.columns.tolist() != columns:
        raise ValueError(f"Unexpected column names: {columns}")
    grouped = df.groupby("dose")
    df2 = pd.DataFrame(
        data=[
            grouped.has_tumor.count().rename("n", inplace=True),
            grouped.adj_n.sum().rename("adj_n", inplace=True),
            grouped.has_tumor.sum().rename("incidence", inplace=True),
        ]
    ).T.reset_index()
    df2.loc[:, "proportion"] = df2.incidence / df2.n
    df2.loc[:, "adj_proportion"] = df2.incidence / df2.adj_n
    return df2


def calculate(
    doses: list[float],
    day: list[int],
    has_tumor: list[int],
    k: float | None = 3,
    max_day: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate polyk adjustment on a dataset

    Args:
        doses (list[float]): A list of dose values
        day (list[int]): A list of days when observed
        has_tumor (list[int]): Binary flag for if entity had a tumor
        k (Optional[float], optional): Poly k adjustment value; defaults to 3.
        max_day (Optional[int], optional): Maximum observation day; defaults to calculating from
            dataset,

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Two data frames. The first is the individual data
            with the adjusted value, the second is summary data showing adjusted and unadjusted
            incidence data for use in dichotomous dose response modeling.
    """
    df = pd.DataFrame(dict(dose=doses, day=day, has_tumor=has_tumor))
    df2 = adjust_n(df, k, max_day)
    df3 = summary_stats(df2)
    return df2, df3


class Adjustment:
    def __init__(
        self,
        doses: list[float],
        day: list[int],
        has_tumor: list[int],
        k: float | None = 3,
        max_day: float | None = None,
    ) -> None:
        self.input_data = pd.DataFrame(dict(dose=doses, day=day, has_tumor=has_tumor))
        self.adjusted_data = adjust_n(self.input_data, k, max_day)
        self.summary = summary_stats(self.adjusted_data)

    def summary_figure(self, units: str = ""):
        fig = plotting.create_empty_figure()
        ax = fig.gca()
        ax.set_xlabel(f"Dose ({units})" if units else "Dose")
        ax.set_ylabel("Proportion (%)")
        ax.margins(plotting.PLOT_MARGINS)
        ax.set_title("Adjusted Proportion vs Original Proportion")
        ax.plot(
            self.summary.dose,
            self.summary.proportion,
            "o-",
            color="blue",
            label="Original Proportion",
            markersize=8,
            markeredgewidth=1,
            markeredgecolor="white",
        )
        ax.plot(
            self.summary.dose,
            self.summary.adj_proportion,
            "^-",
            color="red",
            label="Adjusted Proportion",
            markersize=8,
            markeredgewidth=1,
            markeredgecolor="white",
        )

        legend = ax.legend(**plotting.LEGEND_OPTS)
        for handle in legend.legend_handles:
            handle.set_markersize(8)

        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        return fig

    def tumor_incidence_figure(self, xunits: str = "", yunits: str = ""):
        markers = "o^svDP"
        df = self.input_data.copy()
        df.loc[:, "cummulative_tumor"] = df.groupby("dose").has_tumor.cumsum()

        marker = cycle(markers)

        fig = plotting.create_empty_figure()
        ax = fig.gca()
        ax.set_xlabel(f"Study duration ({xunits})" if xunits else "Study duration")
        ax.set_ylabel("Cumulative tumor incidence")
        ax.margins(plotting.PLOT_MARGINS)
        ax.set_title("Tumor incidence over study duration")

        if yunits:
            yunits = f" {yunits}"

        for value, d in df.groupby("dose"):
            ax.plot(
                d.day,
                d.cummulative_tumor,
                f"{next(marker)}-",
                label=str(value) + yunits,
                markersize=8,
                markeredgewidth=1,
                markeredgecolor="white",
            )

        legend = ax.legend(**plotting.LEGEND_OPTS)
        for handle in legend.legend_handles:
            handle.set_markersize(8)

        return fig

    def write_docx_inputs_table(self, report: Report):
        """Add an input data table to the document."""
        hdr = report.styles.tbl_header
        body = report.styles.tbl_body
        tbl = report.document.add_table(len(self.input_data) + 1, 3, style=report.styles.table)

        write_cell(tbl.cell(0, 0), "dose", style=hdr)
        write_cell(tbl.cell(0, 1), "day", style=hdr)
        write_cell(tbl.cell(0, 2), "has_tumor", style=hdr)

        for idx, v in enumerate(
            zip(self.input_data.dose, self.input_data.day, self.input_data.has_tumor, strict=True)
        ):
            write_cell(tbl.cell(idx + 1, 0), v[0], style=body)
            write_cell(tbl.cell(idx + 1, 1), v[1], style=body)
            write_cell(tbl.cell(idx + 1, 2), v[2], style=body)

    def write_docx_summary_table(self, report: Report):
        """Add a 'result'' data table with adjusted figures to the document."""
        hdr = report.styles.tbl_header
        body = report.styles.tbl_body
        tbl = report.document.add_table(len(self.summary) + 1, 6, style=report.styles.table)

        write_cell(tbl.cell(0, 0), "dose", style=hdr)
        write_cell(tbl.cell(0, 1), "n", style=hdr)
        write_cell(tbl.cell(0, 2), "adj_n", style=hdr)
        write_cell(tbl.cell(0, 3), "incidence", style=hdr)
        write_cell(tbl.cell(0, 4), "proportion", style=hdr)
        write_cell(tbl.cell(0, 5), "adj_proportion", style=hdr)

        for idx, val in enumerate(
            zip(
                self.summary.dose,
                self.summary.n,
                self.summary.adj_n,
                self.summary.incidence,
                self.summary.proportion,
                self.summary.adj_proportion,
                strict=True,
            )
        ):
            write_cell(tbl.cell(idx + 1, 0), val[0], style=body)
            write_cell(tbl.cell(idx + 1, 1), val[1], style=body)
            write_cell(tbl.cell(idx + 1, 2), val[2], style=body)
            write_cell(tbl.cell(idx + 1, 3), val[3], style=body)
            write_cell(tbl.cell(idx + 1, 4), val[4], style=body)
            write_cell(tbl.cell(idx + 1, 5), val[5], style=body)

    def to_docx(
        self,
        report: Report | None = None,
        header_level: int = 1,
        show_title: bool = True,
    ):
        if report is None:
            report = Report.build_default()

        if show_title:
            h1 = report.styles.get_header_style(header_level)
            report.document.add_paragraph("Poly K Adjustment", h1)

        h2 = report.styles.get_header_style(header_level + 1)
        report.document.add_paragraph("Summary", h2)
        report.document.add_paragraph(self.write_docx_summary_table(report))
        report.document.add_paragraph(add_mpl_figure(report.document, self.summary_figure(), 6))
        report.document.add_paragraph("Plots", h2)
        report.document.add_paragraph(
            add_mpl_figure(report.document, self.tumor_incidence_figure(), 6)
        )
        report.document.add_paragraph("Table", h2)
        report.document.add_paragraph(self.write_docx_summary_table(report))
        report.document.add_paragraph("Data", h2)
        self.write_docx_inputs_table(report)

        return report.document

    def to_excel(self) -> BytesIO:
        f = BytesIO()
        with pd.ExcelWriter(f) as writer:
            for name, df in [("adjusted", self.adjusted_data), ("summary", self.summary)]:
                df.to_excel(writer, sheet_name=name, index=False)
        return f
