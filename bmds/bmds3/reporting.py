from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..constants import Dtype
from ..datasets.base import DatasetBase
from ..reporting.footnotes import TableFootnote
from ..reporting.styling import Report, add_mpl_figure, set_column_width, write_cell

if TYPE_CHECKING:
    from .sessions import BmdsSession


def write_dataset(report: Report, dataset: DatasetBase):
    # TODO - doses dropped
    # TODO - dataset name (@ session level); session name if exists, else datset name, else dataset id, else "BMDS Session"
    styles = report.styles
    footnotes = TableFootnote()

    hdr = styles.tbl_header

    dose_units_text = dataset._get_dose_units_text()
    response_units_text = dataset._get_response_units_text()

    if dataset.dtype is Dtype.CONTINUOUS:
        tbl = report.document.add_table(3, dataset.num_dose_groups + 1, style=styles.table)

        write_cell(tbl.cell(0, 0), "Dose" + dose_units_text, hdr)
        write_cell(tbl.cell(1, 0), "N", hdr)
        write_cell(tbl.cell(2, 0), "Mean ± SD" + response_units_text, hdr)

        for i, (dose, n, mean, stdev) in enumerate(
            zip(dataset.doses, dataset.ns, dataset.means, dataset.stdevs)
        ):
            write_cell(tbl.cell(0, i + 1), dose, styles.tbl_body)
            write_cell(tbl.cell(1, i + 1), n, styles.tbl_body)
            write_cell(tbl.cell(2, i + 1), f"{mean} ± {stdev}", styles.tbl_body)

        for i, col in enumerate(tbl.columns):
            w = 0.75 if i == 0 else (styles.portrait_width - 0.75) / dataset.num_dose_groups
            set_column_width(col, w)

    elif dataset.dtype is Dtype.DICHOTOMOUS or dataset.dtype is Dtype.DICHOTOMOUS_CANCER:
        tbl = report.document.add_table(2, dataset.num_dose_groups + 1, style=styles.table)

        write_cell(tbl.cell(0, 0), "Dose" + dose_units_text, hdr)
        write_cell(tbl.cell(1, 0), "Affected / Total (%)" + response_units_text, hdr)

        for i, (dose, inc, n) in enumerate(zip(dataset.doses, dataset.incidences, dataset.ns)):
            frac = inc / float(n)
            write_cell(tbl.cell(0, i + 1), dose, styles.tbl_body)
            write_cell(tbl.cell(1, i + 1), f"{inc}/{n}\n({frac:.1%})", styles.tbl_body)

        for i, col in enumerate(tbl.columns):
            w = 0.75 if i == 0 else (styles.portrait_width - 0.75) / dataset.num_dose_groups
            set_column_width(col, w)

    elif dataset.dtype is Dtype.CONTINUOUS_INDIVIDUAL:
        raise NotImplementedError("TODO")

    else:
        raise ValueError("Unknown dtype: {dataset.dtype}")

    # write footnote
    if len(footnotes) > 0:
        footnotes.add_footnote_text(report.document, styles.tbl_footnote)


def write_frequentist_table(report: Report, session: BmdsSession):
    styles = report.styles
    hdr = report.styles.tbl_header
    body = report.styles.tbl_body

    footnotes = TableFootnote()
    tbl = report.document.add_table(len(session.models) + 1, 9, style=styles.table)

    write_cell(tbl.cell(0, 0), "Model", style=hdr)
    write_cell(tbl.cell(0, 1), "BMDL", style=hdr)
    write_cell(tbl.cell(0, 2), "BMD", style=hdr)
    write_cell(tbl.cell(0, 3), "BMDU", style=hdr)
    write_cell(tbl.cell(0, 4), "P value", style=hdr)
    write_cell(tbl.cell(0, 5), "AIC", style=hdr)
    write_cell(tbl.cell(0, 6), "Scaled Residual for Dose Group near BMD", style=hdr)
    write_cell(tbl.cell(0, 7), "Scaled Residual for Control Dose Group", style=hdr)
    write_cell(tbl.cell(0, 8), "Recommendation and Notes", style=hdr)

    # write body
    for idx, model in enumerate(session.models, start=1):
        write_cell(tbl.cell(idx, 0), model.name(), body)
        write_cell(tbl.cell(idx, 1), model.results.bmdl, body)
        write_cell(tbl.cell(idx, 2), model.results.bmd, body)
        write_cell(tbl.cell(idx, 3), model.results.bmdu, body)
        write_cell(tbl.cell(idx, 4), "-", body)
        write_cell(tbl.cell(idx, 5), model.results.fit.aic, body)
        write_cell(tbl.cell(idx, 6), model.results.gof.roi, body)
        write_cell(tbl.cell(idx, 7), model.results.gof.residual[0], body)
        write_cell(tbl.cell(idx, 8), "", body)

    # set column width
    widths = np.array([1.75, 0.8, 0.8, 0.7, 0.7, 0.7, 0.7, 0.7, 1.75])
    widths = widths / (widths.sum() / report.styles.portrait_width)
    for width, col in zip(widths, tbl.columns):
        set_column_width(col, width)

    # write footnote
    if len(footnotes) > 0:
        footnotes.add_footnote_text(report.document, report.styles.tbl_footnote)


def write_model_average_table(report: Report, session: BmdsSession):
    styles = report.styles
    hdr = styles.tbl_header
    body = styles.tbl_body
    tbl = report.document.add_table(2, 3, style=styles.table)

    write_cell(tbl.cell(0, 0), "BMDL", hdr)
    write_cell(tbl.cell(0, 1), "BMD", hdr)
    write_cell(tbl.cell(0, 2), "BMDU", hdr)

    write_cell(tbl.cell(1, 0), session.model_average.results.bmdl, body)
    write_cell(tbl.cell(1, 1), session.model_average.results.bmd, body)
    write_cell(tbl.cell(1, 2), session.model_average.results.bmdu, body)


def plot_bma(report: Report, session: BmdsSession):
    styles = report.styles
    report.document.add_paragraph("TODO - add", styles.tbl_body)


def write_bayesian_table(report: Report, session: BmdsSession):
    styles = report.styles
    report.document.add_paragraph()
    hdr = report.styles.tbl_header
    body = report.styles.tbl_body

    footnotes = TableFootnote()
    tbl = report.document.add_table(len(session.models) + 1, 9, style=styles.table)

    write_cell(tbl.cell(0, 0), "Model", style=hdr)
    write_cell(tbl.cell(0, 1), "Prior Weights", style=hdr)
    write_cell(tbl.cell(0, 2), "Posterior Weights", style=hdr)
    write_cell(tbl.cell(0, 3), "BMDL", style=hdr)
    write_cell(tbl.cell(0, 4), "BMD", style=hdr)
    write_cell(tbl.cell(0, 5), "BMDU", style=hdr)
    write_cell(tbl.cell(0, 6), "P Value", style=hdr)
    write_cell(tbl.cell(0, 7), "Scaled Residual for Dose Group near BMD", style=hdr)
    write_cell(tbl.cell(0, 8), "Scaled Residual for Control Dose Group", style=hdr)

    ma = session.model_average
    # write body
    for idx, model in enumerate(session.models, start=1):
        write_cell(tbl.cell(idx, 0), model.name(), body)
        if ma:
            write_cell(tbl.cell(idx, 1), ma.results.priors[idx - 1], body)
            write_cell(tbl.cell(idx, 2), ma.results.posteriors[idx - 1], body)
        else:
            write_cell(tbl.cell(idx, 1), "-", body)
            write_cell(tbl.cell(idx, 2), "-", body)
        write_cell(tbl.cell(idx, 3), model.results.bmdl, body)
        write_cell(tbl.cell(idx, 4), model.results.bmd, body)
        write_cell(tbl.cell(idx, 5), model.results.bmdu, body)
        write_cell(tbl.cell(idx, 6), "-", body)
        write_cell(tbl.cell(idx, 7), model.results.gof.roi, body)
        write_cell(tbl.cell(idx, 8), model.results.gof.residual[0], body)

    if ma:
        idx = len(tbl.rows)
        tbl.add_row()
        write_cell(tbl.cell(idx, 0), "Model Average", body)
        write_cell(tbl.cell(idx, 1), "-", body)
        write_cell(tbl.cell(idx, 2), "-", body)
        write_cell(tbl.cell(idx, 3), ma.results.bmdl, body)
        write_cell(tbl.cell(idx, 4), ma.results.bmd, body)
        write_cell(tbl.cell(idx, 5), ma.results.bmdu, body)
        write_cell(tbl.cell(idx, 6), "-", body)
        write_cell(tbl.cell(idx, 7), "-", body)
        write_cell(tbl.cell(idx, 8), "-", body)

    # set column width
    widths = np.array([1.0, 0.8, 0.8, 0.7, 0.7, 0.7, 0.7, 0.7, 2.5])
    widths = widths / (widths.sum() / report.styles.portrait_width)
    for width, col in zip(widths, tbl.columns):
        set_column_width(col, width)

    # write footnote
    if len(footnotes) > 0:
        footnotes.add_footnote_text(report.document, report.styles.tbl_footnote)


def write_models(report: Report, session: BmdsSession, header_level: int):
    styles = report.styles
    header_style = styles.get_header_style(header_level)
    for model in session.models:
        report.document.add_paragraph(model.name(), header_style)
        if model.has_results:
            report.document.add_paragraph(add_mpl_figure(report.document, model.plot(), 6))
        report.document.add_paragraph(model.text(), styles.fixed_width)
