import numpy as np
import tabulate

from ..constants import Dtype
from ..datasets.base import DatasetBase
from ..reporting.footnotes import TableFootnote
from ..reporting.styling import Report, add_mpl_figure, set_column_width, write_cell


def write_dataset(report: Report, dataset: DatasetBase, header_level: int):
    # TODO - doses dropped
    # TODO - dataset name (@ session level); session name if exists, else datset name, else dataset id, else "BMDS Session"
    styles = report.styles
    footnotes = TableFootnote()

    report.document.add_paragraph("Input dataset", styles.header_2)
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


def write_summary_table(report: Report, session, header_level: int):
    # TODO - collapse models
    # TODO - dose units text
    # TODO - summary notes
    # TODO - add pvalue; for dichotomous it's `model.results.gof.p_value`, for continuous?
    styles = report.styles
    report.document.add_paragraph("Summary table", styles.header_2)
    hdr = report.styles.tbl_header
    body = report.styles.tbl_body

    footnotes = TableFootnote()
    tbl = report.document.add_table(len(session.models) + 2, 6, style=styles.table)

    write_cell(tbl.cell(0, 0), "Model", style=hdr)
    write_cell(tbl.cell(0, 1), "Goodness of fit", style=hdr)
    write_cell(tbl.cell(1, 1), "", style=hdr)  # set style
    write_cell(tbl.cell(1, 2), "AIC", style=hdr)
    write_cell(tbl.cell(0, 3), "BMD", style=hdr)
    write_cell(tbl.cell(0, 4), "BMDL", style=hdr)
    write_cell(tbl.cell(0, 5), "Comments", style=hdr)

    p = tbl.cell(1, 1).paragraphs[0]
    p.add_run("p").italic = True
    p.add_run("-value")

    # merge header columns
    tbl.cell(0, 0).merge(tbl.cell(1, 0))
    tbl.cell(0, 1).merge(tbl.cell(0, 2))
    tbl.cell(0, 3).merge(tbl.cell(1, 3))
    tbl.cell(0, 4).merge(tbl.cell(1, 4))
    tbl.cell(0, 5).merge(tbl.cell(1, 5))

    # write body
    for i, model in enumerate(session.models):
        idx = i + 2
        write_cell(tbl.cell(idx, 0), model.name(), body)
        write_cell(tbl.cell(idx, 1), "-999", body)
        write_cell(tbl.cell(idx, 2), model.results.aic, body)
        write_cell(tbl.cell(idx, 3), model.results.bmd, body)
        write_cell(tbl.cell(idx, 4), model.results.bmdl, body)

    # write comments
    write_cell(tbl.cell(2, 5), "Comments...", body)
    tbl.cell(2, 5).merge(tbl.cell(len(session.models) + 1, 5))

    # set column width
    widths = np.array([1.75, 0.8, 0.8, 0.7, 0.7, 1.75])
    widths = widths / (widths.sum() / report.styles.portrait_width)
    for width, col in zip(widths, tbl.columns):
        set_column_width(col, width)

    # write footnote
    if len(footnotes) > 0:
        footnotes.add_footnote_text(report.document, report.styles.tbl_footnote)


def write_models(report: Report, session, header_level: int):
    styles = report.styles
    report.document.add_paragraph("Models", styles.header_2)
    for model in session.models:
        report.document.add_paragraph(model.name(), styles.header_2)

        if not model.has_results:
            report.document.add_paragraph(
                "Model execution failed. No reports returned...", styles.tbl_body
            )
            continue

        # print figure
        add_mpl_figure(report.document, model.plot(), 6)

        # demo - build params table
        report.document.add_paragraph(
            "This is just an example of what can be done...", styles.tbl_body
        )
        tbl = tabulate.tabulate(
            [model.results.fit.params], headers=model.bmd_model_class.params, tablefmt="fancy_grid",
        )
        report.document.add_paragraph(tbl, styles.outfile)
