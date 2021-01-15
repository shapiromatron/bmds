from ..constants import Dtype
from ..datasets.base import DatasetBase
from ..reporting.footnotes import TableFootnote
from ..reporting.styling import Report, set_column_width, write_cell


def write_dataset(report: Report, dataset: DatasetBase):
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


def write_summary_table(report: Report, session):
    pass


def write_models(report: Report, session):
    pass
