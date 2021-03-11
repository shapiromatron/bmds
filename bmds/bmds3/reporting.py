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
        write_cell(tbl.cell(idx, 2), model.results.fit.aic, body)
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

        # Info Table
        report.document.add_paragraph("Model Info", styles.header_2)
        hdr = styles.tbl_header
        body = styles.tbl_body
        tbl = report.document.add_table(2, 3, style=styles.table)

        write_cell(tbl.cell(0, 0), "Model Name", hdr)
        write_cell(tbl.cell(0, 1), "Dataset Name", hdr)
        write_cell(tbl.cell(0, 2), "Model Form", hdr)

        write_cell(tbl.cell(1, 0), model.name(), body)
        write_cell(tbl.cell(1, 1), model.dataset.metadata.name, body)
        write_cell(tbl.cell(1, 2), "To Do", body)

        if model.dataset.dtype is Dtype.CONTINUOUS:
            # Model Options
            report.document.add_paragraph("Model Options", styles.header_2)
            hdr = styles.tbl_header
            body = styles.tbl_body
            tbl = report.document.add_table(2, 5, style=styles.table)

            write_cell(tbl.cell(0, 0), "BMR Type", hdr)
            write_cell(tbl.cell(0, 1), "BMRF", hdr)
            write_cell(tbl.cell(0, 2), "Tail Probability", hdr)
            write_cell(tbl.cell(0, 3), "Confidence Level", hdr)
            write_cell(tbl.cell(0, 4), "Distribution + Variance", hdr)

            write_cell(tbl.cell(1, 0), model.settings.bmr_type, body)
            write_cell(tbl.cell(1, 1), model.settings.bmr, body)
            write_cell(tbl.cell(1, 2), model.settings.tail_prob, body)
            write_cell(tbl.cell(1, 3), 1 - model.settings.alpha, body)
            write_cell(tbl.cell(1, 4), model.settings.disttype, body)

            # Model Data
            report.document.add_paragraph("Model Data", styles.header_2)
            hdr = styles.tbl_header
            body = styles.tbl_body
            tbl = report.document.add_table(2, 4, style=styles.table)

            write_cell(tbl.cell(0, 0), "Dependent Variable", hdr)
            write_cell(tbl.cell(0, 1), "Independent Variable", hdr)
            write_cell(tbl.cell(0, 2), "Number of Observations", hdr)
            write_cell(tbl.cell(0, 3), "Adverse Direction", hdr)

            write_cell(tbl.cell(1, 0), "Dose", body)
            write_cell(tbl.cell(1, 1), "Mean", body)
            write_cell(tbl.cell(1, 2), len(model.dataset.doses), body)
            write_cell(tbl.cell(1, 3), "Up", body)

        if model.dataset.dtype is Dtype.DICHOTOMOUS:
            # Model Options
            report.document.add_paragraph("Model Options", styles.header_2)
            hdr = styles.tbl_header
            body = styles.tbl_body
            tbl = report.document.add_table(2, 3, style=styles.table)

            write_cell(tbl.cell(0, 0), "Risk Type", hdr)
            write_cell(tbl.cell(0, 1), "BMR", hdr)
            write_cell(tbl.cell(0, 2), "Confidence Level", hdr)

            write_cell(tbl.cell(1, 0), model.settings.bmr_type, body)
            write_cell(tbl.cell(1, 1), model.settings.bmr, body)
            write_cell(tbl.cell(1, 2), 1 - model.settings.alpha, body)

            # Model Data
            report.document.add_paragraph("Model Data", styles.header_2)
            hdr = styles.tbl_header
            body = styles.tbl_body
            tbl = report.document.add_table(2, 4, style=styles.table)

            write_cell(tbl.cell(0, 0), "Dependent Variable", hdr)
            write_cell(tbl.cell(0, 1), "Independent Variable", hdr)
            write_cell(tbl.cell(0, 2), "Number of Observations", hdr)
            write_cell(tbl.cell(0, 3), "Adverse Direction", hdr)

            write_cell(tbl.cell(1, 0), "Dose", body)
            write_cell(tbl.cell(1, 1), "Fraction Affected", body)
            write_cell(tbl.cell(1, 2), len(model.dataset.doses), body)
            write_cell(tbl.cell(1, 3), "Up", body)

        # print figure
        add_mpl_figure(report.document, model.plot(), 6)

        # continuous summary
        if model.dataset.dtype is Dtype.CONTINUOUS:
            # Continuous Summary
            report.document.add_paragraph("Continuous Summary", styles.header_2)
            hdr = styles.tbl_header
            body = styles.tbl_body
            tbl = report.document.add_table(2, 7, style=styles.table)

            write_cell(tbl.cell(0, 0), "BMD", style=hdr)
            write_cell(tbl.cell(0, 1), "BMDL", style=hdr)
            write_cell(tbl.cell(0, 2), "BMDU", style=hdr)  # set style
            write_cell(tbl.cell(0, 3), "AIC", style=hdr)
            write_cell(tbl.cell(0, 4), "LL", style=hdr)
            write_cell(tbl.cell(0, 5), "Model DF", style=hdr)
            write_cell(tbl.cell(0, 6), "hi-squared", style=hdr)

            write_cell(tbl.cell(1, 0), model.results.bmd, body)
            write_cell(tbl.cell(1, 1), model.results.bmdl, body)
            write_cell(tbl.cell(1, 2), model.results.bmdu, body)
            write_cell(tbl.cell(1, 3), model.results.fit.aic, body)
            write_cell(tbl.cell(1, 4), model.results.fit.loglikelihood, body)
            write_cell(tbl.cell(1, 5), model.results.fit.model_df, body)
            write_cell(tbl.cell(1, 6), model.results.fit.chisq, body)

        # dichotomous summary
        if model.dataset.dtype is Dtype.DICHOTOMOUS:
            report.document.add_paragraph("Dichotomous Summary", styles.header_2)
            hdr = styles.tbl_header
            body = styles.tbl_body
            tbl = report.document.add_table(2, 9, style=styles.table)

            write_cell(tbl.cell(0, 0), "BMD", style=hdr)
            write_cell(tbl.cell(0, 1), "BMDL", style=hdr)
            write_cell(tbl.cell(0, 2), "BMDU", style=hdr)  # set style
            write_cell(tbl.cell(0, 3), "AIC", style=hdr)
            write_cell(tbl.cell(0, 4), "LL", style=hdr)
            write_cell(tbl.cell(0, 5), "Model DF", style=hdr)
            write_cell(tbl.cell(0, 6), "p value", style=hdr)
            write_cell(tbl.cell(0, 7), "DOF", style=hdr)
            write_cell(tbl.cell(0, 8), "chi-squared", style=hdr)

            # write body
            write_cell(tbl.cell(1, 0), model.results.bmd, body)
            write_cell(tbl.cell(1, 1), model.results.bmdl, body)
            write_cell(tbl.cell(1, 2), model.results.bmdu, body)
            write_cell(tbl.cell(1, 3), model.results.fit.aic, body)
            write_cell(tbl.cell(1, 4), model.results.fit.loglikelihood, body)
            write_cell(tbl.cell(1, 5), model.results.fit.model_df, body)
            write_cell(tbl.cell(1, 6), model.results.gof.p_value, body)
            write_cell(tbl.cell(1, 7), model.results.fit.total_df, body)
            write_cell(tbl.cell(1, 8), model.results.fit.chisq, body)

        report.document.add_paragraph("Model Parameters", styles.header_2)
        hdr = styles.tbl_header
        body = styles.tbl_body

        tbl = report.document.add_table(
            len(model.results.parameters.names) + 1, 3, style=styles.table
        )

        write_cell(tbl.cell(0, 0), "Variable", style=hdr)
        write_cell(tbl.cell(0, 1), "Parameter", style=hdr)
        write_cell(tbl.cell(0, 2), "Bounded", style=hdr)

        # write body
        for i, (name, value, bounded) in enumerate(
            zip(
                model.results.parameters.names,
                model.results.parameters.values,
                model.results.parameters.bounded,
            )
        ):
            idx = i + 1
            write_cell(tbl.cell(idx, 0), name, body)
            write_cell(tbl.cell(idx, 1), value, body)
            write_cell(tbl.cell(idx, 2), bounded, body)

        # Goodness of Fit -Continuous
        if model.dataset.dtype is Dtype.CONTINUOUS:
            report.document.add_paragraph("Goodness of Fit", styles.header_2)
            hdr = styles.tbl_header
            body = styles.tbl_body

            tbl = report.document.add_table(len(model.dataset.doses) + 1, 6, style=styles.table)

            write_cell(tbl.cell(0, 0), "Dose", style=hdr)
            write_cell(tbl.cell(0, 1), "Est. prob", style=hdr)
            write_cell(tbl.cell(0, 2), "Expected", style=hdr)  # set style
            write_cell(tbl.cell(0, 3), "Observed", style=hdr)
            write_cell(tbl.cell(0, 4), "Sized", style=hdr)
            write_cell(tbl.cell(0, 5), "Scaled Res.", style=hdr)

            # write body
            for i, (dose, est_mean, calc_mean, obs_mean, size, residual) in enumerate(
                zip(
                    model.dataset.doses,
                    model.results.gof.est_mean,
                    model.results.gof.calc_mean,
                    model.results.gof.obs_mean,
                    model.results.gof.size,
                    model.results.gof.residual,
                )
            ):
                idx = i + 1
                write_cell(tbl.cell(idx, 0), dose, body)
                write_cell(tbl.cell(idx, 1), est_mean, body)
                write_cell(tbl.cell(idx, 2), calc_mean, body)
                write_cell(tbl.cell(idx, 3), obs_mean, body)
                write_cell(tbl.cell(idx, 4), size, body)
                write_cell(tbl.cell(idx, 5), residual, body)

            report.document.add_paragraph("Analysis of Deviance", styles.header_2)
            hdr = styles.tbl_header
            body = styles.tbl_body

            tbl = report.document.add_table(
                len(model.results.deviance.names) + 1, 4, style=styles.table
            )

            write_cell(tbl.cell(0, 0), "Model", style=hdr)
            write_cell(tbl.cell(0, 1), "LL", style=hdr)
            write_cell(tbl.cell(0, 2), "Num Params", style=hdr)  # set style
            write_cell(tbl.cell(0, 3), "AIC", style=hdr)

            # write body
            for i, (name, ll, num_param, aic) in enumerate(
                zip(
                    model.results.deviance.names,
                    model.results.deviance.loglikelihoods,
                    model.results.deviance.num_params,
                    model.results.deviance.aics,
                )
            ):
                idx = i + 1
                write_cell(tbl.cell(idx, 0), name, body)
                write_cell(tbl.cell(idx, 1), ll, body)
                write_cell(tbl.cell(idx, 2), num_param, body)
                write_cell(tbl.cell(idx, 3), aic, body)

            report.document.add_paragraph("Test of Interest", styles.header_2)
            hdr = styles.tbl_header
            body = styles.tbl_body

            tbl = report.document.add_table(
                len(model.results.tests.names) + 1, 4, style=styles.table
            )

            write_cell(tbl.cell(0, 0), "Test", style=hdr)
            write_cell(tbl.cell(0, 1), "Likelihood Ratio", style=hdr)
            write_cell(tbl.cell(0, 2), "DF", style=hdr)  # set style
            write_cell(tbl.cell(0, 3), "P Value", style=hdr)

            # write body
            for i, (name, ll, df, p_value) in enumerate(
                zip(
                    model.results.tests.names,
                    model.results.tests.ll_ratios,
                    model.results.tests.dfs,
                    model.results.tests.p_values,
                )
            ):
                idx = i + 1
                write_cell(tbl.cell(idx, 0), name, body)
                write_cell(tbl.cell(idx, 1), ll, body)
                write_cell(tbl.cell(idx, 2), df, body)
                write_cell(tbl.cell(idx, 3), p_value, body)

        # Goodness of Fit -Dichotomous
        if model.dataset.dtype is Dtype.DICHOTOMOUS:
            report.document.add_paragraph("Goodness of Fit", styles.header_2)
            hdr = styles.tbl_header
            body = styles.tbl_body

            tbl = report.document.add_table(len(model.dataset.doses) + 1, 6, style=styles.table)

            write_cell(tbl.cell(0, 0), "Dose", style=hdr)
            write_cell(tbl.cell(0, 1), "Est. prob", style=hdr)
            write_cell(tbl.cell(0, 2), "Expected", style=hdr)  # set style
            write_cell(tbl.cell(0, 3), "Observed", style=hdr)
            write_cell(tbl.cell(0, 4), "Sized", style=hdr)
            write_cell(tbl.cell(0, 5), "Scaled Res.", style=hdr)

            # write body
            for i, (dose, expected, ns, incidence, residual) in enumerate(
                zip(
                    model.dataset.doses,
                    model.results.gof.expected,
                    model.dataset.ns,
                    model.dataset.incidences,
                    model.results.gof.residual,
                )
            ):
                idx = i + 1
                write_cell(tbl.cell(idx, 0), dose, body)
                write_cell(tbl.cell(idx, 1), expected / ns, body)
                write_cell(tbl.cell(idx, 2), expected, body)
                write_cell(tbl.cell(idx, 3), incidence, body)
                write_cell(tbl.cell(idx, 4), ns, body)
                write_cell(tbl.cell(idx, 5), residual, body)

            report.document.add_paragraph("Analysis of Deviance", styles.header_2)
            hdr = styles.tbl_header
            body = styles.tbl_body

            tbl = report.document.add_table(
                len(model.results.deviance.names) + 1, 6, style=styles.table
            )

            write_cell(tbl.cell(0, 0), "Model", style=hdr)
            write_cell(tbl.cell(0, 1), "LL", style=hdr)
            write_cell(tbl.cell(0, 2), "Num Params", style=hdr)  # set style
            write_cell(tbl.cell(0, 3), "Deviance", style=hdr)
            write_cell(tbl.cell(0, 4), "Test DF", style=hdr)
            write_cell(tbl.cell(0, 5), "P Value", style=hdr)

            # write body
            for i, (name, ll, parm, deviance, df, p_value) in enumerate(
                zip(
                    model.results.deviance.names,
                    model.results.deviance.ll,
                    model.results.deviance.params,
                    model.results.deviance.deviance,
                    model.results.deviance.df,
                    model.results.deviance.p_value,
                )
            ):
                idx = i + 1
                write_cell(tbl.cell(idx, 0), name, body)
                write_cell(tbl.cell(idx, 1), ll, body)
                write_cell(tbl.cell(idx, 2), parm, body)
                write_cell(tbl.cell(idx, 3), deviance, body)
                write_cell(tbl.cell(idx, 4), df, body)
                write_cell(tbl.cell(idx, 5), p_value, body)

        # CDF Table
        report.document.add_paragraph("CDF Table", styles.header_2)
        hdr = styles.tbl_header
        body = styles.tbl_body

        tbl = report.document.add_table(
            len(model.results.fit.bmd_dist[1]) + 1, 2, style=styles.table
        )

        write_cell(tbl.cell(0, 0), "Percentile", style=hdr)
        write_cell(tbl.cell(0, 1), "BMD", style=hdr)

        # write body
        for i, (percentile, bmd) in enumerate(
            zip(model.results.fit.bmd_dist[1], model.results.fit.bmd_dist[0])
        ):
            idx = i + 1
            write_cell(tbl.cell(idx, 0), percentile, body)
            write_cell(tbl.cell(idx, 1), bmd, body)
