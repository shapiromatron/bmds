from typing import Self

import numpy as np
import pandas as pd

from ... import bmdscore
from ...constants import Version
from ...datasets.dichotomous import DichotomousDataset
from ...reporting.footnotes import TableFootnote
from ...reporting.styling import Report, add_mpl_figure, set_column_width, write_cell
from ...version import __version__
from .. import reporting
from ..constants import NUM_PRIOR_COLS, PriorClass, PriorType
from ..reporting import write_pvalue_header
from ..types.dichotomous import DichotomousModelSettings
from ..types.multi_tumor import (
    MultitumorAnalysis,
    MultitumorResult,
    MultitumorSchema,
    MultitumorSettings,
)
from ..types.priors import ModelPriors, Prior
from ..types.sessions import VersionSchema
from .base import InputModelSettings
from .dichotomous import Multistage


def write_frequentist_table(report: Report, session):
    """Add frequentist table to document."""
    styles = report.styles
    hdr = report.styles.tbl_header
    body = report.styles.tbl_body

    footnotes = TableFootnote()
    tbl = report.document.add_table(len(session.models) + 1, 9, style=styles.table)

    write_cell(tbl.cell(0, 0), "Model", style=hdr)
    write_cell(tbl.cell(0, 1), "BMDL", style=hdr)
    write_cell(tbl.cell(0, 2), "BMD", style=hdr)
    write_cell(tbl.cell(0, 3), "BMDU", style=hdr)
    write_pvalue_header(tbl.cell(0, 4), style=hdr)
    write_cell(tbl.cell(0, 5), "AIC", style=hdr)
    write_cell(tbl.cell(0, 6), "Scaled Residual for Dose Group near BMD", style=hdr)
    write_cell(tbl.cell(0, 7), "Scaled Residual for Control Dose Group", style=hdr)
    write_cell(tbl.cell(0, 8), "Recommendation and Notes", style=hdr)

    # write body
    recommended_index = None
    # (
    #     session.recommender.results.recommended_model_index
    #     if session.has_recommended_model
    #     else None
    # )
    selected_index = 0
    recommendations = None
    for idx, model in enumerate(session.models):
        row = idx + 1
        write_cell(tbl.cell(row, 0), model[0].name(), body)
        # if recommended_index == idx:
        #     footnotes.add_footnote(tbl.cell(row, 0).paragraphs[0], "Recommended best-fitting model")
        # if selected_index == idx:
        #     footnotes.add_footnote(tbl.cell(row, 0).paragraphs[0], session.selected.notes)
        write_cell(tbl.cell(row, 1), model[0].results.bmdl, body)
        write_cell(tbl.cell(row, 2), model[0].results.bmd, body)
        write_cell(tbl.cell(row, 3), model[0].results.bmdu, body)
        write_cell(tbl.cell(row, 4), model[0].get_gof_pvalue(), body)
        write_cell(tbl.cell(row, 5), model[0].results.fit.aic, body)
        write_cell(tbl.cell(row, 6), model[0].results.gof.roi, body)
        write_cell(tbl.cell(row, 7), model[0].results.gof.residual[0], body)

        cell = tbl.cell(row, 8)
        if recommendations:
            p = cell.paragraphs[0]
            p.style = body
            run = p.add_run(recommendations.bin_text(idx) + "\n")
            run.bold = True
            p.add_run(recommendations.notes_text(idx))
        else:
            write_cell(tbl.cell(row, 8), "-", body)

    # set column width
    widths = np.array([1.75, 0.8, 0.8, 0.7, 0.7, 0.7, 0.7, 0.7, 1.75])
    widths = widths / (widths.sum() / styles.portrait_width)
    for width, col in zip(widths, tbl.columns, strict=True):
        set_column_width(col, width)

    # write footnote
    if len(footnotes) > 0:
        footnotes.add_footnote_text(report.document, styles.tbl_footnote)


def write_inputs_table(report: Report, session):
    """Add an input summary table to the document."""
    if len(session.models) == 0:
        raise ValueError("No models available")

    styles = report.styles
    hdr = report.styles.tbl_header
    body = report.styles.tbl_body

    rows = session.models[0][0].settings.docx_table_data()
    tbl = report.document.add_table(len(rows), 2, style=styles.table)
    for idx, (key, value) in enumerate(rows):
        write_cell(tbl.cell(idx, 0), key, style=hdr)
        write_cell(tbl.cell(idx, 1), value, style=hdr if idx == 0 else body)


def write_models(report: Report, session, bmd_cdf_table: bool, header_level: int):
    for model in session.models:
        write_model(report, model, bmd_cdf_table, header_level)


def write_model(report: Report, model, bmd_cdf_table: bool, header_level: int):
    styles = report.styles
    header_style = styles.get_header_style(header_level)
    report.document.add_paragraph(model[0].name(), header_style)
    # if model.has_results:
    report.document.add_paragraph(add_mpl_figure(report.document, model[0].plot(), 6))
    # if bmd_cdf_table:
    #     report.document.add_paragraph(add_mpl_figure(report.document, model.cdf_plot(), 6))
    report.document.add_paragraph(model[0].text(), styles.fixed_width)


def multistage_cancer_prior() -> ModelPriors:
    # fmt: off
    priors = [
        Prior(name="g",  type=PriorType.Uniform, initial_value=-17, stdev=0, min_value=-18, max_value=18),
        Prior(name="b1", type=PriorType.Uniform, initial_value=0.1, stdev=0, min_value=0, max_value=1e4),
        Prior(name="b2", type=PriorType.Uniform, initial_value=0.1, stdev=0, min_value=0, max_value=1e4),
    ]
    # fmt: on
    return ModelPriors(
        prior_class=PriorClass.frequentist_restricted, priors=priors, variance_priors=None
    )


def _model_name(result) -> str:
    degree = result.parameters.names[-1][-1]
    return f"Multistage {degree}°"


class MultitumorBase:
    version_str: str
    version_pretty: str
    version_tuple: tuple[int, ...]

    def __init__(
        self,
        datasets: list[DichotomousDataset],
        degrees: list[int] | None = None,
        model_settings: DichotomousModelSettings | dict | None = None,
        id: int | str | None = None,
        results: MultitumorResult | None = None,
    ):
        if len(datasets) == 0:
            raise ValueError("Must provide at least one dataset")
        self.id = id
        self.datasets = datasets
        for i, dataset in enumerate(datasets, start=1):
            if dataset.metadata.id is None:
                dataset.metadata.id = i
        self.degrees: list[int] = degrees or [0] * len(datasets)
        self.settings: DichotomousModelSettings = self.get_base_settings(model_settings)
        self.results = results
        self.structs: tuple | None = None
        self.models: list[list[MultistageCancer]] = []

    def get_base_settings(
        self, settings: DichotomousModelSettings | dict | None
    ) -> DichotomousModelSettings:
        if settings is None:
            return DichotomousModelSettings()
        elif isinstance(settings, DichotomousModelSettings):
            return settings
        else:
            return DichotomousModelSettings.parse_obj(settings)

    def to_cpp(self) -> MultitumorAnalysis:
        dataset_models = []
        dataset_results = []
        ns = []
        for i, dataset in enumerate(self.datasets):
            mc_models = []
            self.models.append(mc_models)
            models = []
            results = []
            ns.append(dataset.num_dose_groups)
            degree_i = self.degrees[i]
            degrees_i = (
                range(degree_i, degree_i + 1) if degree_i > 0 else range(2, dataset.num_dose_groups)
            )
            for degree in degrees_i:
                settings = self.settings.copy(
                    update=dict(degree=degree, priors=multistage_cancer_prior())
                )
                model = MultistageCancer(dataset, settings=settings)
                inputs = model._build_inputs()
                structs = inputs.to_cpp()
                models.append(structs.analysis)
                results.append(structs.result)
                mc_models.append(model)
            dataset_models.append(models)
            dataset_results.append(results)

        analysis = bmdscore.python_multitumor_analysis()
        analysis.BMD_type = self.settings.bmr_type.value
        analysis.BMR = self.settings.bmr
        analysis.alpha = self.settings.alpha
        analysis.degree = self.degrees
        analysis.models = dataset_models
        analysis.n = ns
        analysis.ndatasets = len(self.datasets)
        analysis.nmodels = [len(models) for models in dataset_models]
        analysis.prior = []
        analysis.prior_cols = NUM_PRIOR_COLS

        result = bmdscore.python_multitumor_result()
        result.ndatasets = len(self.datasets)
        result.nmodels = [len(results) for results in dataset_results]
        result.models = dataset_results

        return MultitumorAnalysis(analysis, result)

    def execute(self):
        self.structs = self.to_cpp()
        self.structs.execute()
        self.results = MultitumorResult.from_model(self)
        return self.results

    def text(self) -> str:
        return self.results.text(self.datasets, self.models)

    def to_dict(self):
        return self.serialize().dict()

    def serialize(self) -> MultitumorSchema:
        ...

    @classmethod
    def from_serialized(cls, data: dict) -> Self:
        try:
            version = data["version"]["string"]
        except KeyError:
            raise ValueError("Invalid JSON format")

        if version == Multitumor330.version_str:
            return Multitumor330Schema.parse_obj(data).deserialize()
        else:
            raise ValueError("Unknown BMDS version")

    def _serialize_version(self) -> VersionSchema:
        return VersionSchema(
            string=self.version_str,
            pretty=self.version_pretty,
            numeric=self.version_tuple,
            python=__version__,
            dll=bmdscore.version(),
        )

    def _serialize_settings(self) -> MultitumorSettings:
        return MultitumorSettings(
            degrees=self.degrees,
            bmr=self.settings.bmr,
            bmr_type=self.settings.bmr_type,
            alpha=self.settings.alpha,
        )

    def to_df(self, extras: dict | None = None) -> pd.DataFrame:
        if extras is None:
            extras = {}
        results = self.results
        data = []

        # model average
        ma = extras.copy()
        ma.update(
            dataset_index="-",
            dataset_id="-",
            dataset_name="-",
            dataset_dose_name="-",
            dataset_dose_units="-",
            dataset_response_name="-",
            dataset_response_units="-",
            dataset_doses="-",
            dataset_ns="-",
            dataset_incidences="-",
            model_index=-1,
            model_name="Model average",
            slope_factor=results.slope_factor,
            selected="N/A",
            bmdl=results.bmdl,
            bmd=results.bmd,
            bmdu=results.bmdu,
            aic="-",
            loglikelihood="-",
            p_value="-",
            overall_dof="-",
            bic_equiv="-",
            chi_squared="-",
            residual_of_interest="-",
            residual_at_lowest_dose="-",
        )
        data.append(ma)

        # add models
        for dataset_i, models in enumerate(results.models):
            dataset = self.datasets[dataset_i]
            extras.update(dataset_index=dataset_i)
            dataset.update_record(extras)
            # individual model rows
            for model_i, model in enumerate(models):
                extras.update(
                    model_index=model_i,
                    model_name=_model_name(model),
                    slope_factor="-",
                    selected=results.selected_model_indexes[dataset_i] == model_i,
                )
                model.update_record(extras)
                data.append(extras.copy())

        return pd.DataFrame(data=data)

    def params_df(self, extras: dict | None) -> pd.DataFrame:
        """Returns a pd.DataFrame of all parameters for all models executed.

        Args:
            extras (dict | None): extra columns to prepend
        """
        data = []
        extras = extras or {}
        for dataset_index, dataset_models in enumerate(self.results.models):
            dataset = self.datasets[dataset_index]
            for model_index, model_result in enumerate(dataset_models):
                data.extend(
                    model_result.parameters.rows(
                        extras={
                            **extras,
                            "dataset_id": dataset.metadata.id,
                            "dataset_name": dataset.metadata.name,
                            "model_index": model_index,
                            "model_name": _model_name(model_result),
                        }
                    )
                )
        return pd.DataFrame(data)

    def datasets_df(self, extras: dict | None = None) -> pd.DataFrame:
        """Returns a pd.DataFrame of all datasets within a session.

        Args:
            extras (dict | None): extra columns to prepend
        """

        data = []
        for dataset in self.datasets:
            data.extend(dataset.rows(extras))
        return pd.DataFrame(data)

    def to_docx(
        self,
        report: Report | None = None,
        header_level: int = 1,
        dataset_format_long: bool = True,
        all_models: bool = False,
        bmd_cdf_table: bool = False,
        session_inputs_table: bool = False,
    ):
        """Return a Document object with the session executed

        Args:
            report (Report, optional): A Report dataclass, or None to use default.
            header_level (int, optional): Starting header level. Defaults to 1.
            citation (bool, default True): Include citation
            dataset_format_long (bool, default True): long or wide dataset table format
            all_models (bool, default False):  Show all models, not just selected
            bmd_cdf_table (bool, default False): Export BMD CDF table
            session_inputs_table (bool, default False): Write an inputs table for a session,
                assuming a single model's input settings are representative of all models in a
                session, which may not always be true

        Returns:
            A python docx.Document object with content added.
        """
        if report is None:
            report = Report.build_default()

        h1 = report.styles.get_header_style(header_level)
        h2 = report.styles.get_header_style(header_level + 1)
        report.document.add_paragraph("Session Results", h1)
        for dataset in self.datasets:
            report.document.add_paragraph("Input Dataset", h2)
            reporting.write_dataset_table(report, dataset, dataset_format_long)

        report.document.add_paragraph("Input Settings", h2)
        write_inputs_table(report, self)

        report.document.add_paragraph("Frequentist Summary", h2)
        write_frequentist_table(report, self)
        # if all_models:
        report.document.add_paragraph("Individual Model Results", h2)
        write_models(report, self, bmd_cdf_table, header_level + 2)
        # else:
        #     report.document.add_paragraph("Selected Model", h2)
        #     if self.selected.model:
        #         reporting.write_model(
        #             report, self.selected.model, bmd_cdf_table, header_level + 2
        #         )
        #     else:
        #         report.document.add_paragraph("No model was selected as a best-fitting model.")

        # if citation:
        # reporting.write_citation(report, self.datasets[0], header_level + 1)

        return report.document


class Multitumor330(MultitumorBase):
    version_str = Version.BMDS330.value  # TODO change
    version_pretty = "3.3.0"
    version_tuple = (3, 3, 0)

    def serialize(self) -> MultitumorSchema:
        return Multitumor330Schema(
            version=self._serialize_version(),
            datasets=[ds.serialize() for ds in self.datasets],
            id=self.id,
            settings=self._serialize_settings(),
            results=self.results,
        )


class Multitumor330Schema(MultitumorSchema):
    def deserialize(self) -> Multitumor330:
        datasets = [ds.deserialize() for ds in self.datasets]
        settings = dict(
            bmr=self.settings.bmr,
            bmr_type=self.settings.bmr_type,
            alpha=self.settings.alpha,
        )
        return Multitumor330(
            datasets=datasets,
            degrees=self.settings.degrees,
            model_settings=settings,
            id=self.id,
            results=self.results,
        )


class Multitumor(Multitumor330):
    """Alias for the latest version."""


class MultistageCancer(Multistage):
    def get_model_settings(
        self, dataset: DichotomousDataset, settings: InputModelSettings
    ) -> DichotomousModelSettings:
        override_default_prior = settings is None or (
            isinstance(settings, dict) and "priors" not in settings
        )
        model_settings = super().get_model_settings(dataset, settings)
        if override_default_prior:
            model_settings.priors = self.custom_prior()
        return model_settings

    def custom_prior(self) -> ModelPriors:
        return multistage_cancer_prior()
