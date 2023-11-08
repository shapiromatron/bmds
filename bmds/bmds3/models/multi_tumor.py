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


def write_docx_frequentist_table(report: Report, session):
    """Add frequentist table to document."""
    styles = report.styles
    hdr = report.styles.tbl_header
    body = report.styles.tbl_body

    avg_row = len(session.datasets) > 1

    footnotes = TableFootnote()
    tbl = report.document.add_table(
        len(session.models) + 1 + (1 if avg_row else 0), 9, style=styles.table
    )

    write_cell(tbl.cell(0, 0), "Model", style=hdr)
    write_cell(tbl.cell(0, 1), "BMDL", style=hdr)
    write_cell(tbl.cell(0, 2), "BMD", style=hdr)
    write_cell(tbl.cell(0, 3), "BMDU", style=hdr)
    write_pvalue_header(tbl.cell(0, 4), style=hdr)
    write_cell(tbl.cell(0, 5), "AIC", style=hdr)
    write_cell(tbl.cell(0, 6), "Scaled Residual for Dose Group near BMD", style=hdr)
    write_cell(tbl.cell(0, 7), "Scaled Residual for Control Dose Group", style=hdr)
    write_cell(tbl.cell(0, 8), "Recommendation and Notes", style=hdr)

    if avg_row:
        write_cell(tbl.cell(1, 0), "Average", body)
        write_cell(tbl.cell(1, 1), session.results.bmdl, body)
        write_cell(tbl.cell(1, 2), session.results.bmd, body)
        write_cell(tbl.cell(1, 3), session.results.bmdu, body)
        write_cell(tbl.cell(1, 4), "-", body)
        write_cell(tbl.cell(1, 5), "-", body)
        write_cell(tbl.cell(1, 6), "-", body)
        write_cell(tbl.cell(1, 7), "-", body)
        write_cell(tbl.cell(1, 8), "-", body)

    for ds_idx, model in enumerate(session.models):
        row = ds_idx + 1 + (1 if avg_row else 0)
        idx = session.results.selected_model_indexes[ds_idx]
        write_cell(tbl.cell(row, 0), model[idx].name(), body)
        write_cell(tbl.cell(row, 1), model[idx].results.bmdl, body)
        write_cell(tbl.cell(row, 2), model[idx].results.bmd, body)
        write_cell(tbl.cell(row, 3), model[idx].results.bmdu, body)
        write_cell(tbl.cell(row, 4), model[idx].get_gof_pvalue(), body)
        write_cell(tbl.cell(row, 5), model[idx].results.fit.aic, body)
        write_cell(tbl.cell(row, 6), model[idx].results.gof.roi, body)
        write_cell(tbl.cell(row, 7), model[idx].results.gof.residual[0], body)
        write_cell(tbl.cell(row, 8), "-", body)

    # set column width
    widths = np.array([1.75, 0.8, 0.8, 0.7, 0.7, 0.7, 0.7, 0.7, 1.75])
    widths = widths / (widths.sum() / styles.portrait_width)
    for width, col in zip(widths, tbl.columns, strict=True):
        set_column_width(col, width)

    # write footnote
    if len(footnotes) > 0:
        footnotes.add_footnote_text(report.document, styles.tbl_footnote)


def write_docx_inputs_table(report: Report, session):
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


def write_docx_model(report: Report, model, bmd_cdf_table: bool, header_level: int):
    styles = report.styles
    header_style = styles.get_header_style(header_level)
    report.document.add_paragraph(model.name(), header_style)
    if model.has_results:
        report.document.add_paragraph(add_mpl_figure(report.document, model.plot(), 6))
        # if bmd_cdf_table: # TODO - change - add?
        #     report.document.add_paragraph(add_mpl_figure(report.document, model.cdf_plot(), 6))
        report.document.add_paragraph(model.text(), styles.fixed_width)


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
            return DichotomousModelSettings.model_validate(settings)

    def _build_model_settings(self) -> list[list[DichotomousModelSettings]]:
        # Build individual model settings based from inputs
        settings = []
        for i, dataset in enumerate(self.datasets):
            ds_settings = []
            degree_i = self.degrees[i]
            degrees_i = (
                range(degree_i, degree_i + 1) if degree_i > 0 else range(1, dataset.num_dose_groups)
            )
            for degree in degrees_i:
                model_settings = self.settings.model_copy(
                    update=dict(degree=degree, priors=multistage_cancer_prior())
                )
                ds_settings.append(model_settings)
            settings.append(ds_settings)
        return settings

    def to_cpp(self) -> MultitumorAnalysis:
        all_settings = self._build_model_settings()
        dataset_models = []
        dataset_results = []
        ns = []
        for dataset, dataset_settings in zip(self.datasets, all_settings, strict=True):
            mc_models = []
            self.models.append(mc_models)
            models = []
            results = []
            ns.append(dataset.num_dose_groups)
            for settings in dataset_settings:
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
        return self.serialize().model_dump(by_alias=True)

    def serialize(self) -> MultitumorSchema:
        ...

    @classmethod
    def from_serialized(cls, data: dict) -> Self:
        try:
            version = data["version"]["string"]
        except KeyError:
            raise ValueError("Invalid JSON format")

        if version == Multitumor330.version_str:
            return Multitumor330Schema.model_validate(data).deserialize()
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

    def to_df(self, extras: dict | None = None, clean: bool = True) -> pd.DataFrame:
        """Export an executed session to a pandas dataframe.

        Args:
            extras (dict, optional): Extra items to add to row.
            clean (bool, default True): Remove empty columns.

        Returns:
            pd.DataFrame: A pandas dataframe
        """
        if extras is None:
            extras = {}
        results = self.results
        data = []

        # model average
        ma = extras.copy()
        ma.update(
            dataset_index=np.nan,
            dataset_id=np.nan,
            dataset_name=np.nan,
            dataset_dose_name=np.nan,
            dataset_dose_units=np.nan,
            dataset_response_name=np.nan,
            dataset_response_units=np.nan,
            dataset_doses=np.nan,
            dataset_ns=np.nan,
            dataset_incidences=np.nan,
            model_index=np.nan,
            model_name="Model average",
            slope_factor=results.slope_factor,
            selected=np.nan,
            bmdl=results.bmdl,
            bmd=results.bmd,
            bmdu=results.bmdu,
            aic=np.nan,
            loglikelihood=np.nan,
            p_value=np.nan,
            overall_dof=np.nan,
            bic_equiv=np.nan,
            chi_squared=np.nan,
            residual_of_interest=np.nan,
            residual_at_lowest_dose=np.nan,
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
                    selected=results.selected_model_indexes[dataset_i] == model_i,
                )
                model.update_record(extras)
                data.append(extras.copy())

        df = pd.DataFrame(data=data)
        if clean:
            df = df.dropna(axis=1, how="all").fillna("")
        return df

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
        citation: bool = False,
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
            A python docx.Document object with content added, session_inputs_table
        """
        # TODO - change - implement bmd_cdf_table, all_models, etc?
        if report is None:
            report = Report.build_default()

        h1 = report.styles.get_header_style(header_level)
        h2 = report.styles.get_header_style(header_level + 1)
        report.document.add_paragraph("Session Results", h1)
        for dataset in self.datasets:
            report.document.add_paragraph("Input Dataset", h2)
            reporting.write_dataset_table(report, dataset, dataset_format_long)

        report.document.add_paragraph("Input Settings", h2)
        write_docx_inputs_table(report, self)

        report.document.add_paragraph("Frequentist Summary", h2)
        write_docx_frequentist_table(report, self)
        report.document.add_paragraph("Individual Model Results", h2)

        for dataset_models in self.models:
            for model in dataset_models:
                write_docx_model(report, model, bmd_cdf_table, header_level)

        if citation:
            report.document.add_paragraph("# TODO - change", h2)

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
        mt = Multitumor330(
            datasets=datasets,
            degrees=self.settings.degrees,
            model_settings=settings,
            id=self.id,
            results=self.results,
        )
        # hydrate models
        for dataset, ds_settings, ds_results in zip(
            mt.datasets, mt.results.settings, mt.results.models, strict=True
        ):
            models = []
            for settings, results in zip(ds_settings, ds_results, strict=True):
                model = MultistageCancer(dataset=dataset, settings=settings)
                model.results = results
                models.append(model)
            mt.models.append(models)
        return mt


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
