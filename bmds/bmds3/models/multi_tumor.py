from typing import Self

import pandas as pd

from ... import bmdscore
from ...constants import Version
from ...datasets.dichotomous import DichotomousDataset
from ...reporting.styling import Report
from ...version import __version__
from ..constants import NUM_PRIOR_COLS, PriorClass, PriorType
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
        return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})  # TODO

    def params_df(self, extras: dict | None) -> pd.DataFrame:
        """Returns a pd.DataFrame of all parameters for all models executed.

        Args:
            extras (dict | None): extra columns to prepend
        """
        data = []
        extras = extras or {}
        for dataset_index, dataset_models in enumerate(self.results.models):
            dataset = self.datasets[dataset_index]
            for model_index, model_results in enumerate(dataset_models):
                degree = model_results.parameters.names[-1][-1]
                model_name = f"Multistage {degree}°"
                data.extend(
                    model_results.parameters.rows(
                        extras={
                            **extras,
                            "dataset_id": dataset.metadata.id,
                            "dataset_name": dataset.metadata.name,
                            "model_index": model_index,
                            "model_name": model_name,
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
        citation: bool = True,
    ):
        if report is None:
            report = Report.build_default()
        report.document.add_heading("Multitumor Analysis", level=header_level)
        report.document.add_paragraph("TODO")
        if citation:
            report.document.add_heading("TODO", level=header_level + 1)
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
