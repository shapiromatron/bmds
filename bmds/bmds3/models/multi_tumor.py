import pandas as pd
from pydantic import BaseModel

from ... import bmdscore
from ...datasets.dichotomous import DichotomousDataset, DichotomousDatasetSchema
from ...reporting.styling import Report
from ..constants import N_BMD_DIST, NUM_PRIOR_COLS, PriorClass, PriorType
from ..types.dichotomous import DichotomousModelSettings
from ..types.multi_tumor import MultitumorAnalysis, MultitumorConfig, MultitumorResult
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


class MultitumorSchema(BaseModel):
    version: VersionSchema
    id: int | str | None
    datasets: list[DichotomousDatasetSchema]
    settings: MultitumorConfig
    results: MultitumorResult | None


class Multitumor:
    def __init__(
        self,
        datasets: list[DichotomousDataset],
        degrees: list[int] | None = None,
        model_settings: DichotomousModelSettings | dict | None = None,
        id: int | str | None = None,
    ):
        if len(datasets) == 0:
            raise ValueError("Must provide at least one dataset")
        self.id = id
        self.datasets = datasets
        self.degrees: list[int] = degrees or [0] * len(datasets)
        self.settings: DichotomousModelSettings = self.get_base_settings(model_settings)
        self.structs: tuple | None = None
        self.results: MultitumorResult | None = None

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
            models = []
            results = []
            ns.append(dataset.num_dose_groups)
            degree_i = self.degrees[i]
            degrees_i = (
                range(degree_i, degree_i + 1) if degree_i > 0 else range(2, dataset.num_dose_groups)
            )
            for degree in degrees_i:
                # build inputs
                settings = self.settings.copy(
                    update={
                        "degree": degree,
                        "priors": multistage_cancer_prior(),
                        "samples": 0,
                        "burnin": 0,
                    }
                )
                model = MultistageCancer(dataset, settings=settings)
                inputs = model._build_inputs()
                analysis = inputs.to_cpp_analysis()
                models.append(analysis)

                # build outputs
                res = bmdscore.python_dichotomous_model_result()
                res.model = bmdscore.dich_model.d_multistage
                res.nparms = degree + 1
                res.dist_numE = N_BMD_DIST * 2
                res.gof = bmdscore.dichotomous_GOF()
                res.bmdsRes = bmdscore.BMDS_results()
                res.aod = bmdscore.dicho_AOD()
                results.append(res)

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
        return self.results.text()

    def serialize(self) -> MultitumorSchema:
        ...

    def to_dict(self):
        return self.serialize().dict()

    def deserialize(self):
        pass

    def to_df(self, extras: dict | None = None) -> pd.DataFrame:
        return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    def to_docx(
        self,
        report: Report | None = None,
        header_level: int = 1,
        citation: bool = True,
    ):
        if report is None:
            report = Report.build_default()
        report.document.add_heading("Multitumor Analysis", level=header_level)
        if citation:
            report.document.add_heading("todo...", level=header_level + 1)
        return report.document


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
