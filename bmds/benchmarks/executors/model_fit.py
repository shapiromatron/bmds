from ... import constants
from ...bmds2.models import continuous as c2
from ...bmds2.models import dichotomous as d2
from ...bmds3.constants import BMDS_BLANK_VALUE
from ...bmds3.models import continuous as c3
from ...bmds3.models import dichotomous as d3
from ..models import ModelResult
from .shared import nan_to_default


def build_jobs(datasets, version: constants.Version, dtype: constants.Dtype) -> list:
    check = (version, dtype)
    if check == (constants.Version.BMDS270, constants.Dtype.CONTINUOUS):
        models = (
            c2.Power_219,
            c2.Hill_218,
            c2.Polynomial_221,
            c2.Exponential_M3_111,
            c2.Exponential_M5_111,
        )
    elif check == (constants.Version.BMDS270, constants.Dtype.DICHOTOMOUS):
        models = (
            d2.DichotomousHill_13,
            d2.Gamma_217,
            d2.Logistic_215,
            d2.LogLogistic_215,
            d2.LogProbit_34,
            d2.Probit_34,
            d2.Weibull_217,
        )
    elif check == (constants.Version.BMDS330, constants.Dtype.CONTINUOUS):
        models = (
            c3.Power,
            c3.Hill,
            c3.Polynomial,
            c3.ExponentialM3,
            c3.ExponentialM5,
        )
    elif check == (constants.Version.BMDS330, constants.Dtype.DICHOTOMOUS):
        models = (
            d3.DichotomousHill,
            # d3.Gamma,
            d3.Logistic,
            d3.LogLogistic,
            d3.LogProbit,
            d3.Probit,
            d3.Weibull,
        )
    else:
        raise ValueError(f"Unknown state: {check}")

    jobs = []
    for dataset in datasets:
        for model in models:
            jobs.append(model(dataset))

    return jobs


def _execute_fit_270(model) -> ModelResult:
    model.execute()
    if not model.has_successfully_executed:
        return ModelResult(
            bmds_version=model.bmds_version_dir,
            model=model.name,
            dataset_id=model.dataset.metadata.id,
            completed=False,
            inputs=model.get_default(),
            outputs=None,
            bmd=BMDS_BLANK_VALUE,
            bmdl=BMDS_BLANK_VALUE,
            bmdu=BMDS_BLANK_VALUE,
            aic=BMDS_BLANK_VALUE,
        )

    return ModelResult(
        bmds_version=model.bmds_version_dir,
        model=model.name,
        dataset_id=model.dataset.metadata.id,
        completed=True,
        inputs=model.get_default(),
        outputs=model.output,
        bmd=nan_to_default(model.output["BMD"]),
        bmdl=nan_to_default(model.output["BMDL"]),
        bmdu=nan_to_default(model.output["BMDU"]),
        aic=nan_to_default(model.output["AIC"]),
    )


def _execute_fit_330(model) -> ModelResult:
    model.execute()
    if not model.has_results:
        return ModelResult(
            bmds_version=model.model_version,
            model=model.name(),
            dataset_id=model.dataset.metadata.id,
            completed=False,
            inputs=model.settings.model_dump(),
            outputs=None,
            bmd=BMDS_BLANK_VALUE,
            bmdl=BMDS_BLANK_VALUE,
            bmdu=BMDS_BLANK_VALUE,
            aic=BMDS_BLANK_VALUE,
        )

    return ModelResult(
        bmds_version=model.model_version,
        model=model.name(),
        dataset_id=model.dataset.metadata.id,
        completed=True,
        inputs=model.settings.model_dump(),
        outputs=model.results.model_dump(),
        bmd=model.results.bmd,
        bmdl=model.results.bmdl,
        bmdu=model.results.bmdu,
        aic=model.results.fit.aic,
    )


executor = {
    constants.Version.BMDS270: _execute_fit_270,
    constants.Version.BMDS330: _execute_fit_330,
}
