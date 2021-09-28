from typing import List

from ... import constants
from ...bmds2.models import continuous as continuous2
from ...bmds2.models import dichotomous as dichotomous2
from ...bmds3.models import continuous as continuous3
from ...bmds3.models import dichotomous as dichotomous3
from ..models import ModelResult
from ...bmds3.constants import BMDS_BLANK_VALUE
from .shared import nan_to_default


def build_jobs(datasets, version: constants.Version, dtype: constants.Dtype) -> List:
    check = (version, dtype)
    if check == (constants.Version.BMDS270, constants.Dtype.CONTINUOUS):
        models = [continuous2.Power_219]
    elif check == (constants.Version.BMDS270, constants.Dtype.DICHOTOMOUS):
        models = [dichotomous2.Logistic_215]
    elif check == (constants.Version.BMDS330, constants.Dtype.CONTINUOUS):
        models = [continuous3.Power]
    elif check == (constants.Version.BMDS330, constants.Dtype.DICHOTOMOUS):
        models = [dichotomous3.Logistic]
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
            inputs=model.settings.dict(),
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
        inputs=model.settings.dict(),
        outputs=model.results.dict(),
        bmd=model.results.bmd,
        bmdl=model.results.bmdl,
        bmdu=model.results.bmdu,
        aic=model.results.fit.aic,
    )


executor = {
    constants.Version.BMDS270: _execute_fit_270,
    constants.Version.BMDS330: _execute_fit_330,
}
