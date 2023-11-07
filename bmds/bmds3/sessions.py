from __future__ import annotations

import logging
from copy import copy, deepcopy
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
import pandas as pd
from simple_settings import settings

from .. import constants
from ..datasets import DatasetSchemaBase, DatasetType
from ..reporting.styling import Report
from ..utils import citation
from ..version import __version__
from . import reporting
from .constants import PriorClass
from .models import continuous as c3
from .models import dichotomous as d3
from .models import ma
from .models.base import BmdModel, BmdModelAveraging, BmdModelAveragingSchema, BmdModelSchema
from .recommender import Recommender, RecommenderSettings
from .selected import SelectedModel
from .types import sessions as schema
from .types.structs import get_version

logger = logging.getLogger(__name__)


class BmdsSession:
    """A BmdsSession is bmd modeling session for a single dataset.

    The session contains the dataset, model configuration and results, and model recommendations
    and potentially model averaging results too. BmdsSessions are a primary data type that
    should be able to be serialized and deserialized.
    """

    version_str: str
    version_pretty: str
    version_tuple: tuple[int, ...]
    model_options: dict[str, dict]

    def __init__(
        self,
        dataset: DatasetType,
        recommendation_settings: RecommenderSettings | None = None,
    ):
        self.dataset = dataset
        self.models: list[BmdModel] = []
        self.ma_weights: npt.NDArray | None = None
        self.model_average: BmdModelAveraging | None = None
        self.recommendation_settings: RecommenderSettings | None = recommendation_settings
        self.recommender: Recommender | None = None
        self.selected: SelectedModel = SelectedModel(self)

    def add_default_bayesian_models(
        self, global_settings: dict | None = None, model_average: bool = True
    ):
        global_settings = deepcopy(global_settings) if global_settings else {}
        global_settings["priors"] = PriorClass.bayesian
        for name in self.model_options[self.dataset.dtype].keys():
            model_settings = deepcopy(global_settings)
            if name in constants.VARIABLE_POLYNOMIAL:
                model_settings.update(degree=2)
            self.add_model(name, settings=model_settings)

        if model_average and self.dataset.dtype is constants.Dtype.DICHOTOMOUS:
            self.add_model_averaging()

    def add_default_models(self, global_settings=None):
        for name in self.model_options[self.dataset.dtype].keys():
            model_settings = deepcopy(global_settings) if global_settings is not None else None
            if name in constants.VARIABLE_POLYNOMIAL:
                min_poly_order = 1 if name == constants.M_MultistageCancer else 2
                max_poly_order = min(
                    self.dataset.num_dose_groups - 1, settings.MAXIMUM_POLYNOMIAL_ORDER + 1
                )
                for i in range(min_poly_order, max_poly_order):
                    poly_model_settings = (
                        deepcopy(model_settings) if model_settings is not None else {}
                    )
                    poly_model_settings["degree"] = i
                    self.add_model(name, settings=poly_model_settings)
            else:
                self.add_model(name, settings=model_settings)

    def add_model(self, name, settings=None):
        Model = self.model_options[self.dataset.dtype][name]
        instance = Model(dataset=self.dataset, settings=settings)
        self.models.append(instance)

    def set_ma_weights(self, weights: npt.ArrayLike | None = None):
        if weights is None:
            weights = np.full(len(self.models), 1 / len(self.models), dtype=np.float64)
        if len(self.models) != len(weights):
            raise ValueError(f"# model weights ({weights}) != num models {len(self.models)}")
        weights = np.array(weights)
        self.ma_weights = weights / weights.sum()

    def add_model_averaging(self, weights: list[float] | None = None):
        """
        Must be added average other models are added since a shallow copy is taken, and the
        execution of model averaging assumes all other models were executed.
        """
        if weights or self.ma_weights is None:
            self.set_ma_weights(weights)
        instance = ma.BmdModelAveragingDichotomous(session=self, models=copy(self.models))
        self.model_average = instance

    def execute(self):
        # execute individual models
        for model in self.models:
            model.execute_job()

        # execute model average
        if self.model_average is not None:
            self.model_average.execute_job()

    @property
    def recommendation_enabled(self):
        if self.recommender is None:
            self.recommender = Recommender(settings=self.recommendation_settings)
        return self.recommender.settings.enabled

    def recommend(self):
        if self.recommendation_enabled:
            self.recommender.recommend(self.dataset, self.models)
        else:
            raise ValueError("Recommendation not enabled.")

    def select(self, model: BmdModel | None, notes: str = ""):
        self.selected.select(model, notes)

    @property
    def has_recommended_model(self) -> bool:
        return (
            self.recommendation_enabled
            and self.recommender.results.recommended_model_index is not None
        )

    def accept_recommendation(self):
        """Select the recommended model, if one exists."""
        if self.has_recommended_model:
            index = self.recommender.results.recommended_model_index
            self.select(self.models[index], "Selected as best-fitting model")
        else:
            self.select(None, "No model was selected as a best-fitting model")

    def execute_and_recommend(self):
        self.execute()
        self.recommend()

    def is_bayesian(self) -> bool:
        """Determine if models are using a bayesian or frequentist approach."""
        return self.models[0].settings.priors.is_bayesian

    def citation(self) -> dict:
        return citation(self.dll_version())

    # serializing
    # -----------
    def serialize(self) -> schema.SessionSchemaBase:
        ...

    @classmethod
    def dll_version(cls) -> str:
        model = cls.model_options[constants.DICHOTOMOUS][constants.M_Logistic]
        dll = model.get_dll()
        return get_version(dll)

    @classmethod
    def from_serialized(cls, data: dict) -> BmdsSession:
        try:
            version = data["version"]["numeric"]
            dtype = data["dataset"]["dtype"]
        except KeyError:
            raise ValueError("Invalid JSON format")

        dataset = DatasetSchemaBase.get_subclass(dtype).model_validate(data["dataset"])
        model_base_class = BmdModelSchema.get_subclass(dtype)
        data["dataset"] = dataset
        data["models"] = [model_base_class.model_validate(model_) for model_ in data["models"]]
        ma = data.get("model_average")
        if ma:
            data["model_average"] = BmdModelAveragingSchema.get_subclass(dtype).model_validate(ma)
        if tuple(version) == Bmds330.version_tuple:
            return Bmds330Schema.model_validate(data).deserialize()
        else:
            raise ValueError("Unknown BMDS version")

    # reporting
    # ---------
    def to_dict(self):
        return self.serialize().model_dump(by_alias=True)

    def to_df(self, extras: dict | None = None) -> pd.DataFrame:
        """Export an executed session to a pandas dataframe

        Args:
            extras (dict, optional): _description_. Defaults to None.

        Returns:
            pd.DataFrame: A pandas dataframe
        """

        dataset_dict = {}
        self.dataset.update_record(dataset_dict)
        extras = extras or {}

        # add a row for each model
        models = []
        for bmds_model_index, model in enumerate(self.models):
            d: dict[str, Any] = {
                **extras,
                **dataset_dict,
                **dict(bmds_model_index=bmds_model_index, model_name=model.name()),
            }
            model.settings.update_record(d)
            model.results.update_record(d)

            if self.recommendation_enabled and self.recommender.results is not None:
                self.recommender.results.update_record(d, bmds_model_index)
                self.selected.update_record(d, bmds_model_index)

            if self.model_average:
                self.model_average.results.update_record_weights(d, bmds_model_index)

            models.append(d)

        # add model average row
        if self.model_average:
            d = dict(
                **extras,
                **dataset_dict,
                bmds_model_index=100,
                model_name="Model average",
            )
            self.model_average.settings.update_record(d)
            self.model_average.results.update_record(d)
            models.append(d)

        return pd.DataFrame(models)

    def to_docx(
        self,
        report: Report = None,
        header_level: int = 1,
        citation: bool = True,
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

        # remove empty first paragraph, if one exists
        if len(report.document.paragraphs) > 0:
            p = report.document.paragraphs[0]
            if not p.text and not p.runs:
                el = p._element
                el.getparent().remove(el)
                p._p = p._element = None

        h1 = report.styles.get_header_style(header_level)
        h2 = report.styles.get_header_style(header_level + 1)
        report.document.add_paragraph("Session Results", h1)
        report.document.add_paragraph("Input Dataset", h2)
        reporting.write_dataset_table(report, self.dataset, dataset_format_long)

        if session_inputs_table:
            report.document.add_paragraph("Input Settings", h2)
            reporting.write_inputs_table(report, self)

        if self.is_bayesian():
            report.document.add_paragraph("Bayesian Summary", h2)
            reporting.write_bayesian_table(report, self)
            if self.model_average:
                reporting.plot_bma(report, self)
            if all_models:
                report.document.add_paragraph("Individual Model Results", h2)
                reporting.write_models(report, self, bmd_cdf_table, header_level + 2)

        else:
            report.document.add_paragraph("Frequentist Summary", h2)
            reporting.write_frequentist_table(report, self)
            if all_models:
                report.document.add_paragraph("Individual Model Results", h2)
                reporting.write_models(report, self, bmd_cdf_table, header_level + 2)
            else:
                report.document.add_paragraph("Selected Model", h2)
                if self.selected.model:
                    reporting.write_model(
                        report, self.selected.model, bmd_cdf_table, header_level + 2
                    )
                else:
                    report.document.add_paragraph("No model was selected as a best-fitting model.")

        if citation:
            reporting.write_citation(report, self, header_level + 1)

        return report.document


class Bmds330(BmdsSession):
    version_str = constants.BMDS330
    version_pretty = "3.3.0"
    version_tuple = (3, 3, 0)
    model_options: ClassVar = {
        constants.DICHOTOMOUS: {
            constants.M_Logistic: d3.Logistic,
            constants.M_LogLogistic: d3.LogLogistic,
            constants.M_Probit: d3.Probit,
            constants.M_LogProbit: d3.LogProbit,
            constants.M_QuantalLinear: d3.QuantalLinear,
            constants.M_Multistage: d3.Multistage,
            constants.M_Gamma: d3.Gamma,
            constants.M_Weibull: d3.Weibull,
            constants.M_DichotomousHill: d3.DichotomousHill,
        },
        constants.DICHOTOMOUS_CANCER: {
            # constants.M_MultistageCancer: d3.Multistage
        },
        constants.CONTINUOUS: {
            constants.M_Linear: c3.Linear,
            constants.M_Polynomial: c3.Polynomial,
            constants.M_Power: c3.Power,
            constants.M_Hill: c3.Hill,
            constants.M_ExponentialM3: c3.ExponentialM3,
            constants.M_ExponentialM5: c3.ExponentialM5,
        },
        constants.CONTINUOUS_INDIVIDUAL: {
            constants.M_Linear: c3.Linear,
            constants.M_Polynomial: c3.Polynomial,
            constants.M_Power: c3.Power,
            constants.M_Hill: c3.Hill,
            constants.M_ExponentialM3: c3.ExponentialM3,
            constants.M_ExponentialM5: c3.ExponentialM5,
        },
    }

    def serialize(self) -> Bmds330Schema:
        schema = Bmds330Schema(
            version=dict(
                string=self.version_str,
                pretty=self.version_pretty,
                numeric=self.version_tuple,
                python=__version__,
                dll=self.dll_version(),
            ),
            dataset=self.dataset.serialize(),
            models=[model.serialize() for model in self.models],
            selected=self.selected.serialize(),
        )
        if self.model_average is not None:
            schema.bmds_model_average = self.model_average.serialize(self)

        if self.recommender is not None:
            schema.recommender = self.recommender.serialize()

        return schema


class Bmds330Schema(schema.SessionSchemaBase):
    def deserialize(self) -> Bmds330:
        session = Bmds330(dataset=self.dataset.deserialize())
        session.models = [model.deserialize(session.dataset) for model in self.models]
        session.selected = self.selected.deserialize(session)
        if self.bmds_model_average is not None:
            session.model_average = self.bmds_model_average.deserialize(session)
        if self.recommender is not None:
            session.recommendation_settings = self.recommender.settings
            session.recommender = self.recommender.deserialize()
        return session


_bmds_session_versions = {
    constants.BMDS330: Bmds330,
}


def get_model(bmds_version: str, dataset_type: str, model_name: str) -> BmdModel:
    """Get BmdModel class given metadata

    Args:
        bmds_version (str): version
        dataset_type (str): dataset type
        model_name (str): model name

    Returns:
        BmdModel: A BmdModel class
    """
    try:
        return _bmds_session_versions[bmds_version].model_options[dataset_type][model_name]
    except KeyError:
        raise ValueError(f"Model not found: {bmds_version}-{dataset_type}-{model_name}")
