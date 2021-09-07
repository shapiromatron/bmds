from __future__ import annotations

import logging
from copy import copy, deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from simple_settings import settings

from .. import constants
from ..datasets import DatasetSchemaBase, DatasetType
from ..reporting.styling import Report
from . import reporting
from .constants import PriorClass
from .models import continuous as c3
from .models import dichotomous as d3
from .models import ma
from .models.base import BmdModel, BmdModelAveraging, BmdModelAveragingSchema, BmdModelSchema
from .recommender import Recommender, RecommenderSettings
from .selected import SelectedModel
from .types import sessions as schema

logger = logging.getLogger(__name__)


class BmdsSession:
    """A BmdsSession is bmd modeling session for a single dataset.

    The session contains the dataset, model configuration and results, and model recommendations
    and potentially model averaging results too. BmdsSessions are a primary data type that
    should be able to be serialized and deserialized.
    """

    version_str: str
    version_pretty: str
    version_tuple: Tuple[int, ...]
    model_options: Dict[str, Dict]

    def __init__(
        self, dataset: DatasetType, recommendation_settings: Optional[RecommenderSettings] = None,
    ):
        self.dataset = dataset
        self.models: List[BmdModel] = []
        self.ma_weights: Optional[npt.NDArray] = None
        self.model_average: Optional[BmdModelAveraging] = None
        self.recommendation_settings: Optional[RecommenderSettings] = recommendation_settings
        self.recommender: Optional[Recommender] = None
        self.selected: SelectedModel = SelectedModel(self)

    def add_default_bayesian_models(self, global_settings: Dict = None, model_average: bool = True):
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

    def set_ma_weights(self, weights: Optional[npt.ArrayLike] = None):
        if weights is None:
            weights = np.full(len(self.models), 1 / len(self.models), dtype=np.float64)
        if len(self.models) != len(weights):
            raise ValueError(f"# model weights ({weights}) != num models {len(self.models)}")
        weights = np.array(weights)
        self.ma_weights = weights / weights.sum()

    def add_model_averaging(self, weights: Optional[List[float]] = None):
        """
        Must be added average other models are added since a shallow copy is taken, and the
        execution of model averaging assumes all other models were executed.
        """
        if weights or self.ma_weights is None:
            self.set_ma_weights(weights)
        instance = ma.BmdModelAveragingDichotomous(session=self, models=copy((self.models)))
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

    def select(self, model: Optional[BmdModel], notes: str = ""):
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
        """Determine if models are using a bayesian or frequentist approach.

        Looks at the first model's prior to determine if it's bayesian, else assume frequentist.
        """
        # TODO - fix with handle PriorClass.custom
        first_class = self.models[0].settings.priors.prior_class
        return first_class is PriorClass.bayesian

    # serializing
    # -----------
    def serialize(self) -> schema.SessionSchemaBase:
        ...

    @classmethod
    def from_serialized(cls, data: Dict) -> BmdsSession:
        try:
            version = data["version"]["numeric"]
            dtype = data["dataset"]["dtype"]
        except KeyError:
            raise ValueError("Invalid JSON format")

        dataset = DatasetSchemaBase.get_subclass(dtype).parse_obj(data["dataset"])
        model_base_class = BmdModelSchema.get_subclass(dtype)
        data["dataset"] = dataset
        data["models"] = [model_base_class.parse_obj(model_) for model_ in data["models"]]
        ma = data.get("model_average")
        if ma:
            data["model_average"] = BmdModelAveragingSchema.get_subclass(dtype).parse_obj(ma)
        if tuple(version) == Bmds330.version_tuple:
            return Bmds330Schema.parse_obj(data).deserialize()
        else:
            raise ValueError("Unknown BMDS version")

    # reporting
    # ---------
    def to_dict(self):
        return self.serialize().dict()

    def to_df(self) -> pd.DataFrame:
        """
        Export an executed session to a pandas dataframe

        Returns:
            pd.DataFrame: A pandas dataframe
        """

        dataset_dict = {}
        self.dataset.update_record(dataset_dict)

        # add a row for each model
        models = []
        for model_index, model in enumerate(self.models):
            d: dict[str, Any] = dict(
                model_index=model_index, model_name=model.name(),
            )
            model.settings.update_record(d)
            model.results.update_record(d)

            if self.recommendation_enabled and self.recommender.results is not None:
                self.recommender.results.update_record(d, model_index)
                self.selected.update_record(d, model_index)

            if self.model_average:
                self.model_average.results.update_record_weights(d, model_index)

            models.append(d)

        # add model average row
        if self.model_average:
            d = dict(model_index=100, model_name="Model average",)
            self.model_average.settings.update_record(d)
            self.model_average.results.update_record(d)
            models.append(d)

        # merge dataset with other items in dataframe; reorder rows
        df = pd.DataFrame(models)
        columns = list(dataset_dict.keys())
        columns.extend(df.columns.tolist())
        df = df.assign(**dataset_dict)
        df = df[columns]

        return df

    def to_docx(
        self, report: Report = None, header_level: int = 1,
    ):
        """Return a Document object with the session executed

        Args:
            report (Report, optional): A Report dataclass, or None to use default.
            header_level (int, optional): Starting header level. Defaults to 1.

        Returns:
            A python docx.Document object with content added.
        """
        if report is None:
            report = Report.build_default()

        h1 = report.styles.get_header_style(header_level)
        h2 = report.styles.get_header_style(header_level + 1)
        report.document.add_paragraph("Session results", h1)
        report.document.add_paragraph("Input dataset", h2)
        reporting.write_dataset(report, self.dataset)

        if self.is_bayesian():
            report.document.add_paragraph("Bayesian Summary", h2)
            reporting.write_bayesian_table(report, self)
            if self.model_average:
                reporting.plot_bma(report, self)
        else:
            report.document.add_paragraph("Frequentist Summary", h2)
            reporting.write_frequentist_table(report, self)

        report.document.add_paragraph("Individual model results", h2)
        reporting.write_models(report, self, header_level + 2)

        return report.document


class Bmds330(BmdsSession):
    version_str = constants.BMDS330
    version_pretty = "BMDS v3.3.0"
    version_tuple = (3, 3, 0)
    model_options = {
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

    def serialize(self) -> "Bmds330Schema":
        schema = Bmds330Schema(
            version=dict(
                string=self.version_str, pretty=self.version_pretty, numeric=self.version_tuple,
            ),
            dataset=self.dataset.serialize(),
            models=[model.serialize() for model in self.models],
            selected=self.selected.serialize(),
        )
        if self.model_average is not None:
            schema.model_average = self.model_average.serialize(self)

        if self.recommender is not None:
            schema.recommender = self.recommender.serialize()

        return schema


class Bmds330Schema(schema.SessionSchemaBase):
    def deserialize(self) -> Bmds330:
        session = Bmds330(dataset=self.dataset.deserialize())
        session.models = [model.deserialize(session.dataset) for model in self.models]
        session.selected = self.selected.deserialize(session)
        if self.model_average is not None:
            session.model_average = self.model_average.deserialize(session)
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
