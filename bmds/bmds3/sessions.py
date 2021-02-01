import logging
from copy import copy, deepcopy
from typing import Dict, List, Optional, Tuple

import pandas as pd
from simple_settings import settings

from .. import constants
from ..datasets import DatasetSchemaBase, DatasetType
from ..reporting.styling import Report
from . import reporting
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
        self.model_average: Optional[BmdModelAveraging] = None
        self.recommendation_settings: Optional[RecommenderSettings] = recommendation_settings
        self.recommender: Optional[Recommender] = None
        self.selected: SelectedModel = SelectedModel(self)

    def add_default_models(self, global_settings=None):
        for name in self.model_options[self.dataset.dtype].keys():
            model_settings = deepcopy(global_settings) if global_settings is not None else None

            # TODO - change this; use `degree` in settings
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

    def add_model_averaging(self):
        """
        Must be added average other models are added since a shallow copy is taken, and the
        execution of model averaging assumes all other models were executed.
        """
        instance = ma.BmdModelAveragingDichotomous(dataset=self.dataset, models=copy((self.models)))
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

    def execute_and_recommend(self, drop_doses=False):
        self.execute()
        self.recommend()
        if drop_doses:
            raise NotImplementedError("TODO")

    # serializing
    # -----------
    def serialize(self) -> schema.SessionSchemaBase:
        ...

    @classmethod
    def from_serialized(cls, data: Dict) -> "BmdsSession":
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

    def to_df(self, dropna: bool = True) -> pd.DataFrame:
        """
        Export an executed session to a pandas dataframe

        Args:
            dropna (bool, optional): Drop columns with missing data. Defaults to True.

        Returns:
            pd.DataFrame: A pandas dataframe
        """

        def list_to_str(data, default=None):
            if data is None:
                return default
            return "|".join([str(d) for d in data])

        # build dataset
        dataset_dict = dict(
            dataset_id=self.dataset.metadata.id,
            dataset_name=self.dataset.metadata.name,
            doses=list_to_str(getattr(self.dataset, "doses", None)),
            dose_name=self.dataset.metadata.dose_name,
            dose_units=self.dataset.metadata.dose_units,
            ns=list_to_str(getattr(self.dataset, "doses", None)),
            means=list_to_str(getattr(self.dataset, "means", None)),
            stdevs=list_to_str(getattr(self.dataset, "stdevs", None)),
            incidences=list_to_str(getattr(self.dataset, "incidences", None)),
            response_name=self.dataset.metadata.response_name,
            response_units=self.dataset.metadata.response_units,
        )

        # build model rows
        model_rows = []
        model_row_names = [
            "model_index",
            "model_name",
            "bmd",
            "bmdl",
            "bmdu",
            "aic",
            "params",
        ]
        for idx, model in enumerate(self.models):
            model_rows.append(
                [
                    idx,
                    model.name(),
                    model.results.bmd,
                    model.results.bmdl,
                    model.results.bmdu,
                    model.results.aic,
                    list_to_str(model.results.fit.params),
                ]
            )
        df = pd.DataFrame(data=model_rows, columns=model_row_names)

        # add dataset rows
        for key, value in dataset_dict.items():
            df[key] = value

        # reorder
        column_order = list(dataset_dict.keys()) + model_row_names
        df = df.reindex(column_order, axis=1)

        # drop empty columns
        if dropna:
            df = df.dropna(axis=1, how="all")

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

        report.document.add_paragraph("Session results", report.styles.header_1)
        reporting.write_dataset(report, self.dataset, header_level + 1)
        reporting.write_summary_table(report, self, header_level + 1)
        reporting.write_models(report, self, header_level + 1)

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
