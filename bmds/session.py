import os
import logging
import platform
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
from typing import Dict, Tuple

import pandas as pd
from simple_settings import settings

from . import __version__, constants, logic, remote, models, reporter, utils
from .bmds3.models import continuous as c3
from .bmds3.models import dichotomous as d3

__all__ = ("BMDS",)

logger = logging.getLogger(__name__)


class BMDS:
    """
    A single dataset, related models, outputs, and model recommendations.
    """

    version_str: str = ""
    version_pretty: str = ""
    version_tuple: Tuple[int, ...] = ()
    model_options: Dict[str, Dict] = {}

    @utils.classproperty
    def versions(cls) -> Dict:
        """
        Return all available BMDS software versions.

        Example
        -------

        import bmds

        Returns
        -------
        Dict(str, cls) of available BMDS versions.
        """
        return _BMDS_VERSIONS

    @classmethod
    def get_model(cls, version, model_name):
        """
        Return BMDS model class given BMDS version and model-name.
        """
        models = cls.versions[version].model_options
        for keystore in models.values():
            if model_name in keystore:
                return keystore[model_name]
        raise ValueError("Unknown model name")

    @classmethod
    def get_versions(cls):
        return cls.versions.keys()

    @classmethod
    def version(cls, version: str, *args, **kwargs):
        """
        Return a BMDS session of the specified version. If additional
        arguments are provided, an instance of this class is generated.
        """
        cls = cls.versions[version]
        if len(args) > 0 or len(kwargs) > 0:
            return cls(*args, **kwargs)
        return cls

    @classmethod
    def latest_version(cls, *args, **kwargs):
        """
        Return the class of the latest version of the BMDS. If additional
        arguments are provided, an instance of this class is generated.
        """
        latest = list(cls.versions.keys())[-1]
        return cls.version(latest, *args, **kwargs)

    bmr_options = {
        constants.DICHOTOMOUS: constants.DICHOTOMOUS_BMRS,
        constants.DICHOTOMOUS_CANCER: constants.DICHOTOMOUS_BMRS,
        constants.CONTINUOUS: constants.CONTINUOUS_BMRS,
        constants.CONTINUOUS_INDIVIDUAL: constants.CONTINUOUS_BMRS,
    }

    def __init__(self, dtype, dataset=None):
        """
        Attributes:
            dtype (str): dataset type
            models (List[Models]): list of BMDS models to be included
            dataset (Dataset): A BMDS dataset, mutable if doses are dropped
            original_dataset (Dataset): The unchanged original dataset.
            doses_dropped (int): the number of doses dropped in current session
            doses_dropped_sessions (Dict[int, Session): history of prior sessions
        """
        self.dtype = dtype
        if self.dtype not in constants.DTYPES:
            raise ValueError(f"Invalid data type: {dtype}")
        self.models = []
        self.dataset = dataset
        self.original_dataset = deepcopy(self.dataset)
        self.doses_dropped = 0
        self.doses_dropped_sessions = {}

    def get_bmr_options(self):
        return self.bmr_options[self.dtype]

    def get_model_options(self):
        return [model.get_default() for model in self.model_options[self.dtype].values()]

    @property
    def has_models(self):
        return len(self.models) > 0

    def add_default_models(self, global_settings=None):
        for name in self.model_options[self.dtype].keys():
            model_settings = deepcopy(global_settings) if global_settings is not None else None
            if name in constants.VARIABLE_POLYNOMIAL:
                min_poly_order = 1 if name == constants.M_MultistageCancer else 2
                max_poly_order = min(
                    self.dataset.num_dose_groups, settings.MAXIMUM_POLYNOMIAL_ORDER + 1
                )
                for i in range(min_poly_order, max_poly_order):
                    poly_model_settings = (
                        deepcopy(model_settings) if model_settings is not None else {}
                    )
                    poly_model_settings["degree_poly"] = i
                    self.add_model(name, settings=poly_model_settings)
            else:
                self.add_model(name, settings=model_settings)

    def add_model(self, name, settings=None, id=None):
        if self.dataset is None:
            raise ValueError("Add dataset to session before adding models")
        Model = self.model_options[self.dtype][name]
        instance = Model(dataset=self.dataset, settings=settings, id=id)
        self.models.append(instance)

    def execute(self):
        if self._can_execute_locally():
            self._execute()
        else:
            self._execute_remote()

    def _execute(self):
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            promises = executor.map(lambda model: model.execute_job(), self.models)

        # evaluate response; throw Exceptions if raised
        list(promises)

    def _execute_remote(self):
        # submit data
        start_time = datetime.now()
        executable_models = []
        for model in self.models:
            model.execution_start = start_time
            if model.can_be_executed:
                executable_models.append(model)
            else:
                remote._set_results(model)

        if len(executable_models) == 0:
            return

        session = remote._get_requests_session()
        payload = remote._get_payload(executable_models)
        logger.debug(f"Submitting payload: {payload}")
        resp = session.post(session._BMDS_REQUEST_URL, data=payload)

        # parse results for each model
        jsoned = resp.json()
        for model, results in zip(executable_models, jsoned):
            remote._set_results(model, results)

    @property
    def recommendation_enabled(self):
        return hasattr(self, "recommended_model")

    def add_recommender(self):
        self.recommender = logic.Recommender(self.dtype)

    def recommend(self):
        if not hasattr(self, "recommender"):
            self.add_recommender()
        self.recommended_model = self.recommender.recommend(self.dataset, self.models)
        self.recommended_model_index = (
            self.models.index(self.recommended_model)
            if self.recommended_model is not None
            else None
        )
        return self.recommended_model

    def clone(self):
        """
        Clone Session. If there are doses dropped from prior sessions, these prior sessions
        are removed from the clone.
        """
        clone = deepcopy(self)
        clone.doses_dropped_sessions = {}
        return clone

    def execute_and_recommend(self, drop_doses=False):
        """
        Execute and recommend a best-fitting model. If drop_doses and no model
        is recommended, drop the highest dose-group and repeat until either:

        1. a model is recommended, or
        2. the dataset is exhausted (i.e., only 3 dose-groups remain).

        The session instance is equal to the final run which was executed; if doses were dropped
        all previous sessions are saved in self.doses_dropped_sessions.
        """
        self.execute()
        self.recommend()

        if not drop_doses:
            return

        while (
            self.recommended_model is None
            and self.dataset.num_dose_groups > self.dataset.MINIMUM_DOSE_GROUPS
        ):
            self.doses_dropped_sessions[self.doses_dropped] = self.clone()
            self.dataset.drop_dose()
            self.doses_dropped += 1
            self.execute()
            self.recommend()

    @staticmethod
    def _df_ordered_dict(include_io=True):
        # return an ordered defaultdict list
        keys = [
            "dataset_index",
            "doses_dropped",
            "model_name",
            "model_index",
            "model_version",
            "has_output",
            "execution_halted",
            "BMD",
            "BMDL",
            "BMDU",
            "CSF",
            "AIC",
            "pvalue1",
            "pvalue2",
            "pvalue3",
            "pvalue4",
            "Chi2",
            "df",
            "residual_of_interest",
            "warnings",
            "logic_bin",
            "logic_cautions",
            "logic_warnings",
            "logic_failures",
            "recommended",
            "recommended_variable",
        ]

        if include_io:
            keys.extend(["dfile", "outfile", "stdout", "stderr"])

        return {key: [] for key in keys}

    def _add_to_to_ordered_dict(self, d, dataset_index, recommended_only=False):
        """
        Save a session to an ordered dictionary. In some cases, a single session may include
        a final session, as well as other BMDS executions where doses were dropped. This
        will include all sessions.
        """
        if self.doses_dropped_sessions:
            for key in sorted(list(self.doses_dropped_sessions.keys())):
                session = self.doses_dropped_sessions[key]
                session._add_single_session_to_to_ordered_dict(d, dataset_index, recommended_only)
        self._add_single_session_to_to_ordered_dict(d, dataset_index, recommended_only)

    def _add_single_session_to_to_ordered_dict(self, d, dataset_index, recommended_only):
        """
        Save a single session to an ordered dictionary.
        """
        for model_index, model in enumerate(self.models):
            # determine if model should be presented, or if a null-model should
            # be presented (if no model is recommended.)
            show_null = False
            if recommended_only:
                if self.recommendation_enabled:
                    if self.recommended_model is None:
                        if model_index == 0:
                            show_null = True
                        else:
                            continue
                    elif self.recommended_model == model:
                        pass
                    else:
                        continue
                else:
                    if model_index == 0:
                        show_null = True
                    else:
                        continue

            d["dataset_index"].append(dataset_index)
            d["doses_dropped"].append(self.doses_dropped)
            model._to_df(d, model_index, show_null)

    def to_dict(self, dataset_index):
        return dict(
            bmds_version=self.version_str,
            bmds_python_version=__version__,
            dataset_index=dataset_index,
            dataset=self.dataset.to_dict(),
            models=[model.to_dict(i) for i, model in enumerate(self.models)],
            recommended_model_index=getattr(self, "recommended_model_index", None),
        )

    def to_excel(self, filename):
        d = self._df_ordered_dict()
        self._add_to_to_ordered_dict(d, 0)
        df = pd.DataFrame(d)
        filename = os.path.expanduser(filename)
        df.to_excel(filename, index=False)

    def save_plots(self, directory, prefix=None, format="png", dpi=None, recommended_only=False):

        directory = os.path.expanduser(directory)
        if not os.path.exists(directory):
            raise ValueError("Directory not found: {}".format(directory))

        for model in self.models:
            if recommended_only and (
                self.recommendation_enabled is False or model.recommended is False
            ):
                continue

            fn = "{}.{}".format(model.name, format)
            if prefix is not None:
                fn = "{}-{}".format(prefix, fn)

            fig = model.plot()
            fig.savefig(os.path.join(directory, fn), dpi=dpi)
            fig.clear()

    def to_docx(
        self,
        filename=None,
        title=None,
        input_dataset=True,
        summary_table=True,
        recommendation_details=True,
        recommended_model=True,
        all_models=False,
    ):
        """
        Write session outputs to a Word file.

        Parameters
        ----------
        filename : str or None
            If provided, the file is saved to this location, otherwise this
            method returns a docx.Document
        input_dataset : bool
            Include input dataset data table
        summary_table : bool
            Include model summary table
        recommendation_details : bool
            Include model recommendation details table
        recommended_model : bool
            Include the recommended model output and dose-response plot, if
            one exists
        all_models : bool
            Include all models output and dose-response plots

        Returns
        -------
        bmds.Reporter
            The bmds.Reporter object.

        """
        rep = reporter.Reporter()
        rep.add_session(
            self,
            input_dataset,
            summary_table,
            recommendation_details,
            recommended_model,
            all_models,
        )

        if filename:
            rep.save(filename)

        return rep

    def _group_models(self):
        """
        If AIC and BMD are numeric and identical, then treat models as
        identical. Returns a list of lists. The outer list is a list of related
        models, the inner list contains each individual model, sorted by the
        number of parameters in ascending order.

        This is required because in some cases, a higher-order model may not use
        some parameters and can effectively collapse to a lower order model
        (for example, a 2nd order polynomial and power model may collapse to a
        linear model). In summary outputs, we may want to present all models in
        one row, since they are the same model effectively.
        """
        od = {}

        # Add models to appropriate list. We only aggregate models which
        # completed successfully and have a valid AIC and BMD.
        for i, model in enumerate(self.models):
            output = getattr(model, "output", {})
            if output.get("AIC") and output.get("BMD") and output["BMD"] > 0:
                key = "{}-{}".format(output["AIC"], output["BMD"])
                if key in od:
                    od[key].append(model)
                else:
                    od[key] = [model]
            else:
                od[i] = [model]

        # Sort each list by the number of parameters
        def _get_num_params(model):
            return (
                len(model.output["parameters"])
                if hasattr(model, "output") and "parameters" in model.output
                else 0
            )

        for key, _models in od.items():
            _models.sort(key=_get_num_params)

        return list(od.values())

    def _can_execute_locally(self) -> bool:
        major_version = self.version_tuple[0]
        if major_version == 2:
            return platform.system() == "Windows"
        elif major_version == 3:
            return True


class BMDS_v231(BMDS):
    version_str = constants.BMDS231
    version_pretty = "BMDS v2.3.1"
    version_tuple = (2, 3, 1)


class BMDS_v240(BMDS_v231):
    version_str = constants.BMDS240
    version_pretty = "BMDS v2.4.0"
    version_tuple = (2, 4, 0)
    model_options = {
        constants.DICHOTOMOUS: {
            constants.M_Logistic: models.Logistic_214,
            constants.M_LogLogistic: models.LogLogistic_214,
            constants.M_Probit: models.Probit_33,
            constants.M_LogProbit: models.LogProbit_33,
            constants.M_QuantalLinear: models.QuantalLinear_33,
            constants.M_Multistage: models.Multistage_33,
            constants.M_Gamma: models.Gamma_216,
            constants.M_Weibull: models.Weibull_216,
        },
        constants.DICHOTOMOUS_CANCER: {constants.M_MultistageCancer: models.MultistageCancer_110},
        constants.CONTINUOUS: {
            constants.M_Linear: models.Linear_217,
            constants.M_Polynomial: models.Polynomial_217,
            constants.M_Power: models.Power_217,
            constants.M_Hill: models.Hill_217,
            constants.M_ExponentialM2: models.Exponential_M2_19,
            constants.M_ExponentialM3: models.Exponential_M3_19,
            constants.M_ExponentialM4: models.Exponential_M4_19,
            constants.M_ExponentialM5: models.Exponential_M5_19,
        },
        constants.CONTINUOUS_INDIVIDUAL: {
            constants.M_Linear: models.Linear_217,
            constants.M_Polynomial: models.Polynomial_217,
            constants.M_Power: models.Power_217,
            constants.M_Hill: models.Hill_217,
            constants.M_ExponentialM2: models.Exponential_M2_19,
            constants.M_ExponentialM3: models.Exponential_M3_19,
            constants.M_ExponentialM4: models.Exponential_M4_19,
            constants.M_ExponentialM5: models.Exponential_M5_19,
        },
    }


class BMDS_v260(BMDS_v240):
    version_str = constants.BMDS260
    version_pretty = "BMDS v2.6.0"
    version_tuple = (2, 6, 0)
    model_options = {
        constants.DICHOTOMOUS: {
            constants.M_Logistic: models.Logistic_214,
            constants.M_LogLogistic: models.LogLogistic_214,
            constants.M_Probit: models.Probit_33,
            constants.M_LogProbit: models.LogProbit_33,
            constants.M_QuantalLinear: models.QuantalLinear_34,
            constants.M_Multistage: models.Multistage_34,
            constants.M_Gamma: models.Gamma_216,
            constants.M_Weibull: models.Weibull_216,
            constants.M_DichotomousHill: models.DichotomousHill_13,
        },
        constants.DICHOTOMOUS_CANCER: {constants.M_MultistageCancer: models.MultistageCancer_110},
        constants.CONTINUOUS: {
            constants.M_Linear: models.Linear_220,
            constants.M_Polynomial: models.Polynomial_220,
            constants.M_Power: models.Power_218,
            constants.M_Hill: models.Hill_217,
            constants.M_ExponentialM2: models.Exponential_M2_110,
            constants.M_ExponentialM3: models.Exponential_M3_110,
            constants.M_ExponentialM4: models.Exponential_M4_110,
            constants.M_ExponentialM5: models.Exponential_M5_110,
        },
        constants.CONTINUOUS_INDIVIDUAL: {
            constants.M_Linear: models.Linear_220,
            constants.M_Polynomial: models.Polynomial_220,
            constants.M_Power: models.Power_218,
            constants.M_Hill: models.Hill_217,
            constants.M_ExponentialM2: models.Exponential_M2_110,
            constants.M_ExponentialM3: models.Exponential_M3_110,
            constants.M_ExponentialM4: models.Exponential_M4_110,
            constants.M_ExponentialM5: models.Exponential_M5_110,
        },
    }


class BMDS_v2601(BMDS_v240):
    version_str = constants.BMDS2601
    version_pretty = "BMDS v2.6.0.1"
    version_tuple = (2, 6, 0, 1)


class BMDS_v270(BMDS_v2601):
    version_str = constants.BMDS270
    version_pretty = "BMDS v2.7.0"
    version_tuple = (2, 7, 0)
    model_options = {
        constants.DICHOTOMOUS: {
            constants.M_Logistic: models.Logistic_215,
            constants.M_LogLogistic: models.LogLogistic_215,
            constants.M_Probit: models.Probit_34,
            constants.M_LogProbit: models.LogProbit_34,
            constants.M_QuantalLinear: models.QuantalLinear_34,
            constants.M_Multistage: models.Multistage_34,
            constants.M_Gamma: models.Gamma_217,
            constants.M_Weibull: models.Weibull_217,
            constants.M_DichotomousHill: models.DichotomousHill_13,
        },
        constants.DICHOTOMOUS_CANCER: {constants.M_MultistageCancer: models.MultistageCancer_34},
        constants.CONTINUOUS: {
            constants.M_Linear: models.Linear_221,
            constants.M_Polynomial: models.Polynomial_221,
            constants.M_Power: models.Power_219,
            constants.M_Hill: models.Hill_218,
            constants.M_ExponentialM2: models.Exponential_M2_111,
            constants.M_ExponentialM3: models.Exponential_M3_111,
            constants.M_ExponentialM4: models.Exponential_M4_111,
            constants.M_ExponentialM5: models.Exponential_M5_111,
        },
        constants.CONTINUOUS_INDIVIDUAL: {
            constants.M_Linear: models.Linear_221,
            constants.M_Polynomial: models.Polynomial_221,
            constants.M_Power: models.Power_219,
            constants.M_Hill: models.Hill_218,
            constants.M_ExponentialM2: models.Exponential_M2_111,
            constants.M_ExponentialM3: models.Exponential_M3_111,
            constants.M_ExponentialM4: models.Exponential_M4_111,
            constants.M_ExponentialM5: models.Exponential_M5_111,
        },
    }


class BMDS_v330(BMDS):
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
        constants.DICHOTOMOUS_CANCER: {constants.M_MultistageCancer: d3.Multistage},
        constants.CONTINUOUS: {
            constants.M_Linear: c3.Linear,
            constants.M_Polynomial: c3.Polynomial,
            constants.M_Power: c3.Power,
            constants.M_Hill: c3.Hill,
            constants.M_ExponentialM2: c3.ExponentialM2,
            constants.M_ExponentialM3: c3.ExponentialM3,
            constants.M_ExponentialM4: c3.ExponentialM4,
            constants.M_ExponentialM5: c3.ExponentialM5,
        },
        constants.CONTINUOUS_INDIVIDUAL: {
            constants.M_Linear: c3.Linear,
            constants.M_Polynomial: c3.Polynomial,
            constants.M_Power: c3.Power,
            constants.M_Hill: c3.Hill,
            constants.M_ExponentialM2: c3.ExponentialM2,
            constants.M_ExponentialM3: c3.ExponentialM3,
            constants.M_ExponentialM4: c3.ExponentialM4,
            constants.M_ExponentialM5: c3.ExponentialM5,
        },
    }


_BMDS_VERSIONS = {
    constants.BMDS231: BMDS_v231,
    constants.BMDS240: BMDS_v240,
    constants.BMDS260: BMDS_v260,
    constants.BMDS2601: BMDS_v2601,
    constants.BMDS270: BMDS_v270,
    constants.BMDS330: BMDS_v330,
}
