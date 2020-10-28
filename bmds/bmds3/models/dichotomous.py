import ctypes
from typing import Callable, Dict, List, Tuple, Union

from pydantic import BaseModel as PydanticBaseModel
from pydantic import confloat, conint

from ...datasets import DichotomousDataset
from .. import constants, types, types33
from .base import BaseModel, BmdsFunctionManager


class DichotomousModelSettings(PydanticBaseModel):
    bmr: confloat(gt=0) = 0.1
    alpha: confloat(gt=0, lt=1) = 0.05
    background: int = -9999
    bmrType: types.RiskType_t = types.RiskType_t.eExtraRisk
    degree: conint(ge=0, le=8) = 0

    def degree_param_names(self) -> List[str]:
        return [chr(97 + i) for i in range(self.degree)]


class Dichotomous(BaseModel):
    model_id: constants.DichotomousModel
    param_names: Tuple[str, ...] = ()

    @property
    def num_params(self) -> int:
        return len(self.param_names)

    def get_dll_default_frequentist_priors(self) -> List[types.PRIOR]:
        """
        Returns the default list of frequentist priors for a model.
        """
        raise NotImplementedError()

    def _get_dll_default_options(self) -> Tuple[types.BMDS_D_Opts1_t, types.BMDS_D_Opts2_t]:
        d_opts1 = types.BMDS_D_Opts1_t(
            bmr=self.settings.bmr, alpha=self.settings.alpha, background=self.settings.background
        )
        d_opts2 = types.BMDS_D_Opts2_t(
            bmrType=self.settings.bmrType.value, degree=self.settings.degree
        )
        return d_opts1, d_opts2

    def _build_dll_result(self, dataset: DichotomousDataset) -> types.BMD_ANAL:
        num_dg = dataset.num_dose_groups

        _dGoF_t = types.dGoF_t()
        _dGoF_t.pzRow = (types.GoFRow_t * num_dg)()

        analysis = types.BMD_ANAL()
        analysis.PARMS = (ctypes.c_double * types.MY_MAX_PARMS)()
        analysis.boundedParms = (ctypes.c_bool * types.MY_MAX_PARMS)()
        analysis.aCDF = (ctypes.c_double * types.CDF_TABLE_SIZE)()
        analysis.deviance = (types.DichotomousDeviance_t * num_dg)()
        analysis.gof = ctypes.pointer(_dGoF_t)
        analysis.nCDF = types.CDF_TABLE_SIZE

        return analysis

    @property
    def _func(self) -> Callable:
        return BmdsFunctionManager.get_dll_func(
            bmds_version="BMDS330", base_name="libDRBMD", func_name="estimate_sm_laplace_dicho"
        )

    def execute(self) -> types33.DichotomousModelResult:

        # setup inputs
        inputs = types33.DichotomousAnalysis(
            model=self.model_id,
            dataset=self.dataset,
            prior=[1.0, 2.0, 0.0, 0.1, 2.0, 1.0, -20.0, 1e-12, 20.0, 100.0],
            BMD_type=1,
            BMR=0.1,
            alpha=0.05,
            degree=len(self.model_id.params) - 1,
            samples=100,
            burnin=20,
        )

        # setup outputs
        results = types33.DichotomousModelResult(model=self.model_id, dist_numE=200)
        results_struct = results.to_c()

        self._func(ctypes.pointer(inputs.to_c()), ctypes.pointer(results_struct), True)
        results.from_c()

        return results

    def get_model_settings(
        self, settings: Union[DichotomousModelSettings, Dict]
    ) -> DichotomousModelSettings:
        if isinstance(settings, DichotomousModelSettings):
            return settings
        return DichotomousModelSettings.parse_obj(settings)


class Logistic(Dichotomous):
    model_id = constants.DichotomousModel.d_logistic

    def get_dll_default_frequentist_priors(self) -> List[types.PRIOR]:
        """
        Returns the default list of frequentist priors for a model.
        """
        return [
            types.PRIOR(type=0, initialValue=-2, stdDev=1, minValue=-18, maxValue=18),
            types.PRIOR(type=0, initialValue=0.1, stdDev=1, minValue=1, maxValue=10),
        ]


class LogLogistic(Dichotomous):
    model_id = types.DModelID_t.eLogLogistic
    param_names = ("a", "b", "c")

    def get_dll_default_frequentist_priors(self) -> List[types.PRIOR]:
        """
        Returns the default list of frequentist priors for a model.
        """
        return [
            types.PRIOR(type=0, initialValue=-2.0, stdDev=1, minValue=-18, maxValue=18),
            types.PRIOR(type=0, initialValue=-2.0, stdDev=1, minValue=-18, maxValue=18),
            types.PRIOR(type=0, initialValue=1.0, stdDev=1, minValue=1e-4, maxValue=18),
        ]


class Probit(Dichotomous):
    model_id = types.DModelID_t.eProbit
    param_names = ("a", "b")

    def get_dll_default_frequentist_priors(self) -> List[types.PRIOR]:
        """
        Returns the default list of frequentist priors for a model.
        """
        return [
            types.PRIOR(type=0, initialValue=-2, stdDev=1, minValue=-18, maxValue=18),
            types.PRIOR(type=0, initialValue=0.1, stdDev=1, minValue=0, maxValue=18),
        ]


class LogProbit(Dichotomous):
    model_id = types.DModelID_t.eLogProbit
    param_names = ("a", "b", "c")

    def get_dll_default_frequentist_priors(self) -> List[types.PRIOR]:
        """
        Returns the default list of frequentist priors for a model.
        """
        return [
            types.PRIOR(type=0, initialValue=-2.0, stdDev=1, minValue=-18, maxValue=18),
            types.PRIOR(type=0, initialValue=-3.0, stdDev=1, minValue=-18, maxValue=18),
            types.PRIOR(type=0, initialValue=1.0, stdDev=1, minValue=1e-4, maxValue=18),
        ]


class Gamma(Dichotomous):
    model_id = types.DModelID_t.eGamma
    param_names = ("a", "b", "c")

    def get_dll_default_frequentist_priors(self) -> List[types.PRIOR]:
        """
        Returns the default list of frequentist priors for a model.
        """
        return [
            types.PRIOR(type=0, initialValue=-2.0, stdDev=1, minValue=-18, maxValue=18),
            types.PRIOR(type=0, initialValue=1.0, stdDev=1, minValue=0.2, maxValue=18),
            types.PRIOR(type=0, initialValue=0.1, stdDev=1, minValue=0, maxValue=100),
        ]


class QuantalLinear(Dichotomous):
    model_id = types.DModelID_t.eQLinear
    param_names = ("a", "b")

    def get_dll_default_frequentist_priors(self) -> List[types.PRIOR]:
        """
        Returns the default list of frequentist priors for a model.
        """
        return [
            types.PRIOR(type=0, initialValue=-2.0, stdDev=1, minValue=-18, maxValue=18),
            types.PRIOR(type=0, initialValue=0.5, stdDev=1, minValue=0, maxValue=100),
        ]


class Weibull(Dichotomous):
    model_id = types.DModelID_t.eWeibull
    param_names = ("a", "b", "c")

    def get_dll_default_frequentist_priors(self) -> List[types.PRIOR]:
        """
        Returns the default list of frequentist priors for a model.
        """
        return [
            types.PRIOR(type=0, initialValue=-2.0, stdDev=1, minValue=-18, maxValue=18),
            types.PRIOR(type=0, initialValue=0.5, stdDev=1, minValue=1e-6, maxValue=18),
            types.PRIOR(type=0, initialValue=1.0, stdDev=1, minValue=1e-6, maxValue=100),
        ]


class Multistage(Dichotomous):
    model_id = types.DModelID_t.eMultistage

    def get_model_settings(self, settings: Dict) -> DichotomousModelSettings:
        settings.setdefault("degree", 2)
        settings = DichotomousModelSettings.parse_obj(settings)
        if settings.degree < 2:
            raise ValueError(f"Degree must be at least 2: {settings.degree}")
        return settings

    @property
    def param_names(self) -> Tuple[str, ...]:
        params: List[str] = ["g"]
        params.extend(self.settings.degree_param_names())
        return tuple(params)

    @property
    def num_params(self) -> int:
        return self.settings.degree + 1

    def get_dll_default_frequentist_priors(self) -> List[types.PRIOR]:
        """
        Returns the default list of frequentist priors for a model.
        """

        priors = [
            types.PRIOR(type=0, initialValue=-17, stdDev=0, minValue=-18, maxValue=18),
            types.PRIOR(type=0, initialValue=0.1, stdDev=0, minValue=0, maxValue=100),
        ]
        if self.settings.degree > 2:
            priors.extend(
                [
                    types.PRIOR(type=0, initialValue=0.1, stdDev=0, minValue=0, maxValue=1e4)
                    for degree in range(2, self.settings.degree)
                ]
            )
        return priors


class DichotomousHill(Dichotomous):
    model_id = types.DModelID_t.eDHill
    param_names = ("a", "b", "c", "d")

    def get_dll_default_frequentist_priors(self) -> List[types.PRIOR]:
        """
        Returns the default list of frequentist priors for a model.
        """
        return [
            types.PRIOR(type=0, initialValue=-2.0, stdDev=1, minValue=-18, maxValue=18),
            types.PRIOR(type=0, initialValue=0.0, stdDev=1, minValue=-18, maxValue=18),
            types.PRIOR(type=0, initialValue=0.0, stdDev=1, minValue=-18, maxValue=18),
            types.PRIOR(type=0, initialValue=1.0, stdDev=1, minValue=-1e-8, maxValue=18),
        ]
