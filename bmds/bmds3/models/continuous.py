import ctypes
from typing import Callable, Dict, List, Tuple

from pydantic import BaseModel as PydanticBaseModel
from pydantic import confloat, conint

from ...datasets import ContinuousDataset
from .. import types
from .base import BaseModel, BmdsFunctionManager


class ContinuousModelSettings(PydanticBaseModel):
    bmr: confloat(gt=0) = 0.1
    alpha: confloat(gt=0, lt=1) = 0.05
    tailProb: confloat(gt=0, lt=1) = 0.01
    background: int = -9999
    bmrType: types.BMRType_t = types.BMRType_t.eRelativeDev
    degree: conint(ge=0, le=8) = 0
    adverseDirection: conint(ge=0, le=1) = 0
    restriction: conint(ge=0, le=1) = 1
    varType: types.VarType_t = types.VarType_t.eVarTypeNone
    bLognormal: bool = False
    bUserParmInit: bool = False


class Continuous(BaseModel):
    model_id: types.CModelID_t
    param_names: Tuple[str, ...] = ()

    @property
    def num_params(self) -> int:
        return len(self.param_names) + self.settings.varType.num_params()

    def get_dll_default_frequentist_priors(self) -> List[types.PRIOR]:
        """
        Returns the default list of frequentist priors for a model.
        """
        raise NotImplementedError()

    def get_dll_default_options(self) -> types.BMDS_C_Options_t:
        settings = self.settings
        return types.BMDS_C_Options_t(
            bmr=settings.bmr,
            alpha=settings.alpha,
            background=settings.background,
            tailProb=settings.tailProb,
            bmrType=settings.bmrType.value,
            degree=settings.degree,
            adverseDirection=settings.adverseDirection,
            restriction=settings.restriction,
            varType=settings.varType,
            bLognormal=ctypes.c_bool(settings.bLognormal),
            bUserParmInit=ctypes.c_bool(settings.bUserParmInit),
        )

    def _build_dll_result(self, dataset: ContinuousDataset) -> types.BMD_C_ANAL:
        deviance = types.ContinuousDeviance_t()
        deviance.llRows = (types.LLRow_t * types.NUM_LIKELIHOODS_OF_INTEREST)()
        deviance.testRows = (types.TestRow_t * types.NUM_TESTS_OF_INTEREST)()

        analysis = types.BMD_C_ANAL()
        analysis.deviance = deviance
        analysis.PARMS = (ctypes.c_double * types.MY_MAX_PARMS)()
        analysis.gofRow = (types.cGoFRow_t * dataset.num_dose_groups)()
        analysis.boundedParms = (ctypes.c_bool * types.MY_MAX_PARMS)()
        analysis.aCDF = (ctypes.c_double * types.CDF_TABLE_SIZE)()
        analysis.nCDF = types.CDF_TABLE_SIZE

        return analysis

    @property
    def _func(self) -> Callable:
        return BmdsFunctionManager.get_dll_func(
            bmds_version="BMDS312", base_name="cmodels", func_name="run_cmodel"
        )

    def execute(self) -> types.ContinuousResult:
        model_id = ctypes.c_int(self.model_id.value)
        input_type = ctypes.c_int(types.BMDSInputType_t.eCont_4.value)

        # one row for each dose-group
        dataset_array = self.dataset._build_dll_dataset()
        results = self._build_dll_result(self.dataset)
        n = ctypes.c_int(self.dataset.num_dose_groups)

        # using default priors
        priors_ = self.get_dll_default_frequentist_priors()
        priors = (types.PRIOR * len(priors_))(*priors_)

        # using default options
        options = self.get_dll_default_options()

        response_code = self._func(
            ctypes.pointer(model_id),
            ctypes.pointer(results),
            ctypes.pointer(input_type),
            dataset_array,
            priors,
            ctypes.pointer(options),
            ctypes.pointer(n),
        )

        return types.ContinuousResult.from_execution(
            response_code, results, self.dataset.num_dose_groups, self.num_params
        )

    def get_model_settings(self, settings: Dict) -> ContinuousModelSettings:
        if "varType" not in settings:
            settings["varType"] = self.dataset.get_default_variance_model()
        return ContinuousModelSettings.parse_obj(settings)


class ExponentialM2(Continuous):
    model_id = types.CModelID_t.eExp2
    param_names = ("a", "b")

    def get_dll_default_frequentist_priors(self) -> List[types.PRIOR]:
        """
        Returns the default list of frequentist priors for a model.
        """
        return [
            types.PRIOR(type=0, initialValue=0, stdDev=2, minValue=-18, maxValue=18),
            types.PRIOR(type=0, initialValue=1, stdDev=2, minValue=-18, maxValue=18),
            types.PRIOR(
                type=0, initialValue=1, stdDev=2, minValue=-18, maxValue=18
            ),  # for variance?
        ]


class ExponentialM3(Continuous):
    model_id = types.CModelID_t.eExp3
    param_names = ("a", "b", "c")

    def get_dll_default_frequentist_priors(self) -> List[types.PRIOR]:
        """
        Returns the default list of frequentist priors for a model.
        """
        return [
            types.PRIOR(type=0, initialValue=0, stdDev=2, minValue=-18, maxValue=18),
            types.PRIOR(type=0, initialValue=1, stdDev=2, minValue=-18, maxValue=18),
            types.PRIOR(type=0, initialValue=-5, stdDev=0.5, minValue=-18, maxValue=18),
            types.PRIOR(
                type=0, initialValue=1, stdDev=2, minValue=-18, maxValue=18
            ),  # for variance?
        ]


class ExponentialM4(Continuous):
    model_id = types.CModelID_t.eExp4
    param_names = ("a", "b", "c", "d")

    def get_dll_default_frequentist_priors(self) -> List[types.PRIOR]:
        """
        Returns the default list of frequentist priors for a model.
        """
        return [
            types.PRIOR(type=0, initialValue=0, stdDev=2, minValue=-18, maxValue=18),
            types.PRIOR(type=0, initialValue=1, stdDev=2, minValue=-18, maxValue=18),
            types.PRIOR(type=0, initialValue=-5, stdDev=0.5, minValue=-18, maxValue=18),
            types.PRIOR(type=0, initialValue=1.5, stdDev=0.2501, minValue=1e8, maxValue=18),
            types.PRIOR(
                type=0, initialValue=1, stdDev=2, minValue=-18, maxValue=18
            ),  # for variance?
        ]


class ExponentialM5(Continuous):
    model_id = types.CModelID_t.eExp5
    param_names = ("a", "b", "c", "d", "g")

    def get_dll_default_frequentist_priors(self) -> List[types.PRIOR]:
        """
        Returns the default list of frequentist priors for a model.
        """
        return [
            types.PRIOR(type=0, initialValue=0, stdDev=2, minValue=-18, maxValue=18),
            types.PRIOR(type=0, initialValue=1, stdDev=2, minValue=-18, maxValue=18),
            types.PRIOR(type=0, initialValue=-5, stdDev=0.5, minValue=-18, maxValue=18),
            types.PRIOR(type=0, initialValue=1.5, stdDev=0.2501, minValue=1e8, maxValue=18),
            types.PRIOR(type=0, initialValue=1.5, stdDev=0.2501, minValue=1e8, maxValue=18),
            types.PRIOR(
                type=0, initialValue=1, stdDev=2, minValue=-18, maxValue=18
            ),  # for variance?
        ]


class Power(Continuous):
    model_id = types.CModelID_t.ePow
    param_names = ("g", "v", "n")

    def get_dll_default_frequentist_priors(self) -> List[types.PRIOR]:
        """
        Returns the default list of frequentist priors for a model.
        """
        return [
            types.PRIOR(type=0, initialValue=0, stdDev=2, minValue=-18, maxValue=18),
            types.PRIOR(type=0, initialValue=1, stdDev=2, minValue=-18, maxValue=18),
            types.PRIOR(type=0, initialValue=-5, stdDev=0.5, minValue=-18, maxValue=18),
            types.PRIOR(type=0, initialValue=1, stdDev=2, minValue=-18, maxValue=18),
        ]


class Hill(Continuous):
    model_id = types.CModelID_t.eHill
    param_names = ("g", "v", "k", "n")

    def get_dll_default_frequentist_priors(self) -> List[types.PRIOR]:
        """
        Returns the default list of frequentist priors for a model.
        """
        return [
            types.PRIOR(type=0, initialValue=0, stdDev=2, minValue=-18, maxValue=18),
            types.PRIOR(type=0, initialValue=1, stdDev=2, minValue=-18, maxValue=18),
            types.PRIOR(type=0, initialValue=-5, stdDev=0.5, minValue=-18, maxValue=18),
            types.PRIOR(type=0, initialValue=1.5, stdDev=0.2501, minValue=1e-8, maxValue=18),
        ]


class Polynomial(Continuous):
    model_id = types.CModelID_t.ePoly

    @property
    def param_names(self):
        params = ["g"]
        params.extend([f"b{i + 1}" for i in range(self.settings.degree)])
        return tuple(params)

    def get_dll_default_frequentist_priors(self) -> List[types.PRIOR]:
        """
        Returns the default list of frequentist priors for a model.
        """
        priors = [
            types.PRIOR(type=0, initialValue=0, stdDev=2, minValue=-18, maxValue=18),
        ]
        priors.extend(
            [
                types.PRIOR(type=0, initialValue=1, stdDev=2, minValue=-18, maxValue=18)
                for degree in range(self.settings.degree)
            ]
        )
        return priors

    def get_model_settings(self, settings: Dict) -> ContinuousModelSettings:
        settings.setdefault("degree", 2)
        return super().get_model_settings(settings)


class Linear(Polynomial):
    def get_model_settings(self, settings: Dict) -> ContinuousModelSettings:
        settings["degree"] = 1
        return super().get_model_settings(settings)
