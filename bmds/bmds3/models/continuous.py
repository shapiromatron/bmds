import ctypes
from typing import Callable, List, Tuple

from ...datasets import ContinuousDataset
from ...utils import get_dll_func
from .. import types
from .base import BaseModel


class Continuous(BaseModel):
    _func: Callable = get_dll_func(
        bmds_version="BMDS312", base_name="cmodels", func_name="run_cmodel"
    )
    model_id: types.CModelID_t
    param_names: Tuple[str, ...] = ()

    @property
    def num_params(self) -> int:
        params = len(self.param_names)
        if self.variance == types.VarType_t.eConstant:
            return params + 1
        elif self.variance == types.VarType_t.eModeled:
            return params + 2
        else:
            raise ValueError("Unknown variance type")

    def __init__(self):
        self.variance: types.VarType_t = types.VarType_t.eConstant
        self.degree = 1

    def get_dll_default_frequentist_priors(self) -> List[types.PRIOR]:
        """
        Returns the default list of frequentist priors for a model.
        """
        raise NotImplementedError()

    def get_dll_default_options(self) -> types.BMDS_C_Options_t:
        return types.BMDS_C_Options_t(
            bmr=0.1,
            alpha=0.05,
            background=-9999,
            tailProb=0.01,
            bmrType=types.BMRType_t.eRelativeDev.value,
            degree=0,
            adverseDirection=0,
            restriction=1,
            varType=self.variance.value,
            bLognormal=ctypes.c_bool(False),
            bUserParmInit=ctypes.c_bool(False),
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

    def execute(self, dataset: ContinuousDataset) -> types.ContinuousResult:
        model_id = ctypes.c_int(self.model_id.value)
        input_type = ctypes.c_int(types.BMDSInputType_t.eCont_4.value)

        # one row for each dose-group
        dataset_array = dataset._build_dll_dataset()
        results = self._build_dll_result(dataset)
        n = ctypes.c_int(dataset.num_dose_groups)

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
            response_code, results, dataset.num_dose_groups, self.num_params
        )


class Exponential(Continuous):
    def get_dll_default_options(self) -> types.BMDS_C_Options_t:
        options = super().get_dll_default_options()
        options.degree = ctypes.c_int(self.degree)
        return options


class ExponentialM2(Exponential):
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


class ExponentialM3(Exponential):
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


class ExponentialM4(Exponential):
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


class ExponentialM5(Exponential):
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

    def __init__(self, degree: int = 2):
        super().__init__()
        if degree < 1 or degree > 8:
            raise ValueError(f"Invalid degree: {degree}")
        self.degree = degree

    @property
    def param_names(self):
        params = ["g"]
        params.extend([f"b{i + 1}" for i in range(self.degree)])
        return tuple(params)

    @property
    def num_params(self) -> int:
        return self.degree + 1

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
                for degree in range(self.degree)
            ]
        )
        return priors

    def get_dll_default_options(self) -> types.BMDS_C_Options_t:
        options = super().get_dll_default_options()
        options.degree = ctypes.c_int(self.degree)
        return options


class Linear(Polynomial):
    def __init__(self):
        super().__init__()
        self.degree = 1
