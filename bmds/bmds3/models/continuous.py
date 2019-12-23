import ctypes
from typing import Callable, List, Optional, Tuple

from ...datasets import ContinuousDataset
from ...utils import get_dll_func
from .base import BaseModel
from .. import types


class ContinuousResult:
    def __init__(self, results: types.BMD_C_ANAL):
        self.results = results


class Continuous(BaseModel):
    _func: Callable = get_dll_func(
        bmds_version="BMDS312", base_name="cmodels", func_name="run_cmodel"
    )
    model_id: types.CModelID_t
    param_names: Optional[Tuple[str, ...]]

    def __init__(self):
        self.degree = len(self.param_names)

    def get_dll_default_frequentist_priors(self) -> List[types.PRIOR]:
        """
        Returns the default list of frequentist priors for a model.
        """
        raise NotImplementedError()

    def get_dll_default_options(self) -> types.BMDS_C_Options_t:
        return types.BMDS_C_Options_t(
            bmr=ctypes.c_double(0.1),
            alpha=ctypes.c_double(0.05),
            background=ctypes.c_double(-9999),
            tailProb=ctypes.c_double(0.01),
            bmrType=ctypes.c_int(types.BMRType_t.eRelativeDev.value),
            degree=ctypes.c_int(0),
            adverseDirection=ctypes.c_int(0),
            restriction=ctypes.c_int(1),
            varType=ctypes.c_int(types.VarType_t.eConstant.value),
            bLognormal=ctypes.c_bool(False),
            bUserParmInit=ctypes.c_bool(False),
        )

    def execute_dll(self, dataset: ContinuousDataset) -> ContinuousResult:
        model_id = ctypes.c_int(self.model_id.value)
        input_type = ctypes.c_int(types.BMDSInputType_t.eCont_4.value)

        # one row for each dose-group
        dataset_, results = dataset.build_dll_dataset_and_analysis()
        n = ctypes.c_int(len(dataset_))

        # using default priors
        priors_ = self.get_dll_default_frequentist_priors()
        priors = (types.PRIOR * len(priors_))(*priors_)

        # using default options
        options = self.get_dll_default_options()

        response_code = self._func(
            ctypes.pointer(model_id),
            ctypes.pointer(results),
            ctypes.pointer(input_type),
            dataset_,
            priors,
            ctypes.pointer(options),
            ctypes.pointer(n),
        )

        print(response_code)
        print(results)
        return ContinuousResult(results)


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
        if degree < 1 or degree > 8:
            raise ValueError(f"Invalid degree: {degree}")
        self.degree = degree

    @property
    def param_names(self):
        params = ["g"]
        params.extend([f"b{i + 1}" for i in range(self.degree)])
        return tuple(params)

    def get_dll_default_frequentist_priors(self) -> List[types.PRIOR]:
        """
        Returns the default list of frequentist priors for a model.
        """
        priors = [
            types.PRIOR(type=0, initialValue=0, stdDev=2, minValue=-18, maxValue=18),
        ]
        priors.extend([
            types.PRIOR(type=0, initialValue=1, stdDev=2, minValue=-18, maxValue=18)
            for degree in range(self.degree)
        ])
        return priors

    def get_dll_default_options(self) -> types.BMDS_C_Options_t:
        options = super().get_dll_default_options()
        options.degree = ctypes.c_int(self.degree)
        return options


class Linear(Polynomial):
    def __init__(self):
        self.degree = 1
