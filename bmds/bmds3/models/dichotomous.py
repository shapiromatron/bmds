import ctypes
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ...datasets import DichotomousDataset
from ...utils import get_dll_func
from .base import BaseModel
from .. import types


class DichotomousResult:
    def __init__(self, dll_result: types.BMD_ANAL):
        self.model_name: str
        self.dll_result: types.BMD_ANAL = dll_result
        self._dict: Optional[Dict] = None
        self._str: Optional[str] = None

    def as_dict(self) -> Dict:
        if self._dict is None:
            self._dict = dict(
                map=self.dll_result.MAP,
                bmd=self.dll_result.BMD,
                bmdl=self.dll_result.BMDL,
                bmdu=self.dll_result.BMDU,
                aic=self.dll_result.AIC,
                bic=self.dll_result.BIC_Equiv,
                num_parms=self.dll_result.nparms,
                cdf=np.array(self.dll_result.aCDF[:self.dll_result.nCDF])
            )
        return self._dict

    def __str__(self) -> str:
        if self._str is None:
            self._str = str(self.as_dict())
        return self._str


class Dichotomous(BaseModel):
    _func: Callable = get_dll_func(
        bmds_version="BMDS312", base_name="bmds_models", func_name="run_dmodel2"
    )
    model_id: types.DModelID_t

    def get_dll_default_frequentist_priors(self) -> List[types.PRIOR]:
        """
        Returns the default list of frequentist priors for a model.
        """
        raise NotImplementedError()

    def get_dll_default_options(self) -> Tuple[types.BMDS_D_Opts1_t, types.BMDS_D_Opts2_t]:
        d_opts1 = types.BMDS_D_Opts1_t(bmr=0.1, alpha=0.05, background=-9999)
        d_opts2 = types.BMDS_D_Opts2_t(bmrType=types.BMRType_t.eAbsoluteDev.value, degree=0)
        return d_opts1, d_opts2

    def execute_dll(self, dataset: DichotomousDataset) -> DichotomousResult:
        model_id = (ctypes.c_int * 1)(self.model_id.value)
        model_type = (ctypes.c_int * 1)(self.model_id.value)

        dataset_arrary, results = dataset.build_dll_dataset_and_analysis()
        n = ctypes.c_int(len(dataset_arrary))

        priors_ = self.get_dll_default_frequentist_priors()
        priors = (types.PRIOR * len(priors_))(*priors_)

        d_opts1, d_opts2 = self.get_dll_default_options()

        self._func(
            model_id,
            ctypes.pointer(results),
            model_type,
            dataset_arrary,
            priors,
            ctypes.pointer(d_opts1),
            ctypes.pointer(d_opts2),
            ctypes.pointer(n),
        )
        return DichotomousResult(results)


class Logistic(Dichotomous):
    model_id = types.DModelID_t.eLogistic
    param_names = ('a', 'b')

    def get_dll_default_frequentist_priors(self) -> List[types.PRIOR]:
        """
        Returns the default list of frequentist priors for a model.
        """
        return [
            types.PRIOR(type=0, initialValue=-2, stdDev=1, minValue=-18, maxValue=18),
            types.PRIOR(type=0, initialValue=0.1, stdDev=1, minValue=1, maxValue=100),
        ]


class LogLogistic(Dichotomous):
    model_id = types.DModelID_t.eLogLogistic
    param_names = ('a', 'b', 'c')

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

    def get_dll_default_frequentist_priors(self) -> List[types.PRIOR]:
        """
        Returns the default list of frequentist priors for a model.
        """
        return [
            types.PRIOR(type=0, initialValue=-17, stdDev=1, minValue=-18, maxValue=18),
            types.PRIOR(type=0, initialValue=0.1, stdDev=1, minValue=-18, maxValue=100),
            types.PRIOR(type=0, initialValue=0.1, stdDev=1, minValue=-18, maxValue=1e4),
        ]


class DichotomousHill(Dichotomous):
    model_id = types.DModelID_t.eDHill

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
