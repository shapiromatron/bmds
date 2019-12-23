import ctypes
import json
from typing import Callable, Dict, List, Tuple

import numpy as np

from ...datasets import DichotomousDataset
from ...utils import get_dll_func
from .base import BaseModel
from .. import types


class DichotomousResult:
    def __init__(self, data: Dict):
        self.data = data
        self._str = None

    @classmethod
    def from_execution(cls, result: types.BMD_ANAL) -> 'DichotomousResult':
        data = dict(
            map=result.MAP,
            bmd=result.BMD,
            bmdl=result.BMDL,
            bmdu=result.BMDU,
            aic=result.AIC,
            bic=result.BIC_Equiv,
            num_parms=result.nparms,
            cdf=np.array(result.aCDF[:result.nCDF])
        )
        return cls(data=data)

    @classmethod
    def deserialize(cls, json_str: str) -> 'DichotomousResult':
        # TODO - implement
        return cls(data=json.loads(json_str))

    def serialize(self):
        # convert arrays to np.array
        return json.dumps(self._dict)

    def as_dict(self) -> Dict:
        return self.data

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

        response_code = self._func(
            model_id,
            ctypes.pointer(results),
            model_type,
            dataset_arrary,
            priors,
            ctypes.pointer(d_opts1),
            ctypes.pointer(d_opts2),
            ctypes.pointer(n),
        )
        print(response_code)
        return DichotomousResult.from_execution(results)


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

    def __init__(self, degree: int = 2):
        if degree < 2 or degree > 8:
            raise ValueError(f"Invalid degree: {degree}")
        self.degree = degree

    @property
    def param_names(self):
        params = ["g"]
        params.extend([f"{chr(97 + i)}" for i in range(self.degree)])
        return tuple(params)

    def get_dll_default_frequentist_priors(self) -> List[types.PRIOR]:
        """
        Returns the default list of frequentist priors for a model.
        """

        priors = [
            types.PRIOR(type=0, initialValue=-17, stdDev=0, minValue=-18, maxValue=18),
            types.PRIOR(type=0, initialValue=0.1, stdDev=0, minValue=0, maxValue=100),
        ]
        if self.degree > 2:
            priors.extend([
                types.PRIOR(type=0, initialValue=0.1, stdDev=0, minValue=0, maxValue=1e4)
                for degree in range(2, self.degree)
            ])
        return priors

    def get_dll_default_options(self) -> Tuple[types.BMDS_D_Opts1_t, types.BMDS_D_Opts2_t]:
        (d_opts1, d_opts2) = super().get_dll_default_options()
        d_opts2.degree = self.degree
        return d_opts1, d_opts2


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
