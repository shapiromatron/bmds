import ctypes
from typing import List

import numpy as np
from pydantic import BaseModel

from .. import constants
from .common import list_t_c
from .dichotomous import DichotomousModelResultStruct


class DichotomousMAAnalysisStruct(ctypes.Structure):
    _fields_ = [
        ("nmodels", ctypes.c_int),  # number of models for the model average
        (
            "priors",
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
        ),  # list of pointers to prior arrays, priors[i] is the prior array for the ith model ect
        ("nparms", ctypes.POINTER(ctypes.c_int)),  # parameters in each model
        ("actual_parms", ctypes.POINTER(ctypes.c_int)),  # actual number of parameters in the model
        (
            "prior_cols",
            ctypes.POINTER(ctypes.c_int),
        ),  # columns in the prior if there are more in the future, presently there are only 5
        ("models", ctypes.POINTER(ctypes.c_int)),  # list of models
        ("modelPriors", ctypes.POINTER(ctypes.c_double)),  # prior probability on the model
    ]


class DichotomousMAAnalysis(BaseModel):
    models: List[constants.DichotomousModel]
    priors: List[List[constants.Prior]]

    class Config:
        arbitrary_types_allowed = True

    def _priors_to_list(self, priors) -> List[float]:
        """
        allocate memory for all parameters and convert to columnwise matrix
        """
        arr = np.array([list(prior.dict().values()) for prior in priors])

        return arr.T.flatten().tolist()

    def to_c(self) -> DichotomousMAAnalysisStruct:
        _priors = [self._priors_to_list(model_priors) for model_priors in self.priors]
        _prior_arrays = [list_t_c(model_priors, ctypes.c_double) for model_priors in _priors]
        _prior_pointers = [
            ctypes.cast(prior_array, ctypes.POINTER(ctypes.c_double))
            for prior_array in _prior_arrays
        ]
        priors = list_t_c(_prior_pointers, ctypes.POINTER(ctypes.c_double))
        return DichotomousMAAnalysisStruct(
            nmodels=ctypes.c_int(len(self.models)),
            priors=priors,
            actual_parms=list_t_c([model.num_params for model in self.models], ctypes.c_int),
            prior_cols=list_t_c([constants.NUM_PRIOR_COLS] * len(self.models), ctypes.c_int),
            models=list_t_c([model.id for model in self.models], ctypes.c_int),
            modelPriors=list_t_c([0] * len(self.models), ctypes.c_double),
        )


class DichotomousMAResultStruct(ctypes.Structure):
    _fields_ = [
        ("nmodels", ctypes.c_int),  # number of models for each
        (
            "models",
            ctypes.POINTER(ctypes.POINTER(DichotomousModelResultStruct)),
        ),  # individual model fits for each model average
        ("dist_numE", ctypes.c_int),  # number of entries in rows for the bmd_dist
        ("post_probs", ctypes.POINTER(ctypes.c_double)),  # posterior probabilities
        ("bmd_dist", ctypes.POINTER(ctypes.c_double)),  # bmd ma distribution (dist_numE x 2) matrix
    ]


class DichotomousMAResult(BaseModel):
    results: List[DichotomousModelResultStruct]
    num_models: int
    dist_numE: int

    class Config:
        arbitrary_types_allowed = True

    def to_c(self) -> DichotomousMAAnalysisStruct:
        post_probs = [0] * self.num_models
        bmd_dist = [0] * self.dist_numE
        _results = [ctypes.pointer(struct) for struct in self.results]
        return DichotomousMAResultStruct(
            nmodels=ctypes.c_int(self.num_models),
            models=list_t_c(_results, ctypes.POINTER(DichotomousModelResultStruct)),
            dist_numE=ctypes.c_int(self.dist_numE),
            post_probs=list_t_c(post_probs, ctypes.c_double),
            bmd_dist=list_t_c(bmd_dist, ctypes.c_double),
        )

    def from_c(self, struct: DichotomousMAResultStruct):
        # TODO
        return None
