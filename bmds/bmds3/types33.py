import ctypes
import textwrap
from enum import IntEnum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, confloat, conint

from bmds.bmds3.constants import DichotomousModelChoices

from ..datasets import DichotomousDataset
from . import constants


def _list_to_c(list: List[Any], ctype):
    return (ctype * len(list))(*list)


########################
# Dichotomous Structures
########################


class DichotomousRiskType(IntEnum):
    eExtraRisk = 1
    eAddedRisk = 2


class DichotomousModelSettings(BaseModel):
    bmr: confloat(gt=0) = 0.1
    alpha: confloat(gt=0, lt=1) = 0.05
    bmr_type: DichotomousRiskType = DichotomousRiskType.eExtraRisk
    degree: conint(ge=0, le=8) = 0  # multistage only
    samples: conint(ge=10, le=1000) = 100
    burnin: conint(ge=5, le=1000) = 20


class DichotomousAnalysis(BaseModel):
    """
    Purpose - Contains all of the information for a dichotomous analysis.
    It is used do describe a single model analysis, in which all of the
    informationis used, or a MA analysis, in which all the information
    save prior, degree, parms and prior_cols are used.
    """

    model: constants.DichotomousModel
    dataset: DichotomousDataset
    priors: List[constants.Prior]
    BMD_type: int
    BMR: float
    alpha: float
    degree: int
    samples: int
    burnin: int

    class Config:
        arbitrary_types_allowed = True

    @property
    def num_params(self) -> int:
        return (
            self.degree + 1
            if self.model == DichotomousModelChoices.d_multistage.value
            else self.model.num_params
        )

    class Struct(ctypes.Structure):

        _fields_ = [
            ("model", ctypes.c_int),  # Model Type as listed in DichModel
            ("n", ctypes.c_int),  # total number of observations obs/n
            ("Y", ctypes.POINTER(ctypes.c_double)),  # observed +
            ("doses", ctypes.POINTER(ctypes.c_double)),
            ("n_group", ctypes.POINTER(ctypes.c_double)),  # size of the group
            ("prior", ctypes.POINTER(ctypes.c_double)),  # a column order matrix parms X prior_cols
            ("BMD_type", ctypes.c_int),  # 1 = extra ; added otherwise
            ("BMR", ctypes.c_double),
            ("alpha", ctypes.c_double),  # alpha of the analysis
            ("degree", ctypes.c_int),  # degree of polynomial used only multistage
            ("samples", ctypes.c_int),  # number of MCMC samples
            ("burnin", ctypes.c_int),  # size of burnin
            ("parms", ctypes.c_int),  # number of parameters in the model
            ("prior_cols", ctypes.c_int),  # columns in the prior
        ]

        def __str__(self) -> str:
            return textwrap.dedent(
                f"""
                model: {self.model}
                n: {self.n}
                Y: {self.Y[:self.n]}
                doses: {self.doses[:self.n]}
                n_group: {self.n_group[:self.n]}
                prior: {self.prior[:self.parms*self.prior_cols]}
                BMD_type: {self.BMD_type}
                BMR: {self.BMR}
                alpha: {self.alpha}
                degree: {self.degree}
                samples: {self.samples}
                burnin: {self.burnin}
                parms: {self.parms}
                prior_cols: {self.prior_cols}
                """
            )

    def _priors_to_list(self) -> List[float]:
        """
        allocate memory for all parameters and convert to columnwise matrix
        """
        if len(self.priors) == self.num_params:
            # most cases
            arr = np.array([list(prior.dict().values()) for prior in self.priors])
        elif len(self.priors) < self.num_params:
            # special case for multistage; apply all priors; copy last one
            data: List[List[float]] = []
            for prior in self.priors[:-1]:
                data.append(list(prior.dict().values()))
            for _ in range(len(self.priors) - 1, self.num_params):
                data.append(list(self.priors[-1].dict().values()))
            arr = np.array(data)
        else:
            raise ValueError("Unknown state")
        return arr.T.flatten().tolist()

    def to_c(self):
        priors = self._priors_to_list()
        return self.Struct(
            model=ctypes.c_int(self.model.id),
            n=ctypes.c_int(self.dataset.num_dose_groups),
            Y=_list_to_c(self.dataset.incidences, ctypes.c_double),
            doses=_list_to_c(self.dataset.doses, ctypes.c_double),
            n_group=_list_to_c(self.dataset.ns, ctypes.c_double),
            prior=_list_to_c(priors, ctypes.c_double),
            BMD_type=ctypes.c_int(self.BMD_type),
            BMR=ctypes.c_double(self.BMR),
            alpha=ctypes.c_double(self.alpha),
            degree=ctypes.c_int(self.degree),
            samples=ctypes.c_int(self.samples),
            burnin=ctypes.c_int(self.burnin),
            parms=ctypes.c_int(self.num_params),
            prior_cols=ctypes.c_int(constants.NUM_PRIOR_COLS),
        )


class DichotomousModelResult(BaseModel):
    """
    Purpose: Data structure that is populated with all of the necessary
    information for a single model fit.
    """

    model: constants.DichotomousModel
    num_params: int
    dist_numE: int
    params: Optional[Dict[str, float]]
    cov: Optional[np.ndarray]
    max: Optional[float]
    model_df: Optional[float]
    total_df: Optional[float]
    bmd_dist: Optional[np.ndarray]

    class Config:
        arbitrary_types_allowed = True

    class Struct(ctypes.Structure):
        _fields_ = [
            ("model", ctypes.c_int),  # dichotomous model specification
            ("nparms", ctypes.c_int),  # number of parameters in the model
            ("parms", ctypes.POINTER(ctypes.c_double)),  # parameter estimate
            ("cov", ctypes.POINTER(ctypes.c_double)),  # covariance estimate
            ("max", ctypes.c_double),  # value of the likelihood/posterior at the maximum
            ("dist_numE", ctypes.c_int),  # number of entries in rows for the bmd_dist
            ("model_df", ctypes.c_double),  # Used model degrees of freedom
            ("total_df", ctypes.c_double),  # Total degrees of freedom
            (
                "bmd_dist",
                ctypes.POINTER(ctypes.c_double),
            ),  # bmd distribution (dist_numE x 2) matrix
        ]

    def to_c(self):
        parms = np.zeros(self.num_params, dtype=np.float64)
        self.cov = np.zeros(self.num_params ** 2, dtype=np.float64)
        self.bmd_dist = np.zeros(self.dist_numE * 2, dtype=np.float64)
        return self.Struct(
            model=ctypes.c_int(self.model.id),
            nparms=ctypes.c_int(self.num_params),
            parms=parms.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            cov=self.cov.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            dist_numE=ctypes.c_int(self.dist_numE),
            bmd_dist=self.bmd_dist.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

    def from_c(self):
        self.cov = self.cov.reshape(self.num_params, self.num_params)
        self.bmd_dist = self.bmd_dist.reshape(2, self.dist_numE).T

    def bmd_plot(self):
        df = pd.DataFrame(data=self.bmd_dist, columns="bmd quantile".split())
        df = df.query("bmd>0 & bmd < inf")
        df.plot.scatter("bmd", "quantile", xlabel="Dose", ylabel="Propotion")


class DichotomousMAAnalysis(BaseModel):
    """
    Purpose: Fill out all of the information for a dichotomous model average.
    """

    nmodels: int
    priors: List[List[float]]
    nparms: List[int]
    actual_parms: List[int]
    prior_cols: List[int]
    models: List[int]
    modelPriors: List[float]

    class Struct(ctypes.Structure):
        _fields_ = [
            ("nmodels", ctypes.c_int),  # number of models for the model average
            (
                "priors",
                ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
            ),  # list of pointers to prior arrays
            ("nparms", ctypes.POINTER(ctypes.c_int)),  # parameters in each model
            (
                "actual_parms",
                ctypes.POINTER(ctypes.c_int),
            ),  # actual number of parameters in the model
            (
                "prior_cols",
                ctypes.POINTER(ctypes.c_int),
            ),  # columns in the prior if there are "more" in the future
            ("models", ctypes.POINTER(ctypes.c_int)),  # list of models defined by DichModel
            ("modelPriors", ctypes.POINTER(ctypes.c_double)),  # prior probability on the model
        ]

    def to_c(self):
        priors_partial = [_list_to_c(x, ctypes.c_double) for x in self.priors]
        return self.Struct(
            nmodels=ctypes.c_int(self.nmodels),
            priors=_list_to_c(priors_partial, ctypes.POINTER(ctypes.c_double)),
            nparms=_list_to_c(self.nparms, ctypes.c_int),
            actual_parms=_list_to_c(self.actual_parms, ctypes.c_int),
            prior_cols=_list_to_c(self.prior_cols, ctypes.c_int),
            models=_list_to_c(self.models, ctypes.c_int),
            modelPriors=_list_to_c(self.modelPriors, ctypes.c_double),
        )

    @classmethod
    def from_c(cls, struct):
        return cls(
            nmodels=struct.nmodels.value,
            priors=[x[:1] for x in struct.priors[: struct.nmodels.value]],
            nparms=struct.nparms[: struct.nmodels.value],
            actual_parms=struct.actual_parms[: struct.nmodels.value],
            prior_cols=struct.prior_cols[: struct.nmodels.value],
            models=struct.models[: struct.nmodels.value],
            modelPriors=struct.modelPriors[: struct.nmodels.value],
        )


class DichotomousMAResult(BaseModel):
    nmodels: int
    models: List[List[DichotomousModelResult]]
    dist_numE: int
    post_probs: List[float]
    bmd_dist: List[float]

    class Struct(ctypes.Structure):
        _fields_ = [
            ("nmodels", ctypes.c_int),  # number of models for each
            (
                "models",
                ctypes.POINTER(ctypes.POINTER(DichotomousModelResult.Struct)),
            ),  # individual model fits for each model average
            ("dist_numE", ctypes.c_int),  # number of entries in rows for the bmd_dist
            ("post_probs", ctypes.POINTER(ctypes.c_double)),  # posterior probabilities
            (
                "bmd_dist",
                ctypes.POINTER(ctypes.c_double),
            ),  # bmd ma distribution (dist_numE x 2) matrix
        ]

    def to_c(self):
        models_partial = [
            _list_to_c([y.to_c() for y in x], DichotomousModelResult.Struct) for x in self.models
        ]
        return self.Struct(
            nmodels=ctypes.c_int(self.nmodels),
            models=_list_to_c(models_partial, ctypes.POINTER(DichotomousModelResult.Struct),),
            dist_numE=ctypes.c_int(self.dist_numE),
            post_probs=_list_to_c(self.post_probs, ctypes.c_double),
            bmd_dist=_list_to_c(self.bmd_dist, ctypes.c_double),
        )

    @classmethod
    def from_c(cls, struct):
        return cls(
            nmodels=struct.nmodels.value,
            models=[
                [DichotomousModelResult.from_c(y) for y in x[:1]]
                for x in struct.models[: struct.nmodels.value]
            ],
            dist_numE=struct.dist_numE.value,
            post_probs=struct.post_probs[: struct.nmodels.value],
            bmd_dist=struct.bmd_dist[: struct.dis_numE.value ** 2],
        )


#######################
# Continuous Structures
#######################
class ContinuousModelSettings(BaseModel):
    pass


class ContinuousAnalysis(BaseModel):
    model: int
    n: int
    suff_stat: bool
    Y: List[float]
    doses: List[float]
    sd: List[float]
    n_group: List[float]
    prior: List[float]
    BMD_type: int
    isIncreasing: bool
    BMR: float
    tail_prob: float
    disttype: int
    alpha: float
    samples: int
    burnin: int
    parms: int
    prior_cols: int

    class Struct(ctypes.Structure):
        _fields_ = [
            ("model", ctypes.c_int),
            ("n", ctypes.c_int),
            ("suff_stat", ctypes.c_bool),  # true if the data are in sufficient statistics format
            ("Y", ctypes.POINTER(ctypes.c_double)),  # observed data means or actual data
            ("doses", ctypes.POINTER(ctypes.c_double)),
            (
                "sd",
                ctypes.POINTER(ctypes.c_double),
            ),  # SD of the group if suff_stat = true, null otherwise
            (
                "n_group",
                ctypes.POINTER(ctypes.c_double),
            ),  # N for each group if suff_stat = true, null otherwise
            (
                "prior",
                ctypes.POINTER(ctypes.c_double),
            ),  # a column order matrix px5 where p is the number of parameters
            ("BMD_type", ctypes.c_int),  # type of BMD
            ("isIncreasing", ctypes.c_bool),  # if the BMD is defined increasing or decreasing
            ("BMR", ctypes.c_double),  # benchmark response related to the BMD type
            ("tail_prob", ctypes.c_double),  # tail probability
            ("disttype", ctypes.c_int),  # distribution type defined in the enum distribution
            ("alpha", ctypes.c_double),  # specified alpha
            ("samples", ctypes.c_int),  # number of MCMC samples
            ("burnin", ctypes.c_int),  # burn in
            ("parms", ctypes.c_int),  # number of parameters
            ("prior_cols", ctypes.c_int),
        ]

    def to_c(self):
        return self.Struct(
            model=ctypes.c_int(self.model),
            n=ctypes.c_int(self.n),
            suff_stat=ctypes.c_bool(self.suff_stat),
            Y=_list_to_c(self.Y, ctypes.c_double),
            doses=_list_to_c(self.doses, ctypes.c_double),
            sd=_list_to_c(self.sd, ctypes.c_double),
            n_group=_list_to_c(self.n_group, ctypes.c_double),
            prior=_list_to_c(self.prior, ctypes.c_double),
            BMD_type=ctypes.c_int(self.BMD_type),
            isIncreasing=ctypes.c_bool(self.isIncreasing),
            BMR=ctypes.c_double(self.BMR),
            tail_prob=ctypes.c_double(self.tail_prob),
            disttype=ctypes.c_int(self.disttype),
            alpha=ctypes.c_double(self.alpha),
            samples=ctypes.c_int(self.samples),
            burnin=ctypes.c_int(self.burnin),
            parms=ctypes.c_int(self.parms),
            prior_cols=ctypes.c_int(self.prior_cols),
        )

    @classmethod
    def from_c(cls, struct):
        return cls(
            model=struct.model.value,
            n=struct.n.value,
            suff_stat=struct.suff_stat.value,
            Y=struct.Y[: struct.n.value],
            doses=struct.doses[: struct.n.value],
            sd=struct.sd[: struct.n.value],
            n_group=struct.n_group[: struct.n.value],
            prior=struct.prior[: struct.parms.value * struct.prior_cols.value],
            BMD_type=struct.BMD_type.value,
            isIncreasing=struct.isIncreasing.value,
            BMR=struct.BMR.value,
            tail_prob=struct.tail_prob.value,
            disttype=struct.disttype.value,
            alpha=struct.alpha.value,
            samples=struct.samples.value,
            burnin=struct.burnin.value,
            parms=struct.parms.value,
            prior_cols=struct.prior_cols.value,
        )


class ContinuousMAAnalysis(BaseModel):

    nmodels: int
    priors: List[List[float]]
    nparms: List[int]
    actual_parms: List[int]
    prior_cols: List[int]
    models: List[int]
    disttype: List[int]
    modelPriors: List[float]

    class Struct(ctypes.Structure):
        _fields_ = [
            ("nmodels", ctypes.c_int),  # number of models for each
            (
                "priors",
                ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
            ),  # pointer to pointer arrays for the prior
            ("nparms", ctypes.POINTER(ctypes.c_int)),  # parameters in each model
            (
                "actual_parms",
                ctypes.POINTER(ctypes.c_int),
            ),  # actual number of parameters in the model
            (
                "prior_cols",
                ctypes.POINTER(ctypes.c_int),
            ),  # columns in the prior if there are "more" in the future
            ("models", ctypes.POINTER(ctypes.c_int)),  # given model
            ("disttype", ctypes.POINTER(ctypes.c_int)),  # given distribution type
            ("modelPriors", ctypes.POINTER(ctypes.c_double)),  # prior probability on the model
        ]

    def to_c(self):
        priors_partial = [_list_to_c(x, ctypes.c_double) for x in self.priors]
        return self.Struct(
            nmodels=ctypes.c_int(self.nmodels),
            priors=_list_to_c(priors_partial, ctypes.POINTER(ctypes.c_double)),
            nparms=_list_to_c(self.nparms, ctypes.c_int),
            actual_parms=_list_to_c(self.actual_parms, ctypes.c_int),
            prior_cols=_list_to_c(self.prior_cols, ctypes.c_int),
            models=_list_to_c(self.models, ctypes.c_int),
            disttype=_list_to_c(self.disttype, ctypes.c_int),
            modelPriors=_list_to_c(self.modelPriors, ctypes.c_double),
        )

    @classmethod
    def from_c(cls, struct):
        return cls(
            nmodels=struct.nmodels.value,
            priors=[x[:1] for x in struct.priors[: struct.nmodels.value]],
            nparms=struct.nparms[: struct.nmodels.value],
            actual_parms=struct.actual_parms[: struct.nmodels.value],
            prior_cols=struct.prior_cols[: struct.nmodels.value],
            models=struct.models[: struct.nmodels.value],
            disttype=struct.disttype[: struct.nmodels.value],
            modelPriors=struct.modelPriors[: struct.nmodels.value],
        )


class ContinuousModelResult(BaseModel):

    model: int
    dist: int
    nparms: int
    parms: List[float]
    cov: List[float]
    max: float
    dist_numE: int
    bmd_dist: List[float]

    class Struct(ctypes.Structure):
        _fields_ = [
            ("model", ctypes.c_int),  # continuous model specification
            ("dist", ctypes.c_int),  # distribution type
            ("nparms", ctypes.c_int),  # number of parameters in the model
            ("parms", ctypes.POINTER(ctypes.c_double)),  # parameter estimate
            ("cov", ctypes.POINTER(ctypes.c_double)),  # covariance estimate
            ("max", ctypes.c_double),  # value of the likelihood/posterior at the maximum
            ("dist_numE", ctypes.c_int),  # number of entries in rows for the bmd_dist
            (
                "bmd_dist",
                ctypes.POINTER(ctypes.c_double),
            ),  # bmd distribution (dist_numE x 2) matrix
        ]

    def to_c(self):
        return self.Struct(
            model=ctypes.c_int(self.model),
            dist=ctypes.c_int(self.dist),
            nparms=ctypes.c_int(self.nparms),
            parms=_list_to_c(self.parms, ctypes.c_double),
            cov=_list_to_c(self.cov, ctypes.c_double),
            max=ctypes.c_double(self.max),
            dist_numE=ctypes.c_int(self.dist_numE),
            bmd_dist=_list_to_c(self.bmd_dist, ctypes.c_double),
        )

    @classmethod
    def from_c(cls, struct):
        return cls(
            model=struct.model.value,
            dist=struct.dist.value,
            nparms=struct.nparms.value,
            parms=struct.parms[: struct.nparms.value],
            cov=struct.cov[: struct.nparms.value],
            max=struct.max.value,
            dist_numE=struct.dist_numE.value,
            bmd_dist=struct.bmd_dist[: struct.dist_numE ** 2],
        )


class ContinuousMAResult(BaseModel):

    nmodels: int
    models: List[List[ContinuousModelResult]]
    dist_numE: int
    post_probs: List[float]
    bmd_dist: List[float]

    class Struct(ctypes.Structure):
        _fields_ = [
            ("nmodels", ctypes.c_int),  # number of models for each
            ("models", ctypes.POINTER(ctypes.POINTER(ContinuousModelResult.Struct)),),  # priors
            ("dist_numE", ctypes.c_int),  # number of entries in rows for the bmd_dist
            ("post_probs", ctypes.POINTER(ctypes.c_double)),  # posterior probabilities
            (
                "bmd_dist",
                ctypes.POINTER(ctypes.c_double),
            ),  # bmd ma distribution (dist_numE x 2) matrix
        ]

    def to_c(self):
        models_partial = [
            _list_to_c([y.to_c() for y in x], ContinuousModelResult.Struct) for x in self.models
        ]
        return self.Struct(
            nmodels=ctypes.c_int(self.nmodels),
            models=_list_to_c(models_partial, ctypes.POINTER(ContinuousModelResult.Struct),),
            dist_numE=ctypes.c_int(self.dist_numE),
            post_probs=_list_to_c(self.post_probs, ctypes.c_double),
            bmd_dist=_list_to_c(self.bmd_dist, ctypes.c_double),
        )

    @classmethod
    def from_c(cls, struct):
        return cls(
            nmodels=struct.nmodels.value,
            models=[
                [ContinuousModelResult.from_c(y) for y in x[:1]]
                for x in struct.models[: struct.nmodels.value]
            ],
            dist_numE=struct.dist_numE.value,
            post_probs=struct.post_probs[: struct.nmodels.value],
            bmd_dist=struct.bmd_dist[: struct.dis_numE.value ** 2],
        )


#################
# MCMC Structures
#################


class BMDAnalysisMCMC(BaseModel):

    model: int
    burnin: int
    samples: int
    nparms: int
    BMDS: List[float]
    parms: List[float]

    class Struct(ctypes.Structure):
        _fields_ = [
            ("model", ctypes.c_int),  # model used in the analysis
            ("burnin", ctypes.c_uint),  # burnin samples
            ("samples", ctypes.c_uint),  # total samples including burnin
            ("nparms", ctypes.c_uint),  # parameters in the model
            ("BMDS", ctypes.POINTER(ctypes.c_double)),  # array of samples of BMDS length (samples)
            (
                "parms",
                ctypes.POINTER(ctypes.c_double),
            ),  # array of parameters length (samples X parms)
        ]

    def to_c(self):
        return self.Struct(
            model=ctypes.c_int(self.model),
            burnin=ctypes.c_uint(self.burnin),
            samples=ctypes.c_uint(self.samples),
            nparms=ctypes.c_uint(self.nparms),
            BMDS=_list_to_c(self.BMDS, ctypes.c_double),
            parms=_list_to_c(self.parms, ctypes.c_double),
        )

    @classmethod
    def from_c(cls, struct):
        return cls(
            model=struct.model.value,
            burnin=struct.burnin.value,
            samples=struct.samples.value,
            nparms=struct.nparms.value,
            BMDS=struct.BMDS[: struct.samples.value],
            parms=struct.parms[: struct.samples.value * struct.nparms.value],
        )


class MAMCMSFits(BaseModel):

    nfits: int
    analyses: List[List[BMDAnalysisMCMC]]

    class Struct(ctypes.Structure):
        _fields_ = [
            ("nfits", ctypes.c_uint),
            ("analyses", ctypes.POINTER(ctypes.POINTER(BMDAnalysisMCMC.Struct))),
        ]

    def to_c(self):
        analyses_partial = [
            _list_to_c([y.to_c() for y in x], BMDAnalysisMCMC.Struct) for x in self.analyses
        ]
        return self.Struct(
            nfits=ctypes.c_uint(self.nfits),
            analyses=_list_to_c(analyses_partial, ctypes.POINTER(BMDAnalysisMCMC.Struct)),
        )

    @classmethod
    def from_c(cls, struct):
        return cls(
            nfits=struct.nfits.value,
            analyses=[
                [BMDAnalysisMCMC.from_c(y) for y in x[:1]]
                for x in struct.analyses[: struct.nfits.value]
            ],
        )
