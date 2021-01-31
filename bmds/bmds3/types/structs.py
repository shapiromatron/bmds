import ctypes
from textwrap import dedent
from typing import NamedTuple

import numpy as np

from .. import constants


class DichotomousAnalysisStruct(ctypes.Structure):

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
        return dedent(
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


class DichotomousModelResultStruct(ctypes.Structure):
    _fields_ = [
        ("model", ctypes.c_int),  # dichotomous model specification
        ("nparms", ctypes.c_int),  # number of parameters in the model
        ("parms", ctypes.POINTER(ctypes.c_double)),  # parameter estimate
        ("cov", ctypes.POINTER(ctypes.c_double)),  # covariance estimate
        ("max", ctypes.c_double),  # value of the likelihood/posterior at the maximum
        ("dist_numE", ctypes.c_int),  # number of entries in rows for the bmd_dist
        ("model_df", ctypes.c_double),  # Used model degrees of freedom
        ("total_df", ctypes.c_double),  # Total degrees of freedom
        ("bmd_dist", ctypes.POINTER(ctypes.c_double),),  # bmd distribution (dist_numE x 2) matrix
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # reference same memory in struct and numpy
        # https://stackoverflow.com/a/23330369/906385
        self.np_parms = np.zeros(kwargs["nparms"], dtype=np.float64)
        self.parms = np.ctypeslib.as_ctypes(self.np_parms)
        self.np_cov = np.zeros(kwargs["nparms"] ** 2, dtype=np.float64)
        self.cov = np.ctypeslib.as_ctypes(self.np_cov)
        self.np_bmd_dist = np.zeros(kwargs["dist_numE"] * 2, dtype=np.float64)
        self.bmd_dist = np.ctypeslib.as_ctypes(self.np_bmd_dist)

    def __str__(self) -> str:
        return dedent(
            f"""
            model: {self.model}
            nparms: {self.nparms}
            parms: {self.parms[:self.nparms]}
            cov: {self.cov[:self.nparms**2]}
            max: {self.max}
            dist_numE: {self.dist_numE}
            model_df: {self.model_df}
            total_df: {self.total_df}
            bmd_dist: {self.bmd_dist[:self.dist_numE*2]}
            """
        )


class DichotomousPgofResultStruct(ctypes.Structure):
    _fields_ = [
        ("n", ctypes.c_int),  # total number of observations obs/n
        ("expected", ctypes.POINTER(ctypes.c_double)),
        ("residual", ctypes.POINTER(ctypes.c_double)),
        ("test_statistic", ctypes.c_double),
        ("p_value", ctypes.c_double),
        ("df", ctypes.c_double),
    ]

    def __str__(self) -> str:
        return dedent(
            f"""
            n: {self.n}
            expected: {self.expected[:self.n]}
            residual: {self.residual[:self.n]}
            test_statistic: {self.test_statistic}
            p_value: {self.p_value}
            df: {self.df}
            """
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.np_expected = np.zeros(self.n, dtype=np.float64)
        self.expected = np.ctypeslib.as_ctypes(self.np_expected)
        self.np_residual = np.zeros(self.n, dtype=np.float64)
        self.residual = np.ctypeslib.as_ctypes(self.np_residual)


class DichotomousBmdsResultsStruct(ctypes.Structure):
    _fields_ = [
        ("bmd", ctypes.c_double),
        ("bmdl", ctypes.c_double),
        ("bmdu", ctypes.c_double),
        ("aic", ctypes.c_double),
        ("bounded", ctypes.POINTER(ctypes.c_bool)),
    ]

    def __str__(self) -> str:
        return dedent(
            f"""
            bmd: {self.bmd}
            bmdl: {self.bmdl}
            bmdu: {self.bmdu}
            aic: {self.aic}
            bounded: <not shown>
            """
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bmd = constants.BMDS_BLANK_VALUE
        self.bmdl = constants.BMDS_BLANK_VALUE
        self.bmdu = constants.BMDS_BLANK_VALUE
        self.aic = constants.BMDS_BLANK_VALUE
        self.np_bounded = np.zeros(kwargs["num_params"], dtype=np.bool_)
        self.bounded = np.ctypeslib.as_ctypes(self.np_bounded)


class DichotomousStructs(NamedTuple):
    analysis: DichotomousAnalysisStruct
    result: DichotomousModelResultStruct
    gof: DichotomousPgofResultStruct
    summary: DichotomousBmdsResultsStruct
