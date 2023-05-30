import ctypes
from textwrap import dedent
from typing import NamedTuple

import numpy as np

from bmds import bmdscore


def get_version(dll: ctypes.CDLL) -> str:
    """Get the version from the bmds shared object"""
    return bmdscore.version()


# CONTINUOUS MODELS
# -----------------
class ContinuousAnalysisStruct(ctypes.Structure):
    _fields_ = [
        ("model", ctypes.c_int),
        ("n", ctypes.c_int),
        ("suff_stat", ctypes.c_bool),  # true if continuous summary, false if individual data
        ("Y", ctypes.POINTER(ctypes.c_double)),  # observed data means or actual data
        ("doses", ctypes.POINTER(ctypes.c_double)),
        ("sd", ctypes.POINTER(ctypes.c_double)),  # SD of the group if suff_stat = true else null
        ("n_group", ctypes.POINTER(ctypes.c_double)),  # group N if suff_stat = true else null
        ("prior", ctypes.POINTER(ctypes.c_double)),  # column order matrix px5 where p # params
        ("BMD_type", ctypes.c_int),  # type of BMD
        ("isIncreasing", ctypes.c_bool),  # if the BMD is defined increasing or decreasing
        ("BMR", ctypes.c_double),  # benchmark response related to the BMD type
        ("tail_prob", ctypes.c_double),  # tail probability
        ("disttype", ctypes.c_int),  # distribution type defined in the enum distribution
        ("alpha", ctypes.c_double),  # specified alpha
        ("samples", ctypes.c_int),  # number of MCMC samples
        ("degree", ctypes.c_int),
        ("burnin", ctypes.c_int),
        ("parms", ctypes.c_int),  # number of parameters
        ("prior_cols", ctypes.c_int),
        ("transform_dose", ctypes.c_int),
    ]

    def __init__(self, *args, **kwargs):
        self.np_Y = np.array(kwargs.pop("Y"), dtype=np.double)
        self.np_doses = np.array(kwargs.pop("doses"), dtype=np.double)
        self.np_sd = np.array(kwargs.pop("sd"), dtype=np.double)
        self.np_n_group = np.array(kwargs.pop("n_group"), dtype=np.double)
        self.np_prior = kwargs.pop("prior")
        super().__init__(
            *args,
            Y=self.np_Y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            doses=self.np_doses.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            sd=self.np_sd.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            n_group=self.np_n_group.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            prior=self.np_prior.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            **kwargs,
        )

    def __str__(self) -> str:
        txt = dedent(
            f"""
            model: {self.model}
            n: {self.n}
            suff_stat: {self.suff_stat}
            Y: {self.np_Y}
            doses: {self.np_doses}
            sd: {self.np_sd}
            n_group: {self.np_n_group}
            prior<{self.parms, self.prior_cols}>
            <PRIOR>
            priors <unprocessed>: {", ".join(str(v) for v in self.np_prior.tolist())}
            BMD_type: {self.BMD_type}
            isIncreasing: {self.isIncreasing}
            BMR: {self.BMR}
            tail_prob: {self.tail_prob}
            disttype: {self.disttype}
            alpha: {self.alpha}
            samples: {self.samples}
            degree: {self.degree}
            burnin: {self.burnin}
            parms: {self.parms}
            prior_cols: {self.prior_cols}
            transform_dose: {self.transform_dose}
            """
        )
        txt = txt.replace("<PRIOR>", str(self.np_prior.reshape(self.prior_cols, self.parms).T))
        return txt


class ContinuousModelResultStruct(ctypes.Structure):
    _fields_ = [
        ("model", ctypes.c_int),  # continuous model specification
        ("dist", ctypes.c_int),  # distribution type
        ("nparms", ctypes.c_int),  # number of parameters in the model
        ("parms", ctypes.POINTER(ctypes.c_double)),  # parameter estimate
        ("cov", ctypes.POINTER(ctypes.c_double)),  # covariance estimate
        ("max", ctypes.c_double),  # value of the likelihood/posterior at the maximum
        ("dist_numE", ctypes.c_int),  # number of entries in rows for the bmd_dist
        ("model_df", ctypes.c_double),  # Used model degrees of freedom
        ("total_df", ctypes.c_double),  # Total degrees of freedom
        ("bmd", ctypes.c_double),  # The bmd at the maximum
        ("bmd_dist", ctypes.POINTER(ctypes.c_double)),  # bmd distribution (dist_numE x 2) matrix
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # this is modified by Exp3
        self.initial_n = self.nparms
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
            dist: {self.dist}
            nparms: {self.nparms}
            nparms <initial>: {self.initial_n}
            parms: {self.parms[:self.initial_n]}
            cov: {self.cov[:self.initial_n**2]}
            max: {self.max}
            dist_numE: {self.dist_numE}
            model_df: {self.model_df}
            total_df: {self.total_df}
            bmd: {self.bmd}
            bmd_dist: {self.bmd_dist[:self.dist_numE*2]}
            """
        )


class ContinuousGofStruct(ctypes.Structure):
    _fields_ = [
        ("dose", ctypes.POINTER(ctypes.c_double)),
        ("size", ctypes.POINTER(ctypes.c_double)),
        ("estMean", ctypes.POINTER(ctypes.c_double)),
        ("calcMean", ctypes.POINTER(ctypes.c_double)),
        ("obsMean", ctypes.POINTER(ctypes.c_double)),
        ("estSD", ctypes.POINTER(ctypes.c_double)),
        ("calcSD", ctypes.POINTER(ctypes.c_double)),
        ("obsSD", ctypes.POINTER(ctypes.c_double)),
        ("res", ctypes.POINTER(ctypes.c_double)),
        ("n", ctypes.c_int),
        ("ebLower", ctypes.POINTER(ctypes.c_double)),
        ("ebUpper", ctypes.POINTER(ctypes.c_double)),
    ]

    def __str__(self) -> str:
        return dedent(
            f"""
            dose: {self.dose[:self.n]}
            size: {self.size[:self.n]}
            estMean: {self.estMean[:self.n]}
            calcMean: {self.calcMean[:self.n]}
            obsMean: {self.obsMean[:self.n]}
            estSD: {self.estSD[:self.n]}
            calcSD: {self.calcSD[:self.n]}
            obsSD: {self.obsSD[:self.n]}
            res: {self.res[:self.n]}
            n: {self.n}
            ebLower: {self.ebLower[:self.n]}
            ebUpper: {self.ebUpper[:self.n]}
            """
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.np_dose = np.zeros(self.n, dtype=np.float64)
        self.dose = np.ctypeslib.as_ctypes(self.np_dose)
        self.np_size = np.zeros(self.n, dtype=np.float64)
        self.size = np.ctypeslib.as_ctypes(self.np_size)
        self.np_estMean = np.zeros(self.n, dtype=np.float64)
        self.estMean = np.ctypeslib.as_ctypes(self.np_estMean)
        self.np_calcMean = np.zeros(self.n, dtype=np.float64)
        self.calcMean = np.ctypeslib.as_ctypes(self.np_calcMean)
        self.np_obsMean = np.zeros(self.n, dtype=np.float64)
        self.obsMean = np.ctypeslib.as_ctypes(self.np_obsMean)
        self.np_estSD = np.zeros(self.n, dtype=np.float64)
        self.estSD = np.ctypeslib.as_ctypes(self.np_estSD)
        self.np_calcSD = np.zeros(self.n, dtype=np.float64)
        self.calcSD = np.ctypeslib.as_ctypes(self.np_calcSD)
        self.np_obsSD = np.zeros(self.n, dtype=np.float64)
        self.obsSD = np.ctypeslib.as_ctypes(self.np_obsSD)
        self.np_res = np.zeros(self.n, dtype=np.float64)
        self.res = np.ctypeslib.as_ctypes(self.np_res)
        self.np_ebLower = np.zeros(self.n, dtype=np.float64)
        self.ebLower = np.ctypeslib.as_ctypes(self.np_ebLower)
        self.np_ebUpper = np.zeros(self.n, dtype=np.float64)
        self.ebUpper = np.ctypeslib.as_ctypes(self.np_ebUpper)


class ContinuousToiStruct(ctypes.Structure):
    _fields_ = [
        ("llRatio", ctypes.POINTER(ctypes.c_double)),
        ("DF", ctypes.POINTER(ctypes.c_double)),
        ("pVal", ctypes.POINTER(ctypes.c_double)),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.np_llRatio = np.zeros(4, dtype=np.float64)
        self.llRatio = np.ctypeslib.as_ctypes(self.np_llRatio)
        self.np_DF = np.zeros(4, dtype=np.float64)
        self.DF = np.ctypeslib.as_ctypes(self.np_DF)
        self.np_pVal = np.zeros(4, dtype=np.float64)
        self.pVal = np.ctypeslib.as_ctypes(self.np_pVal)


class ContinuousAodStruct(ctypes.Structure):
    _fields_ = [
        ("LL", ctypes.POINTER(ctypes.c_double)),
        ("nParms", ctypes.POINTER(ctypes.c_int)),
        ("AIC", ctypes.POINTER(ctypes.c_double)),
        ("addConst", ctypes.c_double),
        ("TOI", ctypes.POINTER(ContinuousToiStruct)),
    ]

    def __str__(self) -> str:
        toi = self.TOI[0]
        return dedent(
            f"""
            LL: {self.LL[:5]}
            nParms: {self.nParms[:5]}
            AIC: {self.AIC[:5]}
            addConst: {self.addConst}
            llRatio: {toi.llRatio[:4]}
            DF: {toi.DF[:4]}
            pVal: {toi.pVal[:4]}
            """
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.np_LL = np.zeros(5, dtype=np.float64)
        self.LL = np.ctypeslib.as_ctypes(self.np_LL)
        self.np_nParms = np.zeros(5, dtype=np.int32)
        self.nParms = np.ctypeslib.as_ctypes(self.np_nParms)
        self.np_AIC = np.zeros(5, dtype=np.float64)
        self.AIC = np.ctypeslib.as_ctypes(self.np_AIC)
        self.toi_struct = ContinuousToiStruct()
        self.TOI = ctypes.pointer(self.toi_struct)


class ContinuousStructs(NamedTuple):
    analysis: ContinuousAnalysisStruct
    result: ContinuousModelResultStruct
    # used in continuous
    # summary: BmdsResultsStruct
    aod: ContinuousAodStruct
    gof: ContinuousGofStruct

    def __str__(self):
        return dedent(
            f"""
            Analysis:
            {self.analysis}

            Result:
            {self.result}


            AoD:
            {self.aod}

            GoF:
            {self.gof}
            """
        )
