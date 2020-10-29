import ctypes
from enum import IntEnum
from textwrap import dedent
from typing import List

from pydantic import BaseModel

from .. import constants

BMDS_BLANK_VALUE = -9999
NUM_PRIOR_COLS = 5
CDF_TABLE_SIZE = 99
MY_MAX_PARMS = 16
NUM_LIKELIHOODS_OF_INTEREST = 5
NUM_TESTS_OF_INTEREST = 4


class VarType_t(IntEnum):
    eVarTypeNone = 0
    eConstant = 1
    eModeled = 2

    def num_params(self):
        if self.name == "eConstant":
            return 1
        elif self.name == "eModeled":
            return 2
        else:
            raise ValueError(f"Unspecified number of parameters: {self.name}")


class BMDS_C_Options_t(ctypes.Structure):
    _fields_ = [
        ("bmr", ctypes.c_longdouble),
        ("alpha", ctypes.c_longdouble),
        ("background", ctypes.c_longdouble),
        ("tailProb", ctypes.c_longdouble),  # Valid only for hybrid bmr type
        ("bmrType", ctypes.c_int),
        (
            "degree",
            ctypes.c_int,
        ),  # Valid for polynomial type models; for exponential, identifies the submodel
        ("adverseDirection", ctypes.c_int),  # Direction of adversity: 0=auto, 1=up, -1=down
        ("restriction", ctypes.c_int),  # Restriction on parameters for certain models
        ("varType", ctypes.c_int),  # VarType_t
        ("bLognormal", ctypes.c_bool),  # Valid only for continuous models
        ("bUserParmInit", ctypes.c_bool),  # Use specified priors instead of calculated values
    ]


_c_model_class_cw = {
    2: constants.M_ExponentialM2,
    3: constants.M_ExponentialM3,
    4: constants.M_ExponentialM4,
    5: constants.M_ExponentialM5,
    6: constants.M_Hill,
    7: constants.M_Polynomial,
    8: constants.M_Power,
}


class CModelID_t(IntEnum):
    eExp2 = 2
    eExp3 = 3
    eExp4 = 4
    eExp5 = 5
    eHill = 6
    ePoly = 7
    ePow = 8

    def model_class(self) -> str:
        return _c_model_class_cw[self.value]


_d_model_class_cw = {
    1: constants.M_DichotomousHill,
    2: constants.M_Gamma,
    3: constants.M_Logistic,
    4: constants.M_LogLogistic,
    5: constants.M_LogProbit,
    6: constants.M_Multistage,
    7: constants.M_Probit,
    8: constants.M_QuantalLinear,
    9: constants.M_Weibull,
}


class DModelID_t(IntEnum):
    eDHill = 1
    eGamma = 2
    eLogistic = 3
    eLogLogistic = 4
    eLogProbit = 5
    eMultistage = 6
    eProbit = 7
    eQLinear = 8
    eWeibull = 9

    def model_class(self) -> str:
        return _d_model_class_cw[self.value]


class BMDSPrior_t(IntEnum):
    eNone = 0
    eNormal = 1
    eLognormal = 2


class BMRType_t(IntEnum):
    eAbsoluteDev = 1
    eStandardDev = 2
    eRelativeDev = 3
    ePointEstimate = 4
    eExtra = 5  # Not used
    eHybrid_Extra = 6
    eHybrid_Added = 7


class RiskType_t(IntEnum):
    eExtraRisk = 1
    eAddedRisk = 2


class BMDSInputType_t(IntEnum):
    unused = 0
    eCont_2 = 1  # Individual dose-responses
    eCont_4 = 2  # Summarized dose-responses
    eDich_3 = 3  # Regular dichotomous dose-responses
    eDich_4 = 4  # Dichotomous d-r with covariate (e.g., nested)


class cGoFRow_t(ctypes.Structure):
    _fields_ = [
        ("dose", ctypes.c_double),
        ("obsMean", ctypes.c_double),
        ("obsStDev", ctypes.c_double),
        ("calcMedian", ctypes.c_double),
        ("calcGSD", ctypes.c_double),
        ("estMean", ctypes.c_double),
        ("estStDev", ctypes.c_double),
        ("size", ctypes.c_double),
        ("scaledResidual", ctypes.c_double),
        ("ebLower", ctypes.c_double),
        ("ebUpper", ctypes.c_double),
    ]


class GoFRow_t(ctypes.Structure):
    _fields_ = [
        ("dose", ctypes.c_double),
        ("estProb", ctypes.c_double),  # Model-estimated probability for dose
        ("expected", ctypes.c_double),  # Expected dose-response according to the model
        ("observed", ctypes.c_double),
        ("size", ctypes.c_double),
        ("scaledResidual", ctypes.c_double),
        ("ebLower", ctypes.c_double),  # Error bar lower bound
        ("ebUpper", ctypes.c_double),  # Error bar upper bound
    ]


class dGoF_t(ctypes.Structure):
    _fields_ = [
        ("chiSquare", ctypes.c_double),
        ("pvalue", ctypes.c_double),
        ("pzRow", ctypes.POINTER(GoFRow_t)),
        ("df", ctypes.c_int),
        ("n", ctypes.c_int),
    ]


class BMDSInputData_t(ctypes.Structure):
    _fields_ = [
        ("dose", ctypes.c_double),
        ("response", ctypes.c_double),  # Mean value for summary data
        ("groupSize", ctypes.c_double),
        ("col4", ctypes.c_double),  # stddev for cont_4 or covariate for dich_4
    ]


class Prior(ctypes.Structure):
    _fields_ = [
        (
            "type",
            ctypes.c_double,
        ),  # 0= None (frequentist), 1=  normal (Bayesian), 2= log-normal (Bayesian)
        ("initalValue", ctypes.c_double),
        ("stdDev", ctypes.c_double),  # Only used for type= 1 or 2
        ("minValue", ctypes.c_double),
        ("maxValue", ctypes.c_double),
    ]


class BMDS_D_Opts1_t(ctypes.Structure):
    _fields_ = [
        ("bmr", ctypes.c_double),
        ("alpha", ctypes.c_double),
        ("background", ctypes.c_double),
    ]


class BMDS_D_Opts2_t(ctypes.Structure):
    _fields_ = [
        ("bmrType", ctypes.c_int),
        ("degree", ctypes.c_int),  # Polynomial degree for the multistage model
    ]


class DichotomousDeviance_t(ctypes.Structure):
    _fields_ = [
        ("llFull", ctypes.c_double),  # Full model log-likelihood
        ("llReduced", ctypes.c_double),  # Reduced model log-likelihood
        ("devFit", ctypes.c_double),  # Fit model deviance
        ("devReduced", ctypes.c_double),  # Reduced model deviance
        ("pvFit", ctypes.c_double),  # Fit model p-value
        ("pvReduced", ctypes.c_double),  # Reduced model p-value
        ("nparmFull", ctypes.c_int),
        ("nparmFit", ctypes.c_int),
        ("dfFit", ctypes.c_int),
        ("nparmReduced", ctypes.c_int),
        ("dfReduced", ctypes.c_int),
    ]


class BMD_ANAL(ctypes.Structure):
    _fields_ = [
        ("model_id", ctypes.POINTER(ctypes.c_char)),
        ("MAP", ctypes.c_double),  # Equals the -LL for frequentist runs
        ("BMD", ctypes.c_double),
        ("BMDL", ctypes.c_double),
        ("BMDU", ctypes.c_double),
        ("AIC", ctypes.c_double),
        ("BIC_Equiv", ctypes.c_double),  # BIC equivalent for Bayesian runs
        ("PARMS", ctypes.POINTER(ctypes.c_double)),
        (
            "aCDF",
            ctypes.POINTER(ctypes.c_double),
        ),  # Array of cumulative density function values for BMD
        ("deviance", ctypes.POINTER(DichotomousDeviance_t)),
        ("gof", ctypes.POINTER(dGoF_t)),  # Goodness of Fit
        ("boundedParms", ctypes.POINTER(ctypes.c_bool)),
        ("nparms", ctypes.c_int),
        ("nCDF", ctypes.c_int),  # Requested number of aCDF elements to return
    ]


class LLRow_t(ctypes.Structure):
    _fields_ = [
        ("ll", ctypes.c_double),  # Log-likelihood
        ("aic", ctypes.c_double),
        ("model", ctypes.c_int),  # Data model number for test
        ("nParms", ctypes.c_int),  # Count of model parameters
    ]


class TestRow_t(ctypes.Structure):
    _fields_ = [
        ("deviance", ctypes.c_double),  # -2*log-likelihood ratio
        ("pvalue", ctypes.c_double),  # test p-value
        ("testNumber", ctypes.c_int),
        ("df", ctypes.c_int),  # test degrees of freedom
    ]


class ContinuousDeviance_t(ctypes.Structure):
    _fields_ = [
        ("llRows", ctypes.POINTER(LLRow_t)),
        ("testRows", ctypes.POINTER(TestRow_t)),
    ]


class BMD_C_ANAL(ctypes.Structure):
    _fields_ = [
        ("model_id", ctypes.POINTER(ctypes.c_char)),
        ("PARMS", ctypes.POINTER(ctypes.c_double)),
        ("deviance", ContinuousDeviance_t),
        ("gofRow", ctypes.POINTER(cGoFRow_t)),  # Goodness of Fit
        ("boundedParms", ctypes.POINTER(ctypes.c_bool)),
        ("MAP", ctypes.c_double),
        ("BMD", ctypes.c_double),
        ("BMDL", ctypes.c_double),
        ("BMDU", ctypes.c_double),
        ("AIC", ctypes.c_double),
        ("BIC_Equiv", ctypes.c_double),  # BIC equivalent for Bayesian runs
        ("ll_const", ctypes.c_double),  # LL "additive" constant term
        (
            "aCDF",
            ctypes.POINTER(ctypes.c_double),
        ),  # Array of cumulative density function values for BMD
        ("nCDF", ctypes.c_int),  # Requested number of aCDF elements to return
        ("nparms", ctypes.c_int),
        ("bAdverseUp", ctypes.c_bool),
    ]

    def __str__(self):
        return dedent(
            f"""\
        model_id: {self.model_id[0].decode('utf8')} nparms: {self.nparms}
        BMDL: {self.BMDL:.3f}  BMD: {self.BMD:.3f} BMDU: {self.BMDU:.3f}
        AIC: {self.AIC:.3f} BIC: {self.BIC_Equiv:.3f}
        """
        )


class PRIOR(ctypes.Structure):
    _fields_ = [
        (
            "type",
            ctypes.c_double,
        ),  # 0= None (frequentist), 1=  normal (Bayesian), 2= log-normal (Bayesian)
        ("initialValue", ctypes.c_double),
        ("stdDev", ctypes.c_double),  # Only used for type= 1 or 2
        ("minValue", ctypes.c_double),
        ("maxValue", ctypes.c_double),
    ]

    def __repr__(self):
        return f"Prior({self.type}, {self.initialValue}, {self.stdDev}, {self.minValue}, {self.maxValue})"


class DichotomousResultDeviance(BaseModel):
    ll_full: float
    ll_reduced: float
    dev_fit: float
    dev_reduced: float
    pv_fit: float
    pv_reduced: float
    n_parm_full: int
    n_parm_fit: int
    df_fit: int
    n_parm_reduced: int
    df_reduced: int


class DichotomousResultGofRow(BaseModel):
    dose: float
    est_prob: float
    expected: float
    observed: float
    size: float
    scaled_residual: float
    eb_lower: float
    eb_upper: float


class DichotomousResultGof(BaseModel):
    chi_square: float
    p_value: float
    rows: List[DichotomousResultGofRow]
    df: int
    n: int


class DichotomousResult(BaseModel):
    response_code: int
    map: float
    bmd: float
    bmdl: float
    bmdu: float
    aic: float
    bic: float
    parameters: List[float]
    bounded_parameters: bool
    cdf: List[float]
    gof: DichotomousResultGof
    deviances: List[DichotomousResultDeviance]

    @classmethod
    def from_execution(
        cls, response_code: int, result: BMD_ANAL, num_dose_groups: int, num_parameters: int
    ) -> "DichotomousResult":
        """
        Create a new response object from the bmd analysis C type `BMD_ANAL`. class.

        Args:
            response_code (int): the dll execution response code
            result (types.BMD_ANAL): the result struct from execution
            num_dose_groups (int): number of dose groups
            num_parameters (int): number of model parameters
        """
        assert response_code == 0

        return cls(
            response_code=response_code,
            map=result.MAP,
            bmd=result.BMD,
            bmdl=result.BMDL,
            bmdu=result.BMDU,
            aic=result.AIC,
            bic=result.BIC_Equiv,
            parameters=[result.PARMS[i] for i in range(num_parameters)],
            bounded_parameters=bool(result.boundedParms.contents),
            cdf=result.aCDF[: result.nCDF],
            gof=DichotomousResultGof(
                chi_square=result.gof[0].chiSquare,
                p_value=result.gof[0].pvalue,
                rows=[
                    DichotomousResultGofRow(
                        dose=result.gof[0].pzRow[i].dose,
                        est_prob=result.gof[0].pzRow[i].estProb,
                        expected=result.gof[0].pzRow[i].expected,
                        observed=result.gof[0].pzRow[i].observed,
                        size=result.gof[0].pzRow[i].size,
                        scaled_residual=result.gof[0].pzRow[i].scaledResidual,
                        eb_lower=result.gof[0].pzRow[i].ebLower,
                        eb_upper=result.gof[0].pzRow[i].ebUpper,
                    )
                    for i in range(num_dose_groups)
                ],
                df=result.gof[0].df,
                n=result.gof[0].n,
            ),
            deviances=[
                DichotomousResultDeviance(
                    ll_full=result.deviance[i].llFull,
                    ll_reduced=result.deviance[i].llReduced,
                    dev_fit=result.deviance[i].devFit,
                    dev_reduced=result.deviance[i].devReduced,
                    pv_fit=result.deviance[i].pvFit,
                    pv_reduced=result.deviance[i].pvReduced,
                    n_parm_full=result.deviance[i].nparmFull,
                    n_parm_fit=result.deviance[i].nparmFit,
                    df_fit=result.deviance[i].dfFit,
                    n_parm_reduced=result.deviance[i].nparmReduced,
                    df_reduced=result.deviance[i].dfReduced,
                )
                for i in range(num_dose_groups)
            ],
        )


class ContinuousResultGoodnessOfFit(BaseModel):
    dose: float
    obs_mean: float
    obs_stdev: float
    calc_median: float
    calc_gsd: float
    est_mean: float
    est_stdev: float
    size: float
    scaled_residual: float
    eb_lower: float
    eb_upper: float


class ContinuousResultLoglikelihood(BaseModel):
    loglikelihood: float
    aic: float
    model: int
    n_parms: int


class ContinuousResultTestRow(BaseModel):
    deviance: float
    p_value: float
    test_number: int
    df: int


class ContinuousResult(BaseModel):
    response_code: int
    map: float
    bmd: float
    bmdl: float
    bmdu: float
    aic: float
    bic: float
    ll_const: float
    b_adverse_up: bool
    cdf: List[float]
    parameters: List[float]
    bounded_parameters: bool
    gof: List[ContinuousResultGoodnessOfFit]
    loglikelihoods: List[ContinuousResultLoglikelihood]
    test_rows: List[ContinuousResultTestRow]

    @classmethod
    def from_execution(
        cls, response_code: int, result: BMD_C_ANAL, num_dose_groups: int, num_parameters: int
    ) -> "ContinuousResult":
        """
        Create a new response object from the bmd analysis C type `BMD_C_ANAL`. class.

        Args:
            response_code (int): the dll execution response code
            result (types.BMD_C_ANAL): the result struct from execution
            num_dose_groups (int): number of dose groups
            num_parameters (int): number of model parameters
        """
        assert response_code == 0

        return cls(
            response_code=response_code,
            map=result.MAP,
            bmd=result.BMD,
            bmdl=result.BMDL,
            bmdu=result.BMDU,
            aic=result.AIC,
            bic=result.BIC_Equiv,
            ll_const=result.ll_const,
            b_adverse_up=result.bAdverseUp,
            cdf=result.aCDF[: result.nCDF],
            parameters=[result.PARMS[i] for i in range(num_parameters)],
            bounded_parameters=bool(result.boundedParms.contents),
            gof=[
                ContinuousResultGoodnessOfFit(
                    dose=result.gofRow[i].dose,
                    obs_mean=result.gofRow[i].obsMean,
                    obs_stdev=result.gofRow[i].obsStDev,
                    calc_median=result.gofRow[i].calcMedian,
                    calc_gsd=result.gofRow[i].calcGSD,
                    est_mean=result.gofRow[i].estMean,
                    est_stdev=result.gofRow[i].estStDev,
                    size=result.gofRow[i].size,
                    scaled_residual=result.gofRow[i].scaledResidual,
                    eb_lower=result.gofRow[i].ebLower,
                    eb_upper=result.gofRow[i].ebUpper,
                )
                for i in range(num_dose_groups)
            ],
            loglikelihoods=[
                ContinuousResultLoglikelihood(
                    loglikelihood=result.deviance.llRows[i].ll,
                    aic=result.deviance.llRows[i].aic,
                    model=result.deviance.llRows[i].model,
                    n_parms=result.deviance.llRows[i].nParms,
                )
                for i in range(NUM_LIKELIHOODS_OF_INTEREST)
            ],
            test_rows=[
                ContinuousResultTestRow(
                    deviance=result.deviance.testRows[i].deviance,
                    p_value=result.deviance.testRows[i].pvalue,
                    test_number=result.deviance.testRows[i].testNumber,
                    df=result.deviance.testRows[i].df,
                )
                for i in range(NUM_TESTS_OF_INTEREST)
            ],
        )
