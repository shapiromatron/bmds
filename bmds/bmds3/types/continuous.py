import ctypes
from enum import IntEnum
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel

from bmds.bmds3.constants import ContinuousModelChoices
from bmds.datasets.continuous import ContinuousDatasets

from ...constants import BOOL_ICON, Dtype
from ...utils import multi_lstrip, pretty_table
from .. import constants
from .common import NumpyFloatArray, NumpyIntArray, clean_array, list_t_c, residual_of_interest
from .priors import ModelPriors, PriorClass
from .structs import (
    BmdsResultsStruct,
    ContinuousAnalysisStruct,
    ContinuousAodStruct,
    ContinuousGofStruct,
    ContinuousModelResultStruct,
    ContinuousStructs,
)


class ContinuousRiskType(IntEnum):
    AbsoluteDeviation = 1
    StandardDeviation = 2
    RelativeDeviation = 3
    PointEstimate = 4
    Extra = 5  # Not used
    HybridExtra = 6
    HybridAdded = 7


_bmr_text_map = {
    ContinuousRiskType.AbsoluteDeviation: "{} absolute deviation",
    ContinuousRiskType.StandardDeviation: "{} standard deviation",
    ContinuousRiskType.RelativeDeviation: "{:.0%} relative deviation",
    ContinuousRiskType.PointEstimate: "{} point estimation",
    ContinuousRiskType.Extra: "{} extra",
    ContinuousRiskType.HybridExtra: "{} hybrid extra",
    ContinuousRiskType.HybridAdded: "{} hybrid added",
}


class ContinuousModelSettings(BaseModel):
    bmr_type: ContinuousRiskType = ContinuousRiskType.StandardDeviation
    is_increasing: Optional[bool]  # if None; autodetect used
    bmr: float = 1.0
    tail_prob: float = 0.01
    disttype: constants.DistType = constants.DistType.normal
    alpha: float = 0.05
    samples: int = 0
    degree: int = 0  # polynomial only
    burnin: int = 20
    priors: Union[None, PriorClass, ModelPriors]  # if None; default used

    @property
    def bmr_text(self) -> str:
        return _bmr_text_map[self.bmr_type].format(self.bmr)

    @property
    def direction(self) -> str:
        return "Up (↑)" if self.is_increasing else "Down (↓)"

    @property
    def confidence_level(self) -> float:
        return 1 - self.alpha

    def tbl(self, show_degree: bool = True) -> str:
        data = [
            ["BMR", self.bmr_text],
            ["Distribution", f"{self.disttype.distribution_type} + {self.disttype.variance_model}"],
            ["Modeling Direction", self.direction],
            ["Confidence Level", self.confidence_level],
            ["Tail Probability", self.tail_prob],
            ["Modeling Approach", self.priors.prior_class.name],
        ]

        if show_degree:
            data.append(["Degree", self.degree])

        if self.priors.is_bayesian:
            data.extend((["Samples", self.samples], ["Burn-in", self.burnin]))

        return pretty_table(data, "")

    def update_record(self, d: dict) -> None:
        """Update data record for a tabular-friendly export"""
        d.update(
            is_increasing=self.is_increasing,
            bmr=self.bmr,
            bmr_type=self.bmr_type.name,
            alpha=self.alpha,
            tail_prop=self.tail_prob,
            degree=self.degree,
            dist_type=self.disttype.name,
            prior_class=self.priors.prior_class.name,
        )


class ContinuousAnalysis(BaseModel):
    model: constants.ContinuousModel
    dataset: ContinuousDatasets
    priors: ModelPriors
    BMD_type: ContinuousRiskType
    is_increasing: bool
    BMR: float
    tail_prob: float
    disttype: constants.DistType
    alpha: float
    samples: int
    burnin: int
    degree: int

    class Config:
        arbitrary_types_allowed = True

    @property
    def num_params(self) -> int:
        if self.model == ContinuousModelChoices.c_polynomial.value:
            params = self.degree + 1
        else:
            params = len(self.model.params)

        if self.disttype is constants.DistType.normal_ncv:
            params += 2
        else:
            params += 1

        return params

    def _priors_array(self) -> np.ndarray:
        degree = self.degree if self.model.id == constants.ContinuousModelIds.c_polynomial else None
        return self.priors.to_c(degree=degree, dist_type=self.disttype)

    def to_c(self) -> ContinuousStructs:
        priors = self._priors_array()
        priors_pointer = np.ctypeslib.as_ctypes(priors)
        nparms = self.num_params

        struct = ContinuousAnalysisStruct(
            BMD_type=ctypes.c_int(self.BMD_type),
            BMR=ctypes.c_double(self.BMR),
            alpha=ctypes.c_double(self.alpha),
            burnin=ctypes.c_int(self.burnin),
            degree=ctypes.c_int(self.degree),
            disttype=ctypes.c_int(self.disttype),
            isIncreasing=ctypes.c_bool(self.is_increasing),
            model=ctypes.c_int(self.model.id),
            parms=ctypes.c_int(nparms),
            prior=priors_pointer,
            prior_cols=ctypes.c_int(constants.NUM_PRIOR_COLS),
            samples=ctypes.c_int(self.samples),
            tail_prob=ctypes.c_double(self.tail_prob),
        )

        if self.dataset.dtype == Dtype.CONTINUOUS:
            struct.suff_stat = ctypes.c_bool(True)
            struct.Y = list_t_c(self.dataset.means, ctypes.c_double)
            struct.doses = list_t_c(self.dataset.doses, ctypes.c_double)
            struct.n = ctypes.c_int(self.dataset.num_dose_groups)
            struct.n_group = list_t_c(self.dataset.ns, ctypes.c_double)
            struct.sd = list_t_c(self.dataset.stdevs, ctypes.c_double)
        elif self.dataset.dtype == Dtype.CONTINUOUS_INDIVIDUAL:
            struct.suff_stat = ctypes.c_bool(False)
            struct.Y = list_t_c(self.dataset.responses, ctypes.c_double)
            struct.doses = list_t_c(self.dataset.individual_doses, ctypes.c_double)
            struct.n = ctypes.c_int(len(self.dataset.individual_doses))
            struct.n_group = list_t_c([], ctypes.c_double)
            struct.sd = list_t_c([], ctypes.c_double)
        else:
            raise ValueError(f"Invalid dtype: {self.dataset.dtype}")

        return ContinuousStructs(
            analysis=struct,
            result=ContinuousModelResultStruct(nparms=nparms, dist_numE=constants.N_BMD_DIST),
            summary=BmdsResultsStruct(num_params=nparms),
            aod=ContinuousAodStruct(),
            gof=ContinuousGofStruct(n=struct.n),
        )


class ContinuousModelResult(BaseModel):
    dist: int
    loglikelihood: float
    aic: float
    bic_equiv: float
    chisq: float
    model_df: float
    total_df: float
    bmd_dist: NumpyFloatArray

    @classmethod
    def from_model(cls, model) -> "ContinuousModelResult":
        summary = model.structs.summary
        result = model.structs.result
        arr = result.np_bmd_dist.reshape(2, result.dist_numE)
        arr = arr[:, np.isfinite(arr[0, :])]
        arr = arr[:, arr[0, :] > 0]

        return ContinuousModelResult(
            dist=result.dist,
            loglikelihood=result.max,
            aic=summary.aic,
            bic_equiv=summary.BIC_equiv,
            chisq=summary.chisq,
            model_df=result.model_df,
            total_df=result.total_df,
            bmd_dist=arr,
        )

    def dict(self, **kw) -> Dict:
        d = super().dict(**kw)
        return NumpyFloatArray.listify(d)


class ContinuousParameters(BaseModel):
    names: List[str]
    values: NumpyFloatArray
    se: NumpyFloatArray
    lower_ci: NumpyFloatArray
    upper_ci: NumpyFloatArray
    bounded: NumpyFloatArray
    cov: NumpyFloatArray
    prior_type: NumpyIntArray
    prior_initial_value: NumpyFloatArray
    prior_stdev: NumpyFloatArray
    prior_min_value: NumpyFloatArray
    prior_max_value: NumpyFloatArray

    @classmethod
    def get_priors(cls, model) -> np.ndarray:
        priors_list = model.get_priors_list()
        return np.array(priors_list, dtype=np.float64).T

    @classmethod
    def from_model(cls, model) -> "ContinuousParameters":
        result = model.structs.result
        summary = model.structs.summary
        param_names = model.get_param_names()
        priors = cls.get_priors(model)

        cov_n = result.initial_n
        cov = result.np_cov.reshape(result.initial_n, result.initial_n)
        slice = None

        # DLL deletes the c parameter and shifts items down; correct in outputs here
        if model.bmd_model_class.id == constants.ContinuousModelIds.c_exp_m3:

            # do the same for parameter names for consistency
            c_index = param_names.index("c")
            param_names.pop(c_index)

            # shift priors as well
            priors = priors.T
            priors[c_index:-1] = priors[c_index + 1 :]
            priors = priors[:-1].T

            # reshape covariance
            cov_n = result.initial_n - 1
            cov = result.np_cov[: cov_n * cov_n].reshape(cov_n, cov_n)

            # change slice for other variables
            slice = -1

        return cls(
            names=param_names,
            values=result.np_parms[:slice],
            bounded=summary.np_bounded[:slice],
            se=summary.np_stdErr[:slice],
            lower_ci=summary.np_lowerConf[:slice],
            upper_ci=summary.np_upperConf[:slice],
            cov=cov,
            prior_type=priors[0],
            prior_initial_value=priors[1],
            prior_stdev=priors[2],
            prior_min_value=priors[3],
            prior_max_value=priors[4],
        )

    def dict(self, **kw) -> Dict:
        d = super().dict(**kw)
        return NumpyFloatArray.listify(d)

    def tbl(self) -> str:
        headers = "parm|type|initial|stdev|min|max|estimate|lower|upper|bounded".split("|")
        data = []
        for name, type, initial, stdev, min, max, value, lower, upper, bounded in zip(
            self.names,
            self.prior_type,
            self.prior_initial_value,
            self.prior_stdev,
            self.prior_min_value,
            self.prior_max_value,
            self.values,
            self.lower_ci,
            self.upper_ci,
            self.bounded,
        ):
            data.append(
                [name, type, initial, stdev, min, max, value, lower, upper, BOOL_ICON[bounded]]
            )
        return pretty_table(data, headers)

    def covariance_heatmap(self) -> pd.DataFrame:
        df = pd.DataFrame(data=self.cov, columns=self.names, index=self.names)
        df.style.background_gradient(cmap="viridis")
        return df


class ContinuousGof(BaseModel):
    dose: NumpyFloatArray
    size: NumpyIntArray
    est_mean: NumpyFloatArray
    calc_mean: NumpyFloatArray
    obs_mean: NumpyFloatArray
    est_sd: NumpyFloatArray
    calc_sd: NumpyFloatArray
    obs_sd: NumpyFloatArray
    residual: NumpyFloatArray
    eb_lower: NumpyFloatArray
    eb_upper: NumpyFloatArray
    roi: float

    @classmethod
    def from_model(cls, model) -> "ContinuousGof":
        gof = model.structs.gof
        # only keep indexes where the num ob obsMean + obsSD == 0;
        # needed for continuous individual datasets where individual items are collapsed into groups
        mask = np.flatnonzero(np.vstack([gof.np_obsMean, gof.np_obsSD]).sum(axis=0))
        return ContinuousGof(
            dose=gof.np_dose[mask],
            size=gof.np_size[mask],
            est_mean=gof.np_estMean[mask],
            calc_mean=gof.np_calcMean[mask],
            obs_mean=gof.np_obsMean[mask],
            est_sd=gof.np_estSD[mask],
            calc_sd=gof.np_calcSD[mask],
            obs_sd=gof.np_obsSD[mask],
            residual=gof.np_res[mask],
            eb_lower=gof.np_ebLower[mask],
            eb_upper=gof.np_ebUpper[mask],
            roi=residual_of_interest(
                model.structs.summary.bmd, model.dataset.doses, gof.np_res[mask].tolist()
            ),
        )

    def dict(self, **kw) -> Dict:
        d = super().dict(**kw)
        return NumpyFloatArray.listify(d)

    def tbl(self) -> str:
        headers = "Dose|EstMean|CalcMean|ObsMean|EstStdev|CalcStdev|ObsStdev|Residual".split("|")
        data = []
        for idx in range(len(self.dose)):
            data.append(
                [
                    self.dose[idx],
                    self.est_mean[idx],
                    self.calc_mean[idx],
                    self.obs_mean[idx],
                    self.est_sd[idx],
                    self.calc_sd[idx],
                    self.obs_sd[idx],
                    self.residual[idx],
                ]
            )
        return pretty_table(data, headers)

    def n(self) -> int:
        return self.dose.size


class ContinuousDeviance(BaseModel):
    names: List[str]
    loglikelihoods: List[float]
    num_params: List[int]
    aics: List[float]

    @classmethod
    def from_model(cls, model) -> "ContinuousDeviance":
        aod = model.structs.aod
        return cls(
            names=["A1", "A2", "A3", "fitted", "reduced"],
            loglikelihoods=aod.np_LL.tolist(),
            num_params=aod.np_nParms.tolist(),
            aics=aod.np_AIC.tolist(),
        )

    def tbl(self) -> str:
        headers = "Name|Loglikelihood|num params|AIC".split("|")
        data = []
        for (name, loglikelihood, num_param, aic) in zip(
            self.names, self.loglikelihoods, self.num_params, self.aics
        ):
            data.append([name, loglikelihood, num_param, aic])
        return pretty_table(data, headers)


class ContinuousTests(BaseModel):
    names: List[str]
    ll_ratios: List[float]
    dfs: List[float]
    p_values: List[float]

    @classmethod
    def from_model(cls, model) -> "ContinuousTests":
        tests = model.structs.aod.toi_struct
        return cls(
            names=["p_test1", "p_test2", "p_test3", "p_test4"],
            ll_ratios=tests.np_llRatio.tolist(),
            dfs=tests.np_DF.tolist(),
            p_values=tests.np_pVal.tolist(),
        )

    def tbl(self) -> str:
        headers = "Name|Loglikelihood Ratio|DF|p_value".split("|")
        data = []
        for (name, ll_ratio, df, p_value) in zip(
            self.names, self.ll_ratios, self.dfs, self.p_values
        ):
            data.append([name, ll_ratio, df, p_value])
        return pretty_table(data, headers)


class ContinuousPlotting(BaseModel):
    dr_x: NumpyFloatArray
    dr_y: NumpyFloatArray
    bmdl_y: float
    bmd_y: float
    bmdu_y: float

    @classmethod
    def from_model(cls, model, params) -> "ContinuousPlotting":
        summary = model.structs.summary
        xs = np.array([summary.bmdl, summary.bmd, summary.bmdu])
        dr_x = model.dataset.dose_linspace
        dr_y = clean_array(model.dr_curve(dr_x, params))
        critical_ys = clean_array(model.dr_curve(xs, params))
        critical_ys[critical_ys <= 0] = constants.BMDS_BLANK_VALUE
        return cls(
            dr_x=dr_x,
            dr_y=dr_y,
            bmdl_y=critical_ys[0],
            bmd_y=critical_ys[1],
            bmdu_y=critical_ys[2],
        )

    def dict(self, **kw) -> Dict:
        d = super().dict(**kw)
        return NumpyFloatArray.listify(d)


class ContinuousResult(BaseModel):
    bmdl: float
    bmd: float
    bmdu: float
    has_completed: bool
    fit: ContinuousModelResult
    gof: ContinuousGof
    parameters: ContinuousParameters
    deviance: ContinuousDeviance
    tests: ContinuousTests
    plotting: ContinuousPlotting

    def tbl(self) -> str:
        data = [
            ["BMDL", self.bmdl],
            ["BMD", self.bmd],
            ["BMDU", self.bmdu],
            ["AIC", self.fit.aic],
            ["LL", self.fit.loglikelihood],
            ["model_df", self.fit.model_df],
        ]
        return pretty_table(data, "")

    def text(self, dataset: ContinuousDatasets) -> str:
        return multi_lstrip(
            f"""
        Summary:
        {self.tbl()}

        Goodness of fit:
        {self.gof.tbl()}

        Parameters:
        {self.parameters.tbl()}

        Deviances:
        {self.deviance.tbl()}

        Tests:
        {self.tests.tbl()}
        """
        )

    @classmethod
    def from_model(cls, model) -> "ContinuousResult":
        summary = model.structs.summary
        params = ContinuousParameters.from_model(model)
        return cls(
            bmdl=summary.bmdl,
            bmd=summary.bmd,
            bmdu=summary.bmdu,
            has_completed=summary.validResult,
            fit=ContinuousModelResult.from_model(model),
            gof=ContinuousGof.from_model(model),
            parameters=params,
            deviance=ContinuousDeviance.from_model(model),
            tests=ContinuousTests.from_model(model),
            plotting=ContinuousPlotting.from_model(model, params.values),
        )

    def update_record(self, d: dict) -> None:
        """Update data record for a tabular-friendly export"""
        d.update(
            bmdl=self.bmdl,
            bmd=self.bmd,
            bmdu=self.bmdu,
            aic=self.fit.aic,
            loglikelihood=self.fit.loglikelihood,
            p_value1=self.tests.p_values[0],
            p_value2=self.tests.p_values[1],
            p_value3=self.tests.p_values[2],
            p_value4=self.tests.p_values[3],
            model_df=self.fit.model_df,
            total_df=self.fit.total_df,
            chi_squared=self.fit.chisq,
            residual_of_interest=self.gof.roi,
            residual_at_lowest_dose=self.gof.residual[0],
        )
