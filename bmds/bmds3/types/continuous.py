import ctypes
from enum import IntEnum
from typing import Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ...constants import BOOL_ICON, Dtype
from ...datasets.continuous import ContinuousDatasets
from ...utils import multi_lstrip, pretty_table
from .. import constants
from ..constants import ContinuousModelChoices
from .common import NumpyFloatArray, NumpyIntArray, clean_array, residual_of_interest
from .priors import ModelPriors, PriorClass, PriorType
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
    ContinuousRiskType.AbsoluteDeviation: "{} Absolute Deviation",
    ContinuousRiskType.StandardDeviation: "{} Standard Deviation",
    ContinuousRiskType.RelativeDeviation: "{:.0%} Relative Deviation",
    ContinuousRiskType.PointEstimate: "{} Point Estimation",
    ContinuousRiskType.Extra: "{} Extra",
    ContinuousRiskType.HybridExtra: "{} Hybrid Extra",
    ContinuousRiskType.HybridAdded: "{} Hybrid Added",
}


class ContinuousModelSettings(BaseModel):
    bmr_type: ContinuousRiskType = ContinuousRiskType.StandardDeviation
    is_increasing: bool | None = None  # if None; autodetect used
    bmr: float = 1.0
    tail_prob: float = 0.01
    disttype: constants.DistType = constants.DistType.normal
    alpha: float = 0.05
    samples: int = 0
    degree: int = 0  # polynomial only
    burnin: int = 20
    priors: PriorClass | ModelPriors | None = None  # if None; default used

    @property
    def bmr_text(self) -> str:
        return _bmr_text_map[self.bmr_type].format(self.bmr)

    @property
    def direction(self) -> str:
        return "Up (↑)" if self.is_increasing else "Down (↓)"

    @property
    def confidence_level(self) -> float:
        return 1 - self.alpha

    @property
    def distribution(self) -> str:
        return f"{self.disttype.distribution_type} + {self.disttype.variance_model}"

    def tbl(self, show_degree: bool = True) -> str:
        data = [
            ["BMR", self.bmr_text],
            ["Distribution", self.distribution],
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

    def docx_table_data(self) -> list:
        return [
            ["Setting", "Value"],
            ["BMR", self.bmr_text],
            ["Distribution", self.distribution],
            ["Modeling Direction", self.direction],
            ["Maximum Polynomial Degree", self.degree],
            ["Confidence Level", self.confidence_level],
            ["Tail Probability", self.tail_prob],
        ]

    def update_record(self, d: dict) -> None:
        """Update data record for a tabular-friendly export"""
        d.update(
            bmr=self.bmr_text,
            distribution=self.distribution,
            direction=self.direction,
            confidence_level=self.confidence_level,
            tail_probability=self.tail_prob,
            degree=self.degree,
            model_class=self.priors.prior_class.name,
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
    model_config = ConfigDict(arbitrary_types_allowed=True)

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
        nparms = self.num_params
        inputs = dict(
            BMD_type=ctypes.c_int(self.BMD_type),
            BMR=ctypes.c_double(self.BMR),
            alpha=ctypes.c_double(self.alpha),
            burnin=ctypes.c_int(self.burnin),
            degree=ctypes.c_int(self.degree),
            disttype=ctypes.c_int(self.disttype),
            isIncreasing=ctypes.c_bool(self.is_increasing),
            model=ctypes.c_int(self.model.id),
            parms=ctypes.c_int(nparms),
            prior=self._priors_array(),
            prior_cols=ctypes.c_int(constants.NUM_PRIOR_COLS),
            samples=ctypes.c_int(self.samples),
            tail_prob=ctypes.c_double(self.tail_prob),
            transform_dose=ctypes.c_int(0),
        )

        if self.dataset.dtype == Dtype.CONTINUOUS:
            inputs.update(
                suff_stat=ctypes.c_bool(True),
                n=self.dataset.num_dose_groups,
                doses=self.dataset.doses,
                n_group=self.dataset.ns,
                Y=self.dataset.means,
                sd=self.dataset.stdevs,
            )

        elif self.dataset.dtype == Dtype.CONTINUOUS_INDIVIDUAL:
            inputs.update(
                suff_stat=ctypes.c_bool(False),
                n=len(self.dataset.individual_doses),
                doses=self.dataset.individual_doses,
                n_group=[],
                Y=self.dataset.responses,
                sd=[],
            )
        else:
            raise ValueError(f"Invalid dtype: {self.dataset.dtype}")

        struct = ContinuousAnalysisStruct(**inputs)

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
    bmds_model_df: float = Field(alias="model_df")
    total_df: float
    bmd_dist: NumpyFloatArray

    @classmethod
    def from_model(cls, model) -> Self:
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


class ContinuousParameters(BaseModel):
    names: list[str]
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
    def from_model(cls, model) -> Self:
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

    def tbl(self) -> str:
        headers = "Variable|Estimate|Bounded|Std Error|Lower CI|Upper CI".split("|")
        data = []
        for name, value, bounded, se, lower_ci, upper_ci in zip(
            self.names,
            self.values,
            self.bounded,
            self.se,
            self.lower_ci,
            self.upper_ci,
            strict=True,
        ):
            data.append(
                (
                    name,
                    value,
                    BOOL_ICON[bounded],
                    "NA" if bounded else f"{se:g}",
                    "NA" if bounded else f"{lower_ci:g}",
                    "NA" if bounded else f"{upper_ci:g}",
                )
            )
        return pretty_table(data, headers)

    def covariance_heatmap(self) -> pd.DataFrame:
        df = pd.DataFrame(data=self.cov, columns=self.names, index=self.names)
        df.style.background_gradient(cmap="viridis")
        return df

    def rows(self, extras: dict) -> list[dict]:
        rows = []
        for i in range(len(self.names)):
            rows.append(
                {
                    **extras,
                    **dict(
                        name=self.names[i],
                        value=self.values[i],
                        se=self.se[i],
                        lower_ci=self.lower_ci[i],
                        upper_ci=self.upper_ci[i],
                        bounded=bool(self.bounded[i]),
                        initial_distribution=PriorType(self.prior_type[i]).name,
                        initial_value=self.prior_initial_value[i],
                        initial_stdev=self.prior_stdev[i],
                        initial_min_value=self.prior_min_value[i],
                        initial_max_value=self.prior_max_value[i],
                    ),
                }
            )
        return rows


class ContinuousGof(BaseModel):
    dose: NumpyFloatArray
    size: NumpyFloatArray
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
    def from_model(cls, model) -> Self:
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

    def tbl(self, disttype: constants.DistType) -> str:
        mean_headers = "Dose|Size|Observed Mean|Calculated Mean|Estimated Mean|Scaled Residual"
        sd_headers = "Dose|Size|Observed SD|Calculated SD|Estimated SD"
        if disttype == constants.DistType.log_normal:
            mean_headers = mean_headers.replace("ted Mean", "ted Median")
            sd_headers = sd_headers.replace("ted SD", "ted GSD")
        mean_data = []
        sd_data = []
        for idx in range(len(self.dose)):
            mean_data.append(
                [
                    self.dose[idx],
                    self.size[idx],
                    self.obs_mean[idx],
                    self.calc_mean[idx],
                    self.est_mean[idx],
                    self.residual[idx],
                ]
            )
            sd_data.append(
                [
                    self.dose[idx],
                    self.size[idx],
                    self.obs_sd[idx],
                    self.calc_sd[idx],
                    self.est_sd[idx],
                ]
            )
        return "\n".join(
            [
                pretty_table(mean_data, mean_headers.split("|")),
                pretty_table(sd_data, sd_headers.split("|")),
            ]
        )

    def n(self) -> int:
        return self.dose.size


class ContinuousDeviance(BaseModel):
    names: list[str]
    loglikelihoods: list[float]
    num_params: list[int]
    aics: list[float]

    @classmethod
    def from_model(cls, model) -> Self:
        aod = model.structs.aod
        return cls(
            names=["A1", "A2", "A3", "fitted", "reduced"],
            loglikelihoods=aod.np_LL.tolist(),
            num_params=aod.np_nParms.tolist(),
            aics=aod.np_AIC.tolist(),
        )

    def tbl(self) -> str:
        headers = "Model|Log Likelihood|# Params|AIC".split("|")
        data = []
        for name, loglikelihood, num_param, aic in zip(
            self.names, self.loglikelihoods, self.num_params, self.aics, strict=True
        ):
            data.append([name, loglikelihood, num_param, aic])
        return pretty_table(data, headers)


class ContinuousTests(BaseModel):
    names: list[str]
    ll_ratios: list[float]
    dfs: list[float]
    p_values: list[float]

    @classmethod
    def from_model(cls, model) -> Self:
        tests = model.structs.aod.toi_struct
        return cls(
            names=["Test 1", "Test 2", "Test 3", "Test 4"],
            ll_ratios=tests.np_llRatio.tolist(),
            dfs=tests.np_DF.tolist(),
            p_values=tests.np_pVal.tolist(),
        )

    def tbl(self) -> str:
        headers = "Name|Loglikelihood Ratio|Test DOF|P-Value".split("|")
        data = []
        for name, ll_ratio, df, p_value in zip(
            self.names, self.ll_ratios, self.dfs, self.p_values, strict=True
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
    def from_model(cls, model, params) -> Self:
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
            ["BMD", self.bmd],
            ["BMDL", self.bmdl],
            ["BMDU", self.bmdu],
            ["AIC", self.fit.aic],
            ["Log Likelihood", self.fit.loglikelihood],
            ["P-Value", self.tests.p_values[3]],
            ["Model DOF", self.tests.dfs[3]],
        ]
        return pretty_table(data, "")

    def text(self, dataset: ContinuousDatasets, settings: ContinuousModelSettings) -> str:
        return multi_lstrip(
            f"""
        Summary:
        {self.tbl()}

        Model Parameters:
        {self.parameters.tbl()}

        Goodness of Fit:
        {self.gof.tbl(disttype=settings.disttype)}

        Likelihoods of Interest:
        {self.deviance.tbl()}

        Tests of Interest:
        {self.tests.tbl()}
        """
        )

    @classmethod
    def from_model(cls, model) -> Self:
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
            model_dof=self.tests.dfs[3],
            residual_of_interest=self.gof.roi,
            residual_at_lowest_dose=self.gof.residual[0],
        )
