import math
from typing import Any, Optional, Union

from pydantic import BaseModel

from ... import constants
from ...utils import ff
from ..constants import BMDS_BLANK_VALUE, DistType
from .constants import RuleClass

Number = Union[float, int]


class CheckResponse(BaseModel):
    logic_bin: constants.LogicBin
    message: str

    @classmethod
    def success(cls) -> "CheckResponse":
        return CheckResponse(logic_bin=constants.LogicBin.NO_CHANGE, message="")


class Check:

    _enabled_attribute = {
        constants.Dtype.DICHOTOMOUS: "enabled_dichotomous",
        constants.Dtype.DICHOTOMOUS_CANCER: "enabled_dichotomous",
        constants.Dtype.CONTINUOUS: "enabled_continuous",
        constants.Dtype.CONTINUOUS_INDIVIDUAL: "enabled_continuous",
    }

    @classmethod
    def check(cls, dataset, model, rule_settings) -> CheckResponse:
        is_enabled = getattr(rule_settings, cls._enabled_attribute[dataset.dtype])
        if not is_enabled:
            return CheckResponse.success()
        return cls.apply_rule(dataset, model, rule_settings)

    @classmethod
    def apply_rule(cls, dataset, model, rule_settings) -> CheckResponse:
        ...


def is_valid_number(value: Any) -> bool:
    try:
        return math.isfinite(value) and value != BMDS_BLANK_VALUE
    except TypeError:
        return False


def number_or_none(value: Any) -> Optional[Number]:
    if not is_valid_number(value):
        return None
    return value


def pass_if_gte(value: Any, threshold: float) -> bool:
    if not is_valid_number(value):
        return True  # if value is invalid; other checks will report error
    return value >= threshold


def pass_if_lte(value: Any, threshold: float) -> bool:
    if not is_valid_number(value):
        return True  # if value is invalid; other checks will report error
    return value <= threshold


# existence checks
# --------------------------------------------------------------------------------------------------
class ExistenceCheck(Check):
    failure_message_name: str

    @classmethod
    def get_value(cls, dataset, model) -> Optional[Number]:
        ...

    @classmethod
    def apply_rule(cls, dataset, model, rule_settings) -> CheckResponse:
        value = cls.get_value(dataset, model)
        if is_valid_number(value):
            return CheckResponse.success()
        return CheckResponse(
            logic_bin=rule_settings.failure_bin,
            message=f"{cls.failure_message_name} does not exist",
        )


class AicExists(ExistenceCheck):
    failure_message_name = "AIC"

    @classmethod
    def get_value(cls, dataset, model) -> Optional[Number]:
        return model.results.fit.aic


class BmdExists(ExistenceCheck):
    failure_message_name = "BMD"

    @classmethod
    def get_value(cls, dataset, model) -> Optional[Number]:
        return model.results.bmd


class RoiExists(ExistenceCheck):
    failure_message_name = "Residual of interest"

    @classmethod
    def get_value(cls, dataset, model) -> Optional[Number]:
        return model.results.gof.roi


class BmdlExists(ExistenceCheck):
    failure_message_name = "BMDL"

    @classmethod
    def get_value(cls, dataset, model) -> Optional[Number]:
        return model.results.bmdl


class BmduExists(ExistenceCheck):
    failure_message_name = "BMDU"

    @classmethod
    def get_value(cls, dataset, model) -> Optional[Number]:
        return model.results.bmdu


# greater than threshold checks
# --------------------------------------------------------------------------------------------------
class ShouldBeGreaterThan(Check):
    failure_message_name: str

    @classmethod
    def get_value(cls, dataset, model) -> Optional[Number]:
        ...

    @classmethod
    def apply_rule(cls, dataset, model, rule_settings) -> CheckResponse:
        value = cls.get_value(dataset, model)
        threshold = rule_settings.threshold
        if pass_if_gte(value, threshold):
            return CheckResponse.success()
        return CheckResponse(
            logic_bin=rule_settings.failure_bin,
            message=f"{cls.failure_message_name} less than threshold ({ff(value)} < {threshold})",
        )


class GoodnessOfFit(ShouldBeGreaterThan):
    failure_message_name = "Goodness of fit p-value"

    @classmethod
    def get_value(cls, dataset, model) -> Optional[Number]:
        if dataset.dtype in constants.DICHOTOMOUS_DTYPES:
            return number_or_none(model.results.gof.p_value)
        elif dataset.dtype in constants.CONTINUOUS_DTYPES:
            return 0  # TODO - fix


class GoodnessOfFitCancer(GoodnessOfFit):
    failure_message_name = "Goodness of fit p-value (cancer)"


# less than threshold checks
# --------------------------------------------------------------------------------------------------
class ShouldBeLessThan(Check):
    failure_message_name: str

    @classmethod
    def get_value(cls, dataset, model) -> Optional[Number]:
        ...

    @classmethod
    def apply_rule(cls, dataset, model, rule_settings) -> CheckResponse:
        value = cls.get_value(dataset, model)
        threshold = rule_settings.threshold
        if pass_if_lte(value, threshold):
            return CheckResponse.success()
        return CheckResponse(
            logic_bin=rule_settings.failure_bin,
            message=f"{cls.failure_message_name} greater than threshold ({ff(value)} > {threshold})",
        )


class LargeRoi(ShouldBeLessThan):
    failure_message_name = "Abs(Residual of interest)"

    @classmethod
    def get_value(cls, dataset, model) -> Optional[Number]:
        return abs(model.results.gof.roi)


class BmdBmdlRatio(ShouldBeLessThan):
    failure_message_name = "BMD/BMDL ratio"

    @classmethod
    def get_value(cls, dataset, model) -> Optional[Number]:
        bmd = model.results.bmd
        bmdl = model.results.bmd
        if is_valid_number(bmd) and is_valid_number(bmdl) and bmdl != 0:
            return bmd / bmdl
        return None


class LowBmd(ShouldBeLessThan):
    failure_message_name = "lowest dose/BMD ratio"

    @classmethod
    def get_value(cls, dataset, model) -> Optional[Number]:
        min_dose = min([dose for dose in dataset.doses if dose > 0])
        bmd = model.results.bmd
        if is_valid_number(min_dose) and is_valid_number(bmd) and bmd > 0:
            return min_dose / float(bmd)
        return None


class LowBmdl(ShouldBeLessThan):
    failure_message_name = "lowest dose/BMDL ratio"

    @classmethod
    def get_value(cls, dataset, model) -> Optional[Number]:
        min_dose = min([d for d in dataset.doses if d > 0])
        bmdl = model.results.bmdl
        if is_valid_number(min_dose) and is_valid_number(bmdl) and bmdl > 0:
            return min_dose / float(bmdl)
        return None


class HighBmd(ShouldBeLessThan):
    failure_message_name = "BMD/highest dose ratio"

    @classmethod
    def get_value(cls, dataset, model) -> Optional[Number]:
        max_dose = max(dataset.doses)
        bmd = model.results.bmd
        if is_valid_number(max_dose) and is_valid_number(bmd) and bmd != 0:
            return bmd / float(max_dose)
        return None


class HighBmdl(ShouldBeLessThan):
    failure_message_name = "BMDL/highest dose ratio"

    @classmethod
    def get_value(cls, dataset, model) -> Optional[Number]:
        max_dose = max(dataset.doses)
        bmdl = model.results.bmdl
        if is_valid_number(max_dose) and is_valid_number(bmdl) and bmdl != 0:
            return bmdl / float(max_dose)
        return None


class HighControlResidual(ShouldBeLessThan):
    failure_message_name = "Residual at control"

    @classmethod
    def get_value(cls, dataset, model) -> Optional[Number]:
        residual = model.results.gof.residual[0]
        return abs(residual)


class ControlStdevFit(ShouldBeLessThan):
    failure_message_name = "Control stdev. fit"

    @classmethod
    def get_value(cls, dataset, model) -> Optional[Number]:
        # TODO - use correct value
        return 3


# assorted checks
# --------------------------------------------------------------------------------------------------
class VarianceFit(Check):
    @classmethod
    def apply_rule(cls, dataset, model, rule_settings) -> CheckResponse:
        constant_variance = model.settings.disttype != DistType.normal_ncv
        anova = dataset.anova()
        p_value2 = anova.test2.TEST
        p_value3 = anova.test3.TEST

        if is_valid_number(p_value2) and constant_variance and p_value2 < rule_settings.threshold:
            return CheckResponse(
                logic_bin=rule_settings.failure_bin,
                message=f"Variance model poorly fits dataset (p-value 2 = {ff(p_value2)})",
            )

        if (
            is_valid_number(p_value3)
            and not constant_variance
            and p_value3 < rule_settings.threshold
        ):
            return CheckResponse(
                logic_bin=rule_settings.failure_bin,
                message=f"Variance model poorly fits dataset (p-value 3 = {ff(p_value3)})",
            )

        return CheckResponse.success()


class VarianceType(Check):
    @classmethod
    def apply_rule(cls, dataset, model, rule_settings) -> CheckResponse:
        constant_variance = model.settings.disttype != DistType.normal_ncv
        anova = dataset.anova()
        p_value2 = anova.test2.TEST
        threshold = rule_settings.threshold

        message = None
        if is_valid_number(p_value2):
            # constant variance
            if constant_variance and p_value2 < threshold:
                message = f"Incorrect variance model (p-value 2 = {ff(p_value2)}), constant variance selected"
            elif not constant_variance and p_value2 > threshold:
                message = f"Incorrect variance model (p-value 2 = {ff(p_value2)}), modeled variance selected"
        else:
            message = f"Correct variance model cannot be determined (p-value 2 = {ff(p_value2)})"

        if message:
            return CheckResponse(logic_bin=rule_settings.failure_bin, message=message)

        return CheckResponse.success()


class NoDegreesOfFreedom(Check):
    @classmethod
    def apply_rule(cls, dataset, model, rule_settings) -> CheckResponse:
        if dataset.dtype in constants.DICHOTOMOUS_DTYPES:
            value = model.results.gof.df
        elif dataset.dtype in constants.CONTINUOUS_DTYPES:
            value = 1  # TODO - fix
        else:
            raise ValueError("Unknown dtype")

        if value > 0:
            return CheckResponse.success()
        else:
            return CheckResponse(
                logic_bin=rule_settings.failure_bin,
                message="Zero degrees of freedom; saturated model",
            )


class Warnings(Check):
    @classmethod
    def apply_rule(cls, settings, dataset, output) -> CheckResponse:
        # TODO - doesn't exist with bmds3?
        return CheckResponse.success()


RULE_MAP = {
    RuleClass.gof: GoodnessOfFit,
    RuleClass.dof_zero: NoDegreesOfFreedom,
    RuleClass.high_bmd: HighBmd,
    RuleClass.warnings: Warnings,
    RuleClass.high_bmdl: HighBmdl,
    RuleClass.roi_large: LargeRoi,
    RuleClass.gof_cancer: GoodnessOfFitCancer,
    RuleClass.aic_missing: AicExists,
    RuleClass.bmd_missing: BmdExists,
    RuleClass.roi_missing: RoiExists,
    RuleClass.bmdl_missing: BmdlExists,
    RuleClass.bmdu_missing: BmduExists,
    RuleClass.low_bmd_fail: LowBmd,
    RuleClass.low_bmd_warn: LowBmd,
    RuleClass.variance_fit: VarianceFit,
    RuleClass.low_bmdl_fail: LowBmdl,
    RuleClass.low_bmdl_warn: LowBmdl,
    RuleClass.variance_type: VarianceType,
    RuleClass.control_stdev_fit: ControlStdevFit,
    RuleClass.bmd_bmdl_ratio_fail: BmdBmdlRatio,
    RuleClass.bmd_bmdl_ratio_warn: BmdBmdlRatio,
    RuleClass.control_residual_high: HighControlResidual,
}
