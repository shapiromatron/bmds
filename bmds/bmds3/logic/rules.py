from typing import Dict, Optional

from pydantic import BaseModel

from ... import constants
from ..types.continuous import DistType


def nested_get(d: Dict, key_str: str):
    keys = key_str.split(".")
    val = d
    for key in keys:
        if not isinstance(val, dict):
            return None
        val = val.get(key)
    return val


class Rule(BaseModel):
    rule_name: str
    field_name: Optional[str]
    failure_bin: int
    threshold: Optional[float]
    enabled_nested: bool
    enabled_continuous: bool
    enabled_dichotomous: bool

    def __unicode__(self):
        enabled_nested = "✓" if self.enabled_nested else "✕"
        enabled_continuous = "✓" if self.enabled_continuous else "✕"
        enabled_dichotomous = "✓" if self.enabled_dichotomous else "✕"
        threshold = "" if self.threshold is None else f", threshold={self.threshold}"
        return f"{enabled_nested}{enabled_continuous}{enabled_dichotomous} {self.rule_name} [bin={self.binmoji}{threshold}]"

    def enabled(self, dtype):
        # TODO add conditional for nested
        enabled_continuous = self.enabled_continuous and dtype in constants.CONTINUOUS_DTYPES
        enabled_dichotomous = self.enabled_dichotomous and dtype in constants.DICHOTOMOUS_DTYPES
        return enabled_continuous or enabled_dichotomous

    def check(self, dtype, settings, dataset, output):
        if self.enabled(dtype):
            return self.apply_rule(settings, dataset, output)
        else:
            return self.return_pass()

    def get_value(self, dataset, output):
        return nested_get(output, self.field_name)

    @property
    def binmoji(self):
        return constants.BINMOJI[self.failure_bin]

    @property
    def bin_text(self):
        return constants.BIN_TEXT[self.failure_bin]

    def as_row(self):
        return [
            self.rule_name,
            self.enabled_nested,
            self.enabled_continuous,
            self.enabled_dichotomous,
            self.bin_text,
            self.threshold,
        ]

    def return_pass(self):
        return constants.BIN_NO_CHANGE, None

    def apply_rule(self, settings, dataset, output):
        # return tuple of (bin, notes) associated with rule or None
        raise NotImplementedError("Abstract method.")

    def get_failure_message(self, *args):
        raise NotImplementedError("Abstract method.")

    def _is_valid_number(self, val):
        # Ensure number is an int or float, not equal to special case -999.
        return val is not None and val != -999 and (isinstance(val, int) or isinstance(val, float))


class NumericValueExists(Rule):
    # Test succeeds if value is numeric and not -999
    field_name_verbose: str

    def apply_rule(self, settings, dataset, output):
        val = self.get_value(dataset, output)
        if self._is_valid_number(val):
            return self.return_pass()
        else:
            return self.failure_bin, self.get_failure_message()

    def get_failure_message(self):
        return f"{self.field_name_verbose} does not exist"


class AicExists(NumericValueExists):
    rule_name = "AIC calculated"
    field_name = "aic"
    field_name_verbose = "AIC"


class BmdExists(NumericValueExists):
    rule_name = "BMD calculated"
    field_name = "bmd"
    field_name_verbose = "BMD"


class RoiExists(NumericValueExists):
    rule_name = "Residual of Interest calculated"
    field_name = "roi"
    field_name_verbose = "ROI"


class BmdlExists(NumericValueExists):
    rule_name = "BMDL calculated"
    field_name = "bmdl"
    field_name_verbose = "BMDL"


class BmduExists(NumericValueExists):
    rule_name = "BMDU calculated"
    field_name = "bmdu"
    field_name_verbose = "BMDU"


class ShouldBeGreaterThan(Rule):
    # Test fails if value is less-than threshold.
    field_name_verbose: str

    def apply_rule(self, settings, dataset, output):
        val = self.get_value(dataset, output)
        threshold = self.threshold

        if not self._is_valid_number(val) or val >= threshold:
            return self.return_pass()
        else:
            return self.failure_bin, self.get_failure_message(val, threshold)

    def get_failure_message(self, val, threshold):
        return f"{self.field_name_verbose} is less than threshold ({float(val):.3} < {threshold})"


class GoodnessOfFit(ShouldBeGreaterThan):
    rule_name = "Goodness of fit p-test"
    field_name = "gof.p_value"
    field_name_verbose = "Goodness of fit p-value"


class GoodnessOfFitCancer(ShouldBeGreaterThan):
    rule_name = "Goodness of fit p-test (cancer)"
    field_name = "gof.p_value"
    field_name_verbose = "Goodness of fit p-value (cancer)"


class ShouldBeLessThan(Rule):
    # Test fails if value is less-than threshold.
    field_name_verbose: str

    def apply_rule(self, settings, dataset, output):
        val = self.get_value(dataset, output)
        threshold = self.threshold

        if not self._is_valid_number(val) or val <= threshold:
            return self.return_pass()
        else:
            return self.failure_bin, self.get_failure_message(val, threshold)

    def get_failure_message(self, val, threshold):
        return (
            f"{self.field_name_verbose} is greater than threshold ({float(val):.3} > {threshold})"
        )


class LargeRoi(ShouldBeLessThan):
    rule_name = "Abs(Residual of interest) too large"
    field_name_verbose = "Residual of interest"

    def get_value(self, dataset, output):
        return abs(output.get("roi"))


class BmdBmdlRatio(ShouldBeLessThan):
    rule_name = "Ratio of BMD/BMDL"
    field_name_verbose = "BMD/BMDL ratio"

    def get_value(self, dataset, output):
        bmd = output.get("bmd")
        bmdl = output.get("bmdl")
        if self._is_valid_number(bmd) and self._is_valid_number(bmdl) and bmdl != 0:
            return bmd / bmdl


class LowBmd(ShouldBeLessThan):
    rule_name = "BMD lower than lowest dose"
    field_name_verbose = "BMD/lowest dose ratio"

    def get_value(self, dataset, output):
        min_dose = min([d for d in dataset.doses if d > 0])
        bmd = output.get("bmd")
        if self._is_valid_number(min_dose) and self._is_valid_number(bmd) and bmd > 0:
            return min_dose / float(bmd)


class LowBmdl(ShouldBeLessThan):
    rule_name = "BMDL lower than lowest dose"
    field_name_verbose = "BMDL/lowest dose ratio"

    def get_value(self, dataset, output):
        min_dose = min([d for d in dataset.doses if d > 0])
        bmdl = output.get("bmdl")
        if self._is_valid_number(min_dose) and self._is_valid_number(bmdl) and bmdl > 0:
            return min_dose / float(bmdl)


class HighBmd(ShouldBeLessThan):
    rule_name = "BMD higher than highest dose"
    field_name_verbose = "BMD/highest dose ratio"

    def get_value(self, dataset, output):
        max_dose = max(dataset.doses)
        bmd = output.get("bmd")
        if self._is_valid_number(max_dose) and self._is_valid_number(bmd) and bmd != 0:
            return bmd / float(max_dose)


class HighBmdl(ShouldBeLessThan):
    rule_name = "BMDL higher than highest dose"
    field_name_verbose = "BMDL/highest dose ratio"

    def get_value(self, dataset, output):
        max_dose = max(dataset.doses)
        bmdl = output.get("bmdl")
        if self._is_valid_number(max_dose) and self._is_valid_number(bmdl) and max_dose > 0:
            return bmdl / float(max_dose)


class HighControlRisidual(ShouldBeLessThan):
    rule_name = "Abs(Residual at control) too large"
    field_name = "gof.residual"
    field_name_verbose = "Residual at control"

    def get_value(self, dataset, output):
        return abs(super().get_value(dataset, output)[0])


class ControlStdevFit(ShouldBeLessThan):
    rule_name = "Poor control dose std. dev."
    field_name = ""
    field_name_verbose = ""

    def apply_rule(self, settings, dataset, output):
        return self.return_pass()

    # TODO for continuous


class VarianceFit(Rule):
    rule_name = "Constant Variance"

    def apply_rule(self, settings, dataset, output):
        constant_variance = settings.disttype != DistType.normal_ncv

        p_value2 = dataset.anova()[1].TEST

        p_value3 = dataset.anova()[2].TEST

        msg = None
        if self._is_valid_number(p_value2) and constant_variance and p_value2 < 0.1:
            msg = "Variance model poorly fits dataset (p-value 2 = {})".format(p_value2)

        if self._is_valid_number(p_value3) and not constant_variance and p_value3 < 0.1:
            msg = "Variance model poorly fits dataset (p-value 3 = {})".format(p_value3)

        if msg:
            return self.failure_bin, msg
        else:
            return self.return_pass()


class VarianceType(Rule):
    rule_name = "Non-Constant Variance"

    def apply_rule(self, settings, dataset, output):
        constant_variance = settings.disttype != DistType.normal_ncv

        p_value2 = dataset.anova()[1].TEST

        msg = None
        if self._is_valid_number(p_value2):
            # constant variance
            if constant_variance and p_value2 < 0.1:
                msg = "Incorrect variance model (p-value 2 = {}), constant variance selected".format(
                    p_value2
                )
            elif not constant_variance and p_value2 > 0.1:
                msg = "Incorrect variance model (p-value 2 = {}), modeled variance selected".format(
                    p_value2
                )
        else:
            msg = "Correct variance model cannot be determined (p-value 2 = {})".format(p_value2)

        if msg:
            return self.failure_bin, msg
        else:
            return self.return_pass()


class NoDegreesOfFreedom(Rule):
    rule_name = "D.O.F equal 0"
    field_name = "gof.df"

    def apply_rule(self, settings, dataset, output):
        val = self.get_value(dataset, output)

        if val == 0:
            return self.failure_bin, "Zero degrees of freedom; saturated model"
        else:
            return self.return_pass()


class Warnings(Rule):
    rule_name = "BMDS model Warning"

    def apply_rule(self, settings, dataset, output):
        return self.return_pass()

    # TODO not present on bmds3 output
