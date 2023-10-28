from copy import deepcopy
from typing import ClassVar

import numpy as np
from scipy.stats import gamma, norm

from ... import constants
from ...constants import Version
from .base import BMDModel, DefaultParams


class Dichotomous(BMDModel):
    dtype = constants.DICHOTOMOUS
    possible_bmr = ("Extra", "Added")

    def get_bmr_text(self):
        return "{:.0%} {} risk".format(
            self.values["bmr"],
            constants.BMR_INVERTED_CROSSALK[self.dtype][self.values["bmr_type"]],
        )


class DichotomousCancer(Dichotomous):
    dtype = constants.DICHOTOMOUS_CANCER
    possible_bmr = ("Extra", "Added")


# MULTISTAGE
class Multistage_34(Dichotomous):
    bmds_version_dir = Version.BMDS270
    version = 3.4
    date = "05/02/2014"
    minimum_dose_groups = 2
    model_name = constants.M_Multistage
    exe = "multistage"
    exe_plot = "10multista"
    defaults: ClassVar = {
        "bmdl_curve_calculation": DefaultParams.bmdl_curve_calculation,
        "append_or_overwrite": DefaultParams.append_or_overwrite,
        "smooth_option": DefaultParams.smooth_option,
        "max_iterations": DefaultParams.max_iterations,
        "relative_fn_conv": DefaultParams.relative_fn_conv,
        "parameter_conv": DefaultParams.parameter_conv,
        "background": DefaultParams.param_generator(name="Background"),
        "beta1": DefaultParams.param_generator(name="Beta1"),
        "beta2": DefaultParams.param_generator(name="Beta2"),
        "beta3": DefaultParams.param_generator(name="Beta3"),
        "beta4": DefaultParams.param_generator(name="Beta4"),
        "beta5": DefaultParams.param_generator(name="Beta5"),
        "beta6": DefaultParams.param_generator(name="Beta6"),
        "beta7": DefaultParams.param_generator(name="Beta7"),
        "beta8": DefaultParams.param_generator(name="Beta8"),
        "restrict_beta": DefaultParams.restrict(d=1, n="Restrict beta"),
        "degree_poly": DefaultParams.degree_poly(),
        "bmd_calculation": DefaultParams.bmd_calculation,
        "dose_drop": DefaultParams.dose_drop,
        "bmr": DefaultParams.dich_bmr,
        "bmr_type": DefaultParams.dich_bmr_type,
        "confidence_level": DefaultParams.confidence_level,
    }

    @property
    def name(self):
        return f"{self.model_name}-{self._get_degrees()}"

    def _get_degrees(self):
        degree = int(self.values["degree_poly"])
        if not 0 < degree <= 8:
            raise ValueError("Degree must be between 1 and 8, inclusive")
        return degree

    def as_dfile(self):
        self._set_values()
        degree_poly = self._get_degrees()

        params = [f"beta{i}" for i in range(1, degree_poly + 1)]
        params.insert(0, "background")

        return "\n".join(
            [
                self._dfile_print_header_rows(),
                f"{self.dataset.dataset_length} {degree_poly}",
                self._dfile_print_options(
                    "max_iterations",
                    "relative_fn_conv",
                    "parameter_conv",
                    "bmdl_curve_calculation",
                    "restrict_beta",
                    "bmd_calculation",
                    "append_or_overwrite",
                    "smooth_option",
                ),
                self._dfile_print_options("bmr", "bmr_type", "confidence_level"),
                self._dfile_print_parameters(*params),
                self.dataset.as_dfile(),
            ]
        )

    def get_ys(self, xs):
        background = self._get_param("Background")
        ys = np.zeros(xs.size)
        for i in range(1, self._get_degrees() + 1):
            param = self._get_param(f"Beta({i})")
            ys += param * np.power(xs, i)
        ys = background + (1.0 - background) * (1.0 - np.exp(-1.0 * ys))
        return ys


# QUANTAL LINEAR
class QuantalLinear_34(Multistage_34):
    defaults = deepcopy(Multistage_34.defaults)
    defaults["degree_poly"] = DefaultParams.degree_poly(d=1)

    @property
    def name(self):
        return "Quantal linear"


# MULTISTAGE CANCER
class MultistageCancer_34(DichotomousCancer):
    bmds_version_dir = Version.BMDS270
    version = 3.4
    date = "05/02/2014"
    minimum_dose_groups = 2
    model_name = constants.M_MultistageCancer
    exe = "cancer"
    exe_plot = "10cancer"
    defaults: ClassVar = {
        "bmdl_curve_calculation": DefaultParams.bmdl_curve_calculation,
        "append_or_overwrite": DefaultParams.append_or_overwrite,
        "smooth_option": DefaultParams.smooth_option,
        "max_iterations": DefaultParams.max_iterations,
        "relative_fn_conv": DefaultParams.relative_fn_conv,
        "parameter_conv": DefaultParams.parameter_conv,
        "background": DefaultParams.param_generator(name="Background"),
        "beta1": DefaultParams.param_generator(name="Beta1"),
        "beta2": DefaultParams.param_generator(name="Beta2"),
        "beta3": DefaultParams.param_generator(name="Beta3"),
        "beta4": DefaultParams.param_generator(name="Beta4"),
        "beta5": DefaultParams.param_generator(name="Beta5"),
        "beta6": DefaultParams.param_generator(name="Beta6"),
        "beta7": DefaultParams.param_generator(name="Beta7"),
        "beta8": DefaultParams.param_generator(name="Beta8"),
        "restrict_beta": DefaultParams.restrict(d=1, n="Restrict beta"),
        "degree_poly": DefaultParams.degree_poly(),
        "bmd_calculation": DefaultParams.bmd_calculation,
        "dose_drop": DefaultParams.dose_drop,
        "bmr": DefaultParams.dich_bmr,
        "bmr_type": DefaultParams.dich_bmr_type,
        "confidence_level": DefaultParams.confidence_level,
    }

    @property
    def name(self):
        return f"{self.model_name}-{self._get_degrees()}"

    def _get_degrees(self):
        degree = int(self.values["degree_poly"])
        if not 0 < degree <= 8:
            raise ValueError("Degree must be between 1 and 8, inclusive")
        return degree

    def as_dfile(self):
        self._set_values()
        degree_poly = self._get_degrees()

        params = [f"beta{i}" for i in range(1, degree_poly + 1)]
        params.insert(0, "background")

        return "\n".join(
            [
                self._dfile_print_header_rows(),
                f"{self.dataset.dataset_length} {degree_poly}",
                self._dfile_print_options(
                    "max_iterations",
                    "relative_fn_conv",
                    "parameter_conv",
                    "bmdl_curve_calculation",
                    "restrict_beta",
                    "bmd_calculation",
                    "append_or_overwrite",
                    "smooth_option",
                ),
                self._dfile_print_options("bmr", "bmr_type", "confidence_level"),
                self._dfile_print_parameters(*params),
                self.dataset.as_dfile(),
            ]
        )

    def get_ys(self, xs):
        background = self._get_param("Background")
        ys = np.zeros(xs.size)
        for i in range(1, self._get_degrees() + 1):
            param = self._get_param(f"Beta({i})")
            ys += param * np.power(xs, i)
        ys = background + (1.0 - background) * (1.0 - np.exp(-1.0 * ys))
        return ys


# WEIBULL
class Weibull_217(Dichotomous):
    bmds_version_dir = Version.BMDS270
    version = 2.17
    date = "06/23/2017"
    minimum_dose_groups = 3
    model_name = constants.M_Weibull
    exe = "weibull"
    exe_plot = "10weibull"
    defaults: ClassVar = {
        "bmdl_curve_calculation": DefaultParams.bmdl_curve_calculation,
        "append_or_overwrite": DefaultParams.append_or_overwrite,
        "smooth_option": DefaultParams.smooth_option,
        "log_transform": DefaultParams.log_transform(d=0),
        "max_iterations": DefaultParams.max_iterations,
        "relative_fn_conv": DefaultParams.relative_fn_conv,
        "parameter_conv": DefaultParams.parameter_conv,
        "background": DefaultParams.param_generator(),
        "slope": DefaultParams.param_generator(name="Slope"),
        "power": DefaultParams.param_generator(name="Power"),
        "restrict_power": DefaultParams.restrict(d=1, n="Restrict power"),
        "bmd_calculation": DefaultParams.bmd_calculation,
        "dose_drop": DefaultParams.dose_drop,
        "bmr": DefaultParams.dich_bmr,
        "bmr_type": DefaultParams.dich_bmr_type,
        "confidence_level": DefaultParams.confidence_level,
    }

    def as_dfile(self):
        self._set_values()
        return "\n".join(
            [
                self._dfile_print_header_rows(),
                str(self.dataset.dataset_length),
                self._dfile_print_options(
                    "max_iterations",
                    "relative_fn_conv",
                    "parameter_conv",
                    "bmdl_curve_calculation",
                    "restrict_power",
                    "bmd_calculation",
                    "append_or_overwrite",
                    "smooth_option",
                ),
                self._dfile_print_options("bmr", "bmr_type", "confidence_level"),
                self._dfile_print_parameters("background", "slope", "power"),
                self.dataset.as_dfile(),
            ]
        )

    def get_ys(self, xs):
        background = self._get_param("Background")
        slope = self._get_param("Slope")
        power = self._get_param("Power")
        ys = background + (1.0 - background) * (1 - np.exp(-1.0 * slope * np.power(xs, power)))
        return ys


# LOGPROBIT
class LogProbit_34(Dichotomous):
    bmds_version_dir = Version.BMDS270
    version = 3.4
    date = "05/21/2017"
    minimum_dose_groups = 3
    model_name = constants.M_LogProbit
    exe = "probit"
    exe_plot = "10probit"
    defaults: ClassVar = {
        "bmdl_curve_calculation": DefaultParams.bmdl_curve_calculation,
        "append_or_overwrite": DefaultParams.append_or_overwrite,
        "smooth_option": DefaultParams.smooth_option,
        "log_transform": DefaultParams.log_transform(d=1),
        "max_iterations": DefaultParams.max_iterations,
        "relative_fn_conv": DefaultParams.relative_fn_conv,
        "parameter_conv": DefaultParams.parameter_conv,
        "background": DefaultParams.param_generator(),
        "slope": DefaultParams.param_generator(name="Slope"),
        "intercept": DefaultParams.param_generator(name="Intercept"),
        "restrict_slope": DefaultParams.restrict(d=1, n="Restrict slope"),
        "bmd_calculation": DefaultParams.bmd_calculation,
        "dose_drop": DefaultParams.dose_drop,
        "bmr": DefaultParams.dich_bmr,
        "bmr_type": DefaultParams.dich_bmr_type,
        "confidence_level": DefaultParams.confidence_level,
    }

    def as_dfile(self):
        self._set_values()
        return "\n".join(
            [
                self._dfile_print_header_rows(),
                str(self.dataset.dataset_length),
                self._dfile_print_options(
                    "max_iterations",
                    "relative_fn_conv",
                    "parameter_conv",
                    "bmdl_curve_calculation",
                    "log_transform",
                    "restrict_slope",
                    "bmd_calculation",
                    "append_or_overwrite",
                    "smooth_option",
                ),
                self._dfile_print_options("bmr", "bmr_type", "confidence_level"),
                self._dfile_print_parameters("background", "slope", "intercept"),
                self.dataset.as_dfile(),
            ]
        )

    def get_ys(self, xs):
        background = self._get_param("background")
        slope = self._get_param("slope")
        intercept = self._get_param("intercept")
        ys = background + (1.0 - background) * norm.cdf(intercept + slope * np.log(xs))
        return ys


# PROBIT
class Probit_34(Dichotomous):
    bmds_version_dir = Version.BMDS270
    version = 3.4
    date = "05/21/2017"
    minimum_dose_groups = 3
    model_name = constants.M_Probit
    exe = "probit"
    exe_plot = "10probit"
    defaults: ClassVar = {
        "bmdl_curve_calculation": DefaultParams.bmdl_curve_calculation,
        "append_or_overwrite": DefaultParams.append_or_overwrite,
        "smooth_option": DefaultParams.smooth_option,
        "log_transform": DefaultParams.log_transform(d=0),
        "restrict_slope": DefaultParams.restrict(d=0, n="Restrict slope"),
        "max_iterations": DefaultParams.max_iterations,
        "relative_fn_conv": DefaultParams.relative_fn_conv,
        "parameter_conv": DefaultParams.parameter_conv,
        "background": DefaultParams.param_generator(),
        "slope": DefaultParams.param_generator(name="Slope"),
        "intercept": DefaultParams.param_generator(name="Intercept"),
        "bmd_calculation": DefaultParams.bmd_calculation,
        "dose_drop": DefaultParams.dose_drop,
        "bmr": DefaultParams.dich_bmr,
        "bmr_type": DefaultParams.dich_bmr_type,
        "confidence_level": DefaultParams.confidence_level,
    }

    def as_dfile(self):
        self._set_values()
        return "\n".join(
            [
                self._dfile_print_header_rows(),
                str(self.dataset.dataset_length),
                self._dfile_print_options(
                    "max_iterations",
                    "relative_fn_conv",
                    "parameter_conv",
                    "bmdl_curve_calculation",
                    "log_transform",
                    "restrict_slope",
                    "bmd_calculation",
                    "append_or_overwrite",
                    "smooth_option",
                ),
                self._dfile_print_options("bmr", "bmr_type", "confidence_level"),
                self._dfile_print_parameters("background", "slope", "intercept"),
                self.dataset.as_dfile(),
            ]
        )

    def get_ys(self, xs):
        slope = self._get_param("slope")
        intercept = self._get_param("intercept")
        ys = norm.cdf(intercept + slope * xs)
        return ys


# GAMMA
class Gamma_217(Dichotomous):
    bmds_version_dir = Version.BMDS270
    version = 2.17
    date = "06/22/2017"
    minimum_dose_groups = 3
    model_name = constants.M_Gamma
    exe = "gamma"
    exe_plot = "10gammhit"
    defaults: ClassVar = {
        "bmdl_curve_calculation": DefaultParams.bmdl_curve_calculation,
        "append_or_overwrite": DefaultParams.append_or_overwrite,
        "smooth_option": DefaultParams.smooth_option,
        "max_iterations": DefaultParams.max_iterations,
        "relative_fn_conv": DefaultParams.relative_fn_conv,
        "parameter_conv": DefaultParams.parameter_conv,
        "background": DefaultParams.param_generator(),
        "slope": DefaultParams.param_generator(name="Slope"),
        "power": DefaultParams.param_generator(name="Power"),
        "bmd_calculation": DefaultParams.bmd_calculation,
        "restrict_power": DefaultParams.restrict(d=1, n="Restrict power"),
        "dose_drop": DefaultParams.dose_drop,
        "bmr": DefaultParams.dich_bmr,
        "bmr_type": DefaultParams.dich_bmr_type,
        "confidence_level": DefaultParams.confidence_level,
    }

    def as_dfile(self):
        self._set_values()
        return "\n".join(
            [
                self._dfile_print_header_rows(),
                str(self.dataset.dataset_length),
                self._dfile_print_options(
                    "max_iterations",
                    "relative_fn_conv",
                    "parameter_conv",
                    "bmdl_curve_calculation",
                    "restrict_power",
                    "bmd_calculation",
                    "append_or_overwrite",
                    "smooth_option",
                ),
                self._dfile_print_options("bmr", "bmr_type", "confidence_level"),
                self._dfile_print_parameters("background", "slope", "power"),
                self.dataset.as_dfile(),
            ]
        )

    def get_ys(self, xs):
        background = self._get_param("Background")
        slope = self._get_param("Slope")
        power = self._get_param("Power")
        ys = background + (1.0 - background) * gamma.cdf(xs * slope, power)
        return ys


# LOGLOGISTIC
class LogLogistic_215(Dichotomous):
    bmds_version_dir = Version.BMDS270
    version = 2.15
    date = "03/20/2017"
    minimum_dose_groups = 3
    model_name = constants.M_LogLogistic
    exe = "logist"
    exe_plot = "10logist"
    defaults: ClassVar = {
        "bmdl_curve_calculation": DefaultParams.bmdl_curve_calculation,
        "log_transform": DefaultParams.log_transform(d=1),
        "restrict_slope": DefaultParams.restrict(d=1),
        "append_or_overwrite": DefaultParams.append_or_overwrite,
        "smooth_option": DefaultParams.smooth_option,
        "max_iterations": DefaultParams.max_iterations,
        "relative_fn_conv": DefaultParams.relative_fn_conv,
        "parameter_conv": DefaultParams.parameter_conv,
        "background": DefaultParams.param_generator(),
        "slope": DefaultParams.param_generator(name="Slope"),
        "intercept": DefaultParams.param_generator(name="Intercept"),
        "bmd_calculation": DefaultParams.bmd_calculation,
        "dose_drop": DefaultParams.dose_drop,
        "bmr": DefaultParams.dich_bmr,
        "bmr_type": DefaultParams.dich_bmr_type,
        "confidence_level": DefaultParams.confidence_level,
    }

    def as_dfile(self):
        self._set_values()
        return "\n".join(
            [
                self._dfile_print_header_rows(),
                str(self.dataset.dataset_length),
                self._dfile_print_options(
                    "max_iterations",
                    "relative_fn_conv",
                    "parameter_conv",
                    "bmdl_curve_calculation",
                    "log_transform",
                    "restrict_slope",
                    "bmd_calculation",
                    "append_or_overwrite",
                    "smooth_option",
                ),
                self._dfile_print_options("bmr", "bmr_type", "confidence_level"),
                self._dfile_print_parameters("background", "slope", "intercept"),
                self.dataset.as_dfile(),
            ]
        )

    def get_ys(self, xs):
        background = self._get_param("background")
        intercept = self._get_param("intercept")
        slope = self._get_param("slope")
        ys = background + (1.0 - background) / (
            1 + np.exp(-1.0 * intercept - 1.0 * slope * np.log(xs))
        )
        return ys


# LOGISTIC
class Logistic_215(Dichotomous):
    bmds_version_dir = Version.BMDS270
    version = 2.15
    date = "03/20/2017"
    minimum_dose_groups = 3
    model_name = constants.M_Logistic
    exe = "logist"
    exe_plot = "10logist"
    defaults: ClassVar = {
        "bmdl_curve_calculation": DefaultParams.bmdl_curve_calculation,
        "log_transform": DefaultParams.log_transform(d=0),
        "restrict_slope": DefaultParams.restrict(d=0),
        "append_or_overwrite": DefaultParams.append_or_overwrite,
        "smooth_option": DefaultParams.smooth_option,
        "max_iterations": DefaultParams.max_iterations,
        "relative_fn_conv": DefaultParams.relative_fn_conv,
        "parameter_conv": DefaultParams.parameter_conv,
        "background": DefaultParams.param_generator(),
        "slope": DefaultParams.param_generator(name="Slope"),
        "intercept": DefaultParams.param_generator(name="Intercept"),
        "bmd_calculation": DefaultParams.bmd_calculation,
        "dose_drop": DefaultParams.dose_drop,
        "bmr": DefaultParams.dich_bmr,
        "bmr_type": DefaultParams.dich_bmr_type,
        "confidence_level": DefaultParams.confidence_level,
    }

    def as_dfile(self):
        self._set_values()
        return "\n".join(
            [
                self._dfile_print_header_rows(),
                str(self.dataset.dataset_length),
                self._dfile_print_options(
                    "max_iterations",
                    "relative_fn_conv",
                    "parameter_conv",
                    "bmdl_curve_calculation",
                    "log_transform",
                    "restrict_slope",
                    "bmd_calculation",
                    "append_or_overwrite",
                    "smooth_option",
                ),
                self._dfile_print_options("bmr", "bmr_type", "confidence_level"),
                self._dfile_print_parameters("background", "slope", "intercept"),
                self.dataset.as_dfile(),
            ]
        )

    def get_ys(self, xs):
        intercept = self._get_param("intercept")
        slope = self._get_param("slope")
        ys = 1.0 / (1.0 + np.exp(-1.0 * intercept - slope * xs))
        return ys


# DICHOTOMOUS HILL
class DichotomousHill_13(Dichotomous):
    minimum_dose_groups = 4
    model_name = constants.M_DichotomousHill
    bmds_version_dir = Version.BMDS270
    exe = "DichoHill"
    exe_plot = "10DichoHill"
    version = 1.3
    date = "02/28/2013"
    defaults: ClassVar = {
        "bmdl_curve_calculation": DefaultParams.bmdl_curve_calculation,
        "restrict_power": DefaultParams.log_transform(d=1),
        "append_or_overwrite": DefaultParams.append_or_overwrite,
        "smooth_option": DefaultParams.smooth_option,
        "max_iterations": DefaultParams.max_iterations,
        "relative_fn_conv": DefaultParams.relative_fn_conv,
        "parameter_conv": DefaultParams.parameter_conv,
        "v": DefaultParams.param_generator(),
        "g": DefaultParams.param_generator(),
        "intercept": DefaultParams.param_generator(),
        "slope": DefaultParams.param_generator(),
        "bmd_calculation": DefaultParams.bmd_calculation,
        "dose_drop": DefaultParams.dose_drop,
        "bmr": DefaultParams.dich_bmr,
        "bmr_type": DefaultParams.dich_bmr_type,
        "confidence_level": DefaultParams.confidence_level,
    }

    def as_dfile(self):
        self._set_values()
        return "\n".join(
            [
                self._dfile_print_header_rows(),
                str(self.dataset.dataset_length),
                self._dfile_print_options(
                    "max_iterations",
                    "relative_fn_conv",
                    "parameter_conv",
                    "bmdl_curve_calculation",
                    "restrict_power",
                    "bmd_calculation",
                    "append_or_overwrite",
                    "smooth_option",
                ),
                self._dfile_print_options("bmr", "bmr_type", "confidence_level"),
                self._dfile_print_parameters("v", "g", "intercept", "slope"),
                self.dataset.as_dfile(),
            ]
        )

    def get_ys(self, xs):
        v = self._get_param("v")
        g = self._get_param("g")
        intercept = self._get_param("intercept")
        slope = self._get_param("slope")
        ys = v * g + (v - v * g) / (1.0 + np.exp(-intercept - slope * np.log(xs)))
        return ys
