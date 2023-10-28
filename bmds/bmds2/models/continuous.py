import os
from typing import ClassVar

import numpy as np

from ... import constants
from ...constants import Version
from .base import BMDModel, DefaultParams


class Continuous(BMDModel):
    dtype = "C"  # for parsing output; therefore C is equivalent to CI
    possible_bmr = ("Abs. Dev.", "Std. Dev.", "Rel. Dev.", "Point", "Extra")

    def get_bmr_text(self):
        return "{} {}".format(
            self.values["bmr"],
            constants.BMR_INVERTED_CROSSALK[self.dtype][self.values["bmr_type"]],
        )

    def set_constant_variance_value(self):
        # set constant variance if p-test 2 >= 0.1, otherwise use modeled variance
        # 0 = non-homogeneous modeled variance => Var(i) = alpha*mean(i)^rho
        # 1 = constant variance => Var(i) = alpha*mean(i)
        anova = self.dataset.anova()
        return 0 if (anova is None or anova.test2.TEST < 0.1) else 1

    def get_variance_model_name(self):
        return "Modeled variance" if self.values["constant_variance"] == 0 else "Constant variance"


# POLYNOMIAL
class Polynomial_221(Continuous):
    minimum_dose_groups = 2
    model_name = constants.M_Polynomial
    bmds_version_dir = Version.BMDS270
    version = 2.21
    date = "03/14/2017"
    exe = "poly"
    exe_plot = "00poly"
    defaults: ClassVar = {
        "bmdl_curve_calculation": DefaultParams.bmdl_curve_calculation,
        "append_or_overwrite": DefaultParams.append_or_overwrite,
        "smooth_option": DefaultParams.smooth_option,
        "max_iterations": DefaultParams.max_iterations,
        "relative_fn_conv": DefaultParams.relative_fn_conv,
        "parameter_conv": DefaultParams.parameter_conv,
        "alpha": DefaultParams.param_generator("Alpha"),
        "rho": DefaultParams.param_generator("Rho"),
        "beta_0": DefaultParams.param_generator("Beta0"),
        "beta_1": DefaultParams.param_generator("Beta1"),
        "beta_2": DefaultParams.param_generator("Beta2"),
        "beta_3": DefaultParams.param_generator("Beta3"),
        "beta_4": DefaultParams.param_generator("Beta4"),
        "beta_5": DefaultParams.param_generator("Beta5"),
        "beta_7": DefaultParams.param_generator("Beta7"),
        "beta_6": DefaultParams.param_generator("Beta6"),
        "beta_8": DefaultParams.param_generator("Beta8"),
        "restrict_polynomial": DefaultParams.restrict(d=0, n="Restrict polynomial"),
        "degree_poly": DefaultParams.degree_poly(),
        "bmd_calculation": DefaultParams.bmd_calculation,
        "dose_drop": DefaultParams.dose_drop,
        "bmr": DefaultParams.cont_bmr,
        "bmr_type": DefaultParams.cont_bmr_type,
        "confidence_level": DefaultParams.confidence_level,
        "constant_variance": DefaultParams.constant_variance,
    }

    @property
    def name(self):
        return f"{self.model_name}-{self._get_degrees()}"

    def set_restrict_polynomial_value(self):
        return 1 if self.dataset.is_increasing else -1

    def _get_degrees(self):
        degree = int(self.values["degree_poly"])
        if not 0 < degree <= 8:
            raise ValueError("Degree must be between 1 and 8, inclusive")
        return degree

    def as_dfile(self):
        self._set_values()
        degpoly = self._get_degrees()
        params = ["alpha", "rho", "beta_0"]
        for i in range(1, degpoly + 1):
            params.append("beta_" + str(i))

        return "\n".join(
            [
                self._dfile_print_header_rows(),
                str(degpoly),
                f"{self.dataset._BMDS_DATASET_TYPE} {self.dataset.dataset_length} 0",
                self._dfile_print_options(
                    "max_iterations",
                    "relative_fn_conv",
                    "parameter_conv",
                    "bmdl_curve_calculation",
                    "restrict_polynomial",
                    "bmd_calculation",
                    "append_or_overwrite",
                    "smooth_option",
                ),
                self._dfile_print_options(
                    "bmr_type", "bmr", "constant_variance", "confidence_level"
                ),
                self._dfile_print_parameters(*params),
                self.dataset.as_dfile(),
            ]
        )

    def get_ys(self, xs):
        ys = np.zeros(xs.size)
        for i in range(self._get_degrees() + 1):
            param = self._get_param(f"beta_{i}")
            ys += np.power(xs, i) * param
        return ys


# LINEAR
class Linear_221(Polynomial_221):
    minimum_dose_groups = 2
    model_name = constants.M_Linear
    bmds_version_dir = Version.BMDS270
    exe = "poly"
    exe_plot = "00poly"
    version = 2.21
    date = "03/14/2017"
    defaults: ClassVar = {
        "bmdl_curve_calculation": DefaultParams.bmdl_curve_calculation,
        "append_or_overwrite": DefaultParams.append_or_overwrite,
        "smooth_option": DefaultParams.smooth_option,
        "max_iterations": DefaultParams.max_iterations,
        "relative_fn_conv": DefaultParams.relative_fn_conv,
        "parameter_conv": DefaultParams.parameter_conv,
        "alpha": DefaultParams.param_generator("Alpha"),
        "rho": DefaultParams.param_generator("Rho"),
        "beta_0": DefaultParams.param_generator("Beta0"),
        "beta_1": DefaultParams.param_generator("Beta1"),
        "restrict_polynomial": DefaultParams.restrict(d=0, n="Restrict polynomial"),
        "degree_poly": DefaultParams.degree_poly(d=1, showName=False),
        "bmd_calculation": DefaultParams.bmd_calculation,
        "dose_drop": DefaultParams.dose_drop,
        "bmr": DefaultParams.cont_bmr,
        "bmr_type": DefaultParams.cont_bmr_type,
        "confidence_level": DefaultParams.confidence_level,
        "constant_variance": DefaultParams.constant_variance,
    }

    @property
    def name(self):
        return self.model_name

    def as_dfile(self):
        self._set_values()
        return "\n".join(
            [
                self._dfile_print_header_rows(),
                "1",
                f"{self.dataset._BMDS_DATASET_TYPE} {self.dataset.dataset_length} 0",
                self._dfile_print_options(
                    "max_iterations",
                    "relative_fn_conv",
                    "parameter_conv",
                    "bmdl_curve_calculation",
                    "restrict_polynomial",
                    "bmd_calculation",
                    "append_or_overwrite",
                    "smooth_option",
                ),
                self._dfile_print_options(
                    "bmr_type", "bmr", "constant_variance", "confidence_level"
                ),
                self._dfile_print_parameters("alpha", "rho", "beta_0", "beta_1"),
                self.dataset.as_dfile(),
            ]
        )


# EXPONENTIAL M2
class Exponential(Continuous):
    def _get_model_name(self):
        return f"{self.exe}_{self.output_prefix.lower()}"

    def get_outfile(self, dfile):
        # Exponential model output files play differently.
        #
        # Append the model-prefix to outfile.
        #
        # Note: this function has a side-effect of adding the blank out and 002
        # files which are created to be automatically cleaned-up.
        outfile = super().get_outfile(dfile)
        oo2 = outfile.replace(".out", ".002")
        if os.path.exists(outfile):
            self.tempfiles.append(outfile)
        if os.path.exists(oo2):
            self.tempfiles.append(oo2)
        path, fn = os.path.split(outfile)
        fn = self.output_prefix + fn
        return os.path.join(path, fn)

    def set_adverse_direction(self):
        return 1 if self.dataset.is_increasing else -1

    def as_dfile(self):
        self._set_values()
        params = self._dfile_print_parameters("alpha", "rho", "a", "b", "c", "d")
        return "\n".join(
            [
                self._dfile_print_header_rows(),
                self.exp_run_settings.format(
                    self.dataset._BMDS_DATASET_TYPE,
                    self.dataset.dataset_length,
                    self.set_adverse_direction(),
                ),
                self._dfile_print_options(
                    "max_iterations",
                    "relative_fn_conv",
                    "parameter_conv",
                    "bmdl_curve_calculation",
                    "bmd_calculation",
                    "append_or_overwrite",
                    "smooth_option",
                ),
                self._dfile_print_options(
                    "bmr_type", "bmr", "constant_variance", "confidence_level"
                ),
                "\n".join([params] * 4),
                self.dataset.as_dfile(),
            ]
        )


class Exponential_M2_111(Exponential):
    bmds_version_dir = Version.BMDS270
    version = 1.11
    date = "03/14/2017"
    minimum_dose_groups = 2
    model_name = constants.M_ExponentialM2
    exe = "exponential"
    exe_plot = "Expo_CPlot"
    exp_run_settings = "{} {} {} 1000 11 0 1"
    defaults: ClassVar = {
        "bmdl_curve_calculation": DefaultParams.bmdl_curve_calculation,
        "append_or_overwrite": DefaultParams.append_or_overwrite,
        "smooth_option": DefaultParams.smooth_option,
        "max_iterations": DefaultParams.max_iterations,
        "relative_fn_conv": DefaultParams.relative_fn_conv,
        "parameter_conv": DefaultParams.parameter_conv,
        "alpha": DefaultParams.param_generator("Alpha"),
        "rho": DefaultParams.param_generator("Rho"),
        "a": DefaultParams.param_generator("a"),
        "b": DefaultParams.param_generator("b"),
        "c": DefaultParams.param_generator("c"),
        "d": DefaultParams.param_generator("d"),
        "bmd_calculation": DefaultParams.bmd_calculation,
        "dose_drop": DefaultParams.dose_drop,
        "bmr": DefaultParams.cont_bmr,
        "bmr_type": DefaultParams.cont_bmr_type,
        "confidence_level": DefaultParams.confidence_level,
        "constant_variance": DefaultParams.constant_variance,
    }
    output_prefix = "M2"

    def get_ys(self, xs):
        sign = 1.0 if self.dataset.is_increasing else -1.0
        a = self._get_param("a")
        b = self._get_param("b")
        ys = a * np.exp(sign * b * xs)
        return ys


# EXPONENTIAL M3
class Exponential_M3_111(Exponential_M2_111):
    minimum_dose_groups = 3
    model_name = constants.M_ExponentialM3
    exp_run_settings = "{} {} {} 0100 22 0 1"
    output_prefix = "M3"

    def get_ys(self, xs):
        sign = 1.0 if self.dataset.is_increasing else -1.0
        a = self._get_param("a")
        b = self._get_param("b")
        d = self._get_param("d")
        ys = a * np.exp(sign * np.power(b * xs, d))
        return ys


# EXPONENTIAL M4
class Exponential_M4_111(Exponential_M2_111):
    minimum_dose_groups = 3
    model_name = constants.M_ExponentialM4
    exp_run_settings = "{} {} {} 0010 33 0 1"
    output_prefix = "M4"

    def get_ys(self, xs):
        a = self._get_param("a")
        b = self._get_param("b")
        c = self._get_param("c")
        ys = a * (c - (c - 1.0) * np.exp(-1.0 * b * xs))
        return ys


# EXPONENTIAL M5
class Exponential_M5_111(Exponential_M2_111):
    minimum_dose_groups = 4
    model_name = constants.M_ExponentialM5
    exp_run_settings = "{} {} {} 0001 44 0 1"
    output_prefix = "M5"

    def get_ys(self, xs):
        a = self._get_param("a")
        b = self._get_param("b")
        c = self._get_param("c")
        d = self._get_param("d")
        ys = a * (c - (c - 1.0) * np.exp(-1.0 * np.power(b * xs, d)))
        return ys


# POWER
class Power_219(Continuous):
    bmds_version_dir = Version.BMDS270
    version = 2.19
    date = "03/14/2017"
    minimum_dose_groups = 3
    model_name = constants.M_Power
    exe = "power"
    exe_plot = "00power"
    defaults: ClassVar = {
        "bmdl_curve_calculation": DefaultParams.bmdl_curve_calculation,
        "append_or_overwrite": DefaultParams.append_or_overwrite,
        "smooth_option": DefaultParams.smooth_option,
        "log_transform": DefaultParams.log_transform(d=0),
        "max_iterations": DefaultParams.max_iterations,
        "relative_fn_conv": DefaultParams.relative_fn_conv,
        "parameter_conv": DefaultParams.parameter_conv,
        "alpha": DefaultParams.param_generator("Alpha"),
        "rho": DefaultParams.param_generator("Rho"),
        "control": DefaultParams.param_generator("Control"),
        "slope": DefaultParams.param_generator("Slope"),
        "power": DefaultParams.param_generator("Power"),
        "restrict_power": DefaultParams.restrict(d=1, n="Restrict power"),
        "bmd_calculation": DefaultParams.bmd_calculation,
        "dose_drop": DefaultParams.dose_drop,
        "bmr": DefaultParams.cont_bmr,
        "bmr_type": DefaultParams.cont_bmr_type,
        "confidence_level": DefaultParams.confidence_level,
        "constant_variance": DefaultParams.constant_variance,
    }

    def as_dfile(self):
        self._set_values()
        return "\n".join(
            [
                self._dfile_print_header_rows(),
                f"{self.dataset._BMDS_DATASET_TYPE} {self.dataset.dataset_length} 0",
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
                self._dfile_print_options(
                    "bmr_type", "bmr", "constant_variance", "confidence_level"
                ),
                self._dfile_print_parameters("alpha", "rho", "control", "slope", "power"),
                self.dataset.as_dfile(),
            ]
        )

    def get_ys(self, xs):
        slope = self._get_param("slope")
        control = self._get_param("control")
        power = self._get_param("power")
        ys = control + slope * np.power(xs, power)
        return ys


# HILL
class Hill_218(Continuous):
    bmds_version_dir = Version.BMDS270
    version = 2.18
    date = "03/14/2017"
    minimum_dose_groups = 4
    model_name = constants.M_Hill
    exe = "hill"
    exe_plot = "00Hill"
    defaults: ClassVar = {
        "bmdl_curve_calculation": DefaultParams.bmdl_curve_calculation,
        "append_or_overwrite": DefaultParams.append_or_overwrite,
        "smooth_option": DefaultParams.smooth_option,
        "log_transform": DefaultParams.log_transform(d=0),
        "max_iterations": DefaultParams.max_iterations,
        "relative_fn_conv": DefaultParams.relative_fn_conv,
        "parameter_conv": DefaultParams.parameter_conv,
        "alpha": DefaultParams.param_generator("Alpha"),
        "rho": DefaultParams.param_generator("Rho"),
        "intercept": DefaultParams.param_generator("Intercept"),
        "v": DefaultParams.param_generator("V"),
        "n": DefaultParams.param_generator("N"),
        "k": DefaultParams.param_generator("K"),
        "restrict_n": DefaultParams.restrict(d=1, n="Restrict N>1"),
        "bmd_calculation": DefaultParams.bmd_calculation,
        "dose_drop": DefaultParams.dose_drop,
        "bmr": DefaultParams.cont_bmr,
        "bmr_type": DefaultParams.cont_bmr_type,
        "confidence_level": DefaultParams.confidence_level,
        "constant_variance": DefaultParams.constant_variance,
    }

    def as_dfile(self):
        self._set_values()
        return "\n".join(
            [
                self._dfile_print_header_rows(),
                f"{self.dataset._BMDS_DATASET_TYPE} {self.dataset.dataset_length} 0",
                self._dfile_print_options(
                    "max_iterations",
                    "relative_fn_conv",
                    "parameter_conv",
                    "bmdl_curve_calculation",
                    "restrict_n",
                    "bmd_calculation",
                    "append_or_overwrite",
                    "smooth_option",
                ),
                self._dfile_print_options(
                    "bmr_type", "bmr", "constant_variance", "confidence_level"
                ),
                self._dfile_print_parameters("alpha", "rho", "intercept", "v", "n", "k"),
                self.dataset.as_dfile(),
            ]
        )

    def get_ys(self, xs):
        intercept = self._get_param("intercept")
        v = self._get_param("v")
        n = self._get_param("n")
        k = self._get_param("k")
        ys = intercept + (v * np.power(xs, n)) / (np.power(k, n) + np.power(xs, n))
        return ys
