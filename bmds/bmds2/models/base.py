import abc
import logging
import os
import subprocess
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import ClassVar

import numpy as np
from simple_settings import settings

from ... import constants, plotting
from ...utils import TempFileList
from ..parser import OutputParser

logger = logging.getLogger(__name__)
ROOT = Path(__file__).parents[2] / "bin"


class RunStatus(IntEnum):
    SUCCESS = 1
    FAILURE = 2
    DID_NOT_RUN = 3


class BMDModel(abc.ABC):
    """
    Parent class for individual BMDS models.

    The interface for all models is identical to the base BMDModel class, and
    is therefore documented here.

    Example
    -------

    >>> dataset = bmds.ContinuousDataset(
            doses=[0, 10, 50, 150, 400],
            ns=[25, 25, 24, 24, 24],
            means=[2.61, 2.81, 2.96, 4.66, 11.23],
            stdevs=[0.81, 1.19, 1.37, 1.72, 2.84]
        )
    >>> model = bmds.models.Polynomial_220(
            dataset,
            settings={"degree_poly": 3}
        )
    >>> model.execute()
    >>> model.output['BMD']
    88.3549
    """

    def __init__(self, dataset, settings=None, id=None):
        self.tempfiles = TempFileList()
        self.id = id
        self.dataset = dataset
        self.settings = settings or {}
        self.values = {}
        self.output_created = False
        self.execution_halted = False
        self.execution_start = None
        self.execution_end = None
        self.stdout = ""
        self.stderr = ""

    def _set_job_outputs(self, status, **kwargs):
        # If status is RunStatus.SUCCESS, then three required fields:
        #   - stdout, stderr, output
        execution_end = datetime.now()
        if status is RunStatus.SUCCESS:
            output_text = kwargs["output"]
            self.output_created = output_text is not None
            self.stdout = kwargs["stdout"]
            self.stderr = kwargs["stderr"]
            if self.output_created:
                self.outfile = output_text
                self.output = self.parse_outfile(output_text)

        elif status is RunStatus.FAILURE:
            self.execution_halted = True

        elif status is RunStatus.DID_NOT_RUN:
            pass

        self.execution_end = execution_end

        if hasattr(self, "output"):
            self.output.update(
                execution_start_time=self.execution_start.isoformat(),
                execution_end_time=execution_end.isoformat(),
                execution_duration=self.execution_duration,
            )

    def execute_job(self):
        """
        Execute the BMDS model and parse outputs if successful.
        """
        self.execution_start = datetime.now()

        # exit early if execution is not possible
        if not self.can_be_executed:
            return self._set_job_outputs(RunStatus.DID_NOT_RUN)

        exe = self.get_exe_path()
        dfile = self.write_dfile()
        outfile = self.get_outfile(dfile)
        o2 = outfile.replace(".out", ".002")

        try:
            proc = subprocess.run(
                [exe, dfile],  # noqa: S603
                capture_output=True,
                timeout=settings.BMDS_MODEL_TIMEOUT_SECONDS,
            )

            output = None
            if os.path.exists(outfile):
                with open(outfile) as f:
                    output = f.read()

            self._set_job_outputs(
                RunStatus.SUCCESS,
                stdout=proc.stdout.decode().strip(),
                stderr=proc.stderr.decode().strip(),
                output=output,
            )

        except subprocess.TimeoutExpired:
            self._set_job_outputs(RunStatus.FAILURE)

        finally:
            if os.path.exists(outfile):
                self.tempfiles.append(outfile)
            else:
                with open(dfile) as f:
                    txt = f.read()
                logger.info(f"Output file not created: \n{txt}\n")

            if os.path.exists(o2):
                self.tempfiles.append(o2)

            self.tempfiles.cleanup()

    def execute(self):
        self.execute_job()

    @property
    def execution_duration(self):
        """
        Returns total BMDS execution time, in seconds.
        """
        duration = None
        if self.execution_start and self.execution_end:
            delta = self.execution_end - self.execution_start
            duration = delta.total_seconds()
        return duration

    @classmethod
    def get_default(cls):
        """
        Return default parameters for this model.
        """
        return {"name": cls.model_name, "defaults": cls.defaults}

    @classmethod
    def get_exe_path(cls):
        """
        Return the full path to the executable.
        """
        return os.path.abspath(os.path.join(ROOT, cls.bmds_version_dir, cls.exe + ".exe"))

    @property
    def can_be_executed(self):
        return self.dataset.num_dose_groups >= self.minimum_dose_groups

    @property
    def has_successfully_executed(self):
        """
        Check if model has successful completed.
        """
        return hasattr(self, "outfile")

    @property
    def name(self):
        """
        Return the model name, and degree of polynomial for some models.
        """
        return self.model_name

    def get_outfile(self, dfile):
        return dfile.replace(".(d)", ".out")

    def parse_outfile(self, outfile):
        # returned parsed output dictionary
        try:
            parser = OutputParser(outfile, self.dtype, self.model_name)
        except Exception as err:
            logger.error(f"Parsing failed: \n\n{self.as_dfile()}\n\n{outfile}\n")
            raise err
        return parser.output

    @abc.abstractmethod
    def as_dfile(self):
        """
        Represent this model in the BMDS (d) file input representation.
        """
        ...

    def plot(self):
        """
        After model execution, print the dataset, curve-fit, BMD, and BMDL.

        Example
        -------

        >>> import os
        >>> fn = os.path.expanduser('~/Desktop/image.png')
        >>> fig = model.plot()
        >>> fig.savefig(fn)
        >>> fig.clear()

        .. figure:: ../tests/data/mpl/test_exponential_m4_plot.png
           :scale: 80%
           :align: center
           :alt: Example generated BMD plot

           BMD models can generate plots using the ``plot()`` method; an example
           is shown here.

        """
        fig = self.dataset.plot()
        ax = fig.gca()
        ax.set_title(f"{self.dataset._get_dataset_name()}\n{self.name}, {self.get_bmr_text()}")
        if self.has_successfully_executed:
            self._set_x_range(ax)
            ax.plot(self._xs, self.get_ys(self._xs), label=self.name, **plotting.LINE_FORMAT)
            self._add_bmr_lines(ax)
        else:
            self._add_plot_failure(ax)

        ax.legend(**plotting.LEGEND_OPTS)

        return fig

    @abc.abstractmethod
    def get_ys(self, xs):
        ...

    @abc.abstractmethod
    def get_bmr_text(self):
        ...

    def _add_bmr_lines(self, ax):
        # add BMD and BMDL lines to plot.
        LINE_FORMAT = dict(c="#BFC05D", lw=2, zorder=60)
        bmd = self.output["BMD"]
        bmdl = self.output["BMDL"]
        xdomain = ax.xaxis.get_view_interval()
        ydomain = ax.yaxis.get_view_interval()
        xrng = xdomain[1] - xdomain[0]
        yrng = ydomain[1] - ydomain[0]
        ys = self.get_ys(np.array([bmd, bmdl]))

        ax.axhline(
            ys[0],
            xmin=0,
            xmax=(bmd - xdomain[0]) / xrng,
            label="BMR, BMD, BMDL",
            **LINE_FORMAT,
        )
        ax.axvline(bmd, ymin=0, ymax=(ys[0] - ydomain[0]) / yrng, **LINE_FORMAT)
        ax.axvline(bmdl, ymin=0, ymax=(ys[0] - ydomain[0]) / yrng, **LINE_FORMAT)
        ax.text(
            bmd + xrng * 0.01,
            ydomain[0] + yrng * 0.02,
            "BMD",
            horizontalalignment="left",
            **plotting.BMD_LABEL_FORMAT,
        )
        ax.text(
            bmdl - xrng * 0.01,
            ydomain[0] + yrng * 0.02,
            "BMDL",
            horizontalalignment="right",
            **plotting.BMD_LABEL_FORMAT,
        )

    def _add_plot_failure(self, ax):
        ax.text(
            0.5,
            0.8,
            "ERROR: model cannot be plotted",
            transform=ax.transAxes,
            **plotting.FAILURE_MESSAGE_FORMAT,
        )

    def _set_x_range(self, ax):
        bmd = max(0, self.output["BMD"])
        bmdl = max(0, self.output["BMDL"])
        doses = self.dataset.doses

        # set values for what we'll need to calculate model curve
        min_x = min(bmd, bmdl, *doses)
        max_x = max(bmd, bmdl, *doses)
        self._xs = np.linspace(max(min_x, 1e-9), max_x, 100)

        # add a little extra padding on plot
        padding = plotting.PLOT_MARGINS * (max_x - min_x)
        ax.set_xlim(min_x - padding, max_x + padding)

    def write_dfile(self):
        """
        Write the generated d_file to a temporary file.
        """
        f_in = self.tempfiles.get_tempfile(prefix="bmds-", suffix=".(d)")
        with open(f_in, "w") as f:
            f.write(self.as_dfile())
        return f_in

    def _get_param(self, key):
        return self.output["parameters"][key]["estimate"]

    def _set_values(self):
        self.values = {}
        for k in self.defaults.keys():
            self.values[k] = self._get_option_value(k)

    def _get_option_value(self, key):
        """
        Get option value(s), or use default value if no override value.

        Two output values for 'p' type values (parameters), else one.
        Returns a single value or tuple of two values
        """
        fn_name = f"set_{key}_value"
        if key in self.settings:
            val = self.settings[key]
        elif hasattr(self, fn_name):
            val = getattr(self, fn_name)()
        else:
            val = self.defaults[key]["d"]

        if self.defaults[key]["t"] == constants.FT_PARAM:
            return val.split("|")
        else:
            return val

    def _dfile_print_header_rows(self):
        return f"{self.model_name}\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out"

    def _dfile_print_parameters(self, *params):
        # Print parameters in the specified order. Expects a tuple of parameter
        # names, in the proper order.
        if (self.dtype in constants.CONTINUOUS_DTYPES) and (self.values["constant_variance"] == 1):
            self.values["rho"] = ("s", 0)  # for specified to equal 0
        specifieds = []
        initials = []
        init = "0"  # 1 if initialized, 0 otherwise
        for param in params:
            t, v = self.values[param]
            # now add values
            if t == constants.P_DEFAULT:
                specifieds.append(-9999)
                initials.append(-9999)
            elif t == constants.P_SPECIFIED:
                specifieds.append(v)
                initials.append(-9999)
            elif t == constants.P_INITIALIZED:
                init = "1"
                specifieds.append(-9999)
                initials.append(v)
            else:
                raise ValueError("Unknown parameter specification")

        return "\n".join(
            [" ".join([str(i) for i in specifieds]), init, " ".join([str(i) for i in initials])]
        )

    def _dfile_print_options(self, *params):
        # Return space-separated list of values for dfile
        return " ".join([str(self.values[param]) for param in params])

    def _get_model_name(self):
        return self.exe

    def to_dict(self, model_index):
        """
        Return a summary of the model in a dictionary format for serialization.

        Parameters
        ----------
        model_index : int
            The index of the model in a list of models, should be unique

        Returns
        -------
        out : dictionary
            A dictionary of model inputs, and raw and parsed outputs
        """
        return dict(
            name=self.name,
            model_index=model_index,
            model_name=self.model_name,
            model_version=self.version,
            has_output=self.output_created,
            dfile=self.as_dfile(),
            execution_halted=self.execution_halted,
            stdout=self.stdout,
            stderr=self.stderr,
            outfile=getattr(self, "outfile", None),
            output=getattr(self, "output", None),
            logic_bin=getattr(self, "logic_bin", None),
            logic_notes=getattr(self, "logic_notes", None),
            recommended=getattr(self, "recommended", None),
            recommended_variable=getattr(self, "recommended_variable", None),
        )

    def _to_df(self, d, idx, show_null):
        def _nullify(show_null, value):
            return constants.NULL if show_null else value

        d["model_name"].append(_nullify(show_null, self.name))
        d["model_index"].append(_nullify(show_null, idx))
        d["model_version"].append(_nullify(show_null, self.version))
        d["has_output"].append(_nullify(show_null, self.output_created))
        d["execution_halted"].append(getattr(self, "execution_halted", None))

        # add model outputs
        outputs = {} if show_null else getattr(self, "output", {})

        d["BMD"].append(outputs.get("BMD", "-"))
        d["BMDL"].append(outputs.get("BMDL", "-"))
        d["BMDU"].append(outputs.get("BMDU", "-"))
        d["CSF"].append(outputs.get("CSF", "-"))
        d["AIC"].append(outputs.get("AIC", "-"))
        d["pvalue1"].append(outputs.get("p_value1", "-"))
        d["pvalue2"].append(outputs.get("p_value2", "-"))
        d["pvalue3"].append(outputs.get("p_value3", "-"))
        d["pvalue4"].append(outputs.get("p_value4", "-"))
        d["Chi2"].append(outputs.get("Chi2", "-"))
        d["df"].append(outputs.get("df", "-"))
        d["residual_of_interest"].append(outputs.get("residual_of_interest", "-"))
        d["warnings"].append("\n".join(outputs.get("warnings", ["-"])))

        # add logic bin and warnings
        logics = getattr(self, "logic_notes", {})
        d["logic_bin"].append(_nullify(show_null, self.get_logic_bin_text()))

        txt = "\n".join(logics.get(constants.BIN_NO_CHANGE, ["-"]))
        d["logic_cautions"].append(_nullify(show_null, txt))
        txt = "\n".join(logics.get(constants.BIN_WARNING, ["-"]))
        d["logic_warnings"].append(_nullify(show_null, txt))
        txt = "\n".join(logics.get(constants.BIN_FAILURE, ["-"]))
        d["logic_failures"].append(_nullify(show_null, txt))

        # add recommendation and recommendation variable
        txt = getattr(self, "recommended", "-")
        d["recommended"].append(_nullify(show_null, txt))
        txt = getattr(self, "recommended_variable", "-")
        d["recommended_variable"].append(_nullify(show_null, txt))

        # add verbose outputs if specified
        if "dfile" in d:
            txt = self.as_dfile()
            d["dfile"].append(_nullify(show_null, txt))
        if "outfile" in d:
            txt = getattr(self, "outfile", "-")
            d["outfile"].append(_nullify(show_null, txt))
        if "stdout" in d:
            txt = getattr(self, "stdout", "-")
            d["stdout"].append(_nullify(show_null, txt))
        if "stderr" in d:
            txt = getattr(self, "stderr", "-")
            d["stderr"].append(_nullify(show_null, txt))

    def get_logic_bin_text(self):
        return constants.BIN_TEXT[self.logic_bin] if hasattr(self, "logic_bin") else "-"


class DefaultParams:
    """
    Container to store default modeling input parameters.

    Key crosswalk:
     - c = category
     - t = field type
     - d = default
     - n = name (optional)
    """

    bmdl_curve_calculation: ClassVar = {
        "c": constants.FC_OTHER,
        "t": constants.FT_BOOL,
        "d": 0,
        "n": "BMDL curve calculation",
    }
    append_or_overwrite: ClassVar = {"c": constants.FC_OTHER, "t": constants.FT_BOOL, "d": 0}
    smooth_option: ClassVar = {"c": constants.FC_OTHER, "t": constants.FT_BOOL, "d": 0}
    bmd_calculation: ClassVar = {
        "c": constants.FC_OTHER,
        "t": constants.FT_BOOL,
        "d": 1,
        "n": "BMD calculation",
    }
    dose_drop: ClassVar = {
        "c": constants.FC_OTHER,
        "t": constants.FT_DROPDOSE,
        "d": 0,
        "n": "Doses to drop",
    }
    constant_variance: ClassVar = {
        "c": constants.FC_OTHER,
        "t": constants.FT_BOOL,
        "d": 1,
        "n": "Constant variance",
    }
    max_iterations: ClassVar = {
        "c": constants.FC_OPTIMIZER,
        "t": constants.FT_INTEGER,
        "d": 500,
        "n": "Iteration",
    }
    relative_fn_conv: ClassVar = {
        "c": constants.FC_OPTIMIZER,
        "t": constants.FT_DECIMAL,
        "d": 1.0e-08,
        "n": "Relative function",
    }
    parameter_conv: ClassVar = {
        "c": constants.FC_OPTIMIZER,
        "t": constants.FT_DECIMAL,
        "d": 1.0e-08,
        "n": "Parameter",
    }
    confidence_level: ClassVar = {"c": constants.FC_BMR, "t": constants.FT_DECIMAL, "d": 0.95}
    dich_bmr: ClassVar = {"c": constants.FC_BMR, "t": constants.FT_INTEGER, "d": 0.1}
    dich_bmr_type: ClassVar = {"c": constants.FC_BMR, "t": constants.FT_DECIMAL, "d": 0}
    cont_bmr: ClassVar = {"c": constants.FC_BMR, "t": constants.FT_DECIMAL, "d": 1.0}
    cont_bmr_type: ClassVar = {"c": constants.FC_BMR, "t": constants.FT_INTEGER, "d": 1}

    @staticmethod
    def degree_poly(d=2, showName=True):
        d = {"c": constants.FC_OTHER, "t": constants.FT_INTEGER, "d": d}
        if showName:
            d["n"] = "Degree of polynomial"
        return d

    @staticmethod
    def log_transform(d):
        return {"c": constants.FC_OTHER, "t": constants.FT_BOOL, "d": d}

    @staticmethod
    def param_generator(name=None):
        d = {"c": constants.FC_PARAM, "t": constants.FT_PARAM, "d": "d|"}
        if name:
            d["n"] = name
        return d

    @staticmethod
    def restrict(d, n=None):
        d = {"c": constants.FC_OTHER, "t": constants.FT_BOOL, "d": d}
        if n:
            d["n"] = n
        return d
