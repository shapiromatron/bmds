from __future__ import annotations

import abc
import ctypes
import logging
import platform
from typing import TYPE_CHECKING, ClassVar, NamedTuple, Self

from pydantic import BaseModel

from ... import plotting
from ...constants import CONTINUOUS_DTYPES, DICHOTOMOUS_DTYPES, Dtype
from ...datasets import DatasetType
from ...utils import multi_lstrip, package_root
from ..constants import BmdModelSchema as BmdModelClass
from ..types.priors import priors_tbl

if TYPE_CHECKING:
    from ..sessions import BmdsSession


logger = logging.getLogger(__name__)


class BmdsLibraryManager:
    """
    Cache for managing dll libraries
    """

    def __init__(self):
        raise RuntimeError("Use as a static-class")

    _dll_cache: ClassVar[dict[str, ctypes.CDLL]] = {}

    @classmethod
    def get_dll(cls, bmds_version: str, base_name: str) -> ctypes.CDLL:
        """
        Returns a dll instance. The filename will be OS and environment specific.

        Args:
            bmds_version (str): The bmds version, eg., `BMDS330`
            base_name (str): The base-name for the file eg., `bmds_models`

        Raises:
            EnvironmentError: System could not be determined
            FileNotFoundError: The dll file could not be found

        Returns:
            Callable: the callable function from the dll
        """

        filename = base_name
        os_ = platform.system()
        if os_ == "Windows":
            filename += ".dll"
        elif os_ == "Linux":
            filename += ".so"
        elif os_ == "Darwin":
            filename += ".dylib"
        else:
            raise OSError(f"Unknown OS: {os_}")

        path = package_root / "bin" / bmds_version / filename
        key = str(path)
        dll = cls._dll_cache.get(key)
        if dll is None:
            if not path.exists():
                raise FileNotFoundError(f"Path does not exist: {path}")

            logger.info(f"Loading dll from disk: {key}")
            dll = ctypes.cdll.LoadLibrary(str(path))
            cls._dll_cache[key] = dll

        return dll


InputModelSettings = dict | BaseModel | None


class BmdModel(abc.ABC):
    """
    Captures modeling configuration for model execution.
    Should save no results form model execution or any dataset-specific settings.
    """

    bmd_model_class: BmdModelClass
    model_version: str
    degree_required: bool = False

    def __init__(self, dataset: DatasetType, settings: InputModelSettings = None):
        self.dataset = dataset
        self.settings = self.get_model_settings(dataset, settings)
        self.structs: NamedTuple | None = None  # used for model averaging
        self.results: BaseModel | None = None

    def name(self) -> str:
        # return name of model; may be setting-specific
        return self.bmd_model_class.verbose

    @classmethod
    def get_dll(cls) -> ctypes.CDLL:
        return BmdsLibraryManager.get_dll(bmds_version=cls.model_version, base_name="libDRBMD")

    @property
    def has_results(self) -> bool:
        return self.results is not None and self.results.has_completed is True

    @abc.abstractmethod
    def get_model_settings(self, dataset: DatasetType, settings: InputModelSettings) -> BaseModel:
        ...

    @abc.abstractmethod
    def execute(self) -> BaseModel:
        ...

    def execute_job(self):
        self.execute()

    @abc.abstractmethod
    def serialize(self) -> BaseModel:
        ...

    def text(self) -> str:
        """Text representation of model inputs and outputs outputs."""
        title = self.name().center(20) + "\n════════════════════"
        settings = self.model_settings_text()
        if self.has_results:
            results = self.results.text(self.dataset, self.settings)
        else:
            results = "Model has not successfully executed; no results available."

        return "\n\n".join([title, settings, results]) + "\n"

    def model_settings_text(self) -> str:
        input_tbl = self.settings.tbl(self.degree_required)
        prior_tbl = priors_tbl(
            self.get_param_names(), self.get_priors_list(), self.settings.priors.is_bayesian
        )
        return multi_lstrip(
            f"""
        Input Summary:
        {input_tbl}

        Parameter Settings:
        {prior_tbl}
        """
        )

    def plot(self):
        """
        After model execution, print the dataset, curve-fit, BMD, and BMDL.
        """
        if not self.has_results:
            raise ValueError("Cannot plot if results are unavailable")

        fig = self.dataset.plot()
        ax = fig.gca()
        if self.dataset.dtype in DICHOTOMOUS_DTYPES:
            ax.set_ylim(-0.05, 1.05)
        title = f"{self.dataset._get_dataset_name()}\n{self.name()}, {self.settings.bmr_text}"
        ax.set_title(title)
        ax.plot(
            self.results.plotting.dr_x,
            self.results.plotting.dr_y,
            label=f"{self.name()} (BMD, BMDL, BMDU)",
            **plotting.LINE_FORMAT,
        )
        plotting.add_bmr_lines(
            ax, self.results.bmd, self.results.plotting.bmd_y, self.results.bmdl, self.results.bmdu
        )
        ax.legend(**plotting.LEGEND_OPTS)

        # reorder handles and labels
        handles, labels = ax.get_legend_handles_labels()
        order = [1, 0]
        ax.legend(
            [handles[idx] for idx in order], [labels[idx] for idx in order], **plotting.LEGEND_OPTS
        )

        return fig

    def cdf_plot(self):
        if not self.has_results:
            raise ValueError("Cannot plot if results are unavailable")
        fig = plotting.create_empty_figure()
        ax = fig.gca()
        ax.set_xlabel(self.dataset.get_xlabel())
        ax.set_ylabel("Percentile")
        ax.plot(
            self.results.fit.bmd_dist[0],
            self.results.fit.bmd_dist[1],
            **plotting.LINE_FORMAT,
        )
        ax.set_title("BMD cumulative distribution function")
        return fig

    @abc.abstractmethod
    def get_param_names(self) -> list[str]:
        ...

    @abc.abstractmethod
    def get_priors_list(self) -> list[list]:
        ...

    def to_dict(self) -> dict:
        return self.serialize().model_dump()

    @abc.abstractmethod
    def get_gof_pvalue(self) -> float:
        ...


class BmdModelSchema(BaseModel):
    @classmethod
    def get_subclass(cls, dtype: Dtype) -> Self:
        from .continuous import BmdModelContinuousSchema
        from .dichotomous import BmdModelDichotomousSchema

        if dtype in DICHOTOMOUS_DTYPES:
            return BmdModelDichotomousSchema
        elif dtype in CONTINUOUS_DTYPES:
            return BmdModelContinuousSchema
        else:
            raise ValueError(f"Invalid dtype: {dtype}")


class BmdModelAveraging(abc.ABC):
    """
    Captures modeling configuration for model execution.
    Should save no results form model execution or any dataset-specific settings.
    """

    model_version: str

    def __init__(
        self,
        session: BmdsSession,
        models: list[BmdModel],
        settings: InputModelSettings = None,
    ):
        self.session = session
        self.models = models
        # if not settings are not specified copy settings from first model
        initial_settings = settings if settings is not None else models[0].settings
        self.settings = self.get_model_settings(initial_settings)
        self.results: BaseModel | None = None

    def get_dll(self) -> ctypes.CDLL:
        return BmdsLibraryManager.get_dll(bmds_version=self.model_version, base_name="libDRBMD")

    @abc.abstractmethod
    def get_model_settings(self, settings: InputModelSettings) -> BaseModel:
        ...

    @abc.abstractmethod
    def execute(self) -> BaseModel:
        ...

    def execute_job(self):
        self.results = self.execute()

    @property
    def has_results(self) -> bool:
        return self.results is not None

    @abc.abstractmethod
    def serialize(self, session) -> BmdModelAveragingSchema:
        ...

    @abc.abstractmethod
    def plot(self):
        ...

    def to_dict(self) -> dict:
        return self.serialize.dict()


class BmdModelAveragingSchema(BaseModel):
    @classmethod
    def get_subclass(cls, dtype: Dtype) -> Self:
        from .ma import BmdModelAveragingDichotomousSchema

        if dtype in (Dtype.DICHOTOMOUS, Dtype.DICHOTOMOUS_CANCER):
            return BmdModelAveragingDichotomousSchema
        else:
            raise ValueError(f"Invalid dtype: {dtype}")
