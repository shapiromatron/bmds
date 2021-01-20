import ctypes
import logging
import platform
from typing import Dict, List, Optional, Union

from pydantic import BaseModel

from ... import plotting
from ...constants import CONTINUOUS_DTYPES, DICHOTOMOUS_DTYPES, Dtype
from ...datasets import DatasetType
from ...utils import package_root
from ..constants import BmdModelSchema

logger = logging.getLogger(__name__)


class BmdsLibraryManager:
    """
    Cache for managing dll libraries
    """

    def __init__(self):
        raise RuntimeError("Use as a static-class")

    _dll_cache: Dict[str, ctypes.CDLL] = {}

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
            raise EnvironmentError(f"Unknown OS: {os_}")

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


InputModelSettings = Optional[Union[Dict, BaseModel]]


class BmdModel:
    """
    Captures modeling configuration for model execution.
    Should save no results form model execution or any dataset-specific settings.
    """

    bmd_model_class: BmdModelSchema
    model_version: str

    def __init__(self, dataset: DatasetType, settings: InputModelSettings = None):
        self.dataset = dataset
        self.settings = self.get_model_settings(dataset, settings)
        self.results: Optional[BaseModel] = None
        self.inputs_struct: Optional[ctypes.Structure] = None  # used for model averaging
        self.fit_results_struct: Optional[ctypes.Structure] = None  # used for model averaging

    def name(self) -> str:
        # return name of model; may be setting-specific
        return self.bmd_model_class.verbose

    @property
    def has_results(self) -> bool:
        return self.results is not None

    def get_model_settings(self, dataset: DatasetType, settings: InputModelSettings) -> BaseModel:
        raise NotImplementedError("Requires abstract implementation")

    def execute(self) -> BaseModel:
        raise NotImplementedError("Requires abstract implementation")

    def execute_job(self):
        self.results = self.execute()

    def serialize(self) -> BaseModel:
        raise NotImplementedError("Requires abstract implementation")

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
        ax.set_title(f"{self.dataset._get_dataset_name()}\n{self.name()}, ADD BMR")
        ax.plot(self.results.dr_x, self.results.dr_y, label=self.name(), **plotting.LINE_FORMAT)
        self._add_bmr_lines(ax)
        ax.legend(**plotting.LEGEND_OPTS)
        return fig

    def _add_bmr_lines(self, ax):
        res = self.results
        xdomain = ax.xaxis.get_view_interval()
        xrng = xdomain[1] - xdomain[0]

        if res.bmd > 0:
            ax.plot([0, res.bmd], [res.bmd_y, res.bmd_y], **plotting.BMD_LINE_FORMAT)
            ax.text(
                res.bmd + xrng * 0.01,
                0,
                "BMD",
                label="BMR, BMD, BMDL",
                horizontalalignment="left",
                verticalalignment="center",
                **plotting.BMD_LABEL_FORMAT,
            )

        if res.bmdl > 0:
            ax.plot([res.bmdl, res.bmdl], [0, res.bmd_y], **plotting.BMD_LINE_FORMAT)
            ax.text(
                res.bmdl - xrng * 0.01,
                0,
                "BMDL",
                horizontalalignment="right",
                verticalalignment="center",
                **plotting.BMD_LABEL_FORMAT,
            )

        if res.bmd > 0 and res.bmdl > 0:
            ax.plot(
                [res.bmd, res.bmd], [0, res.bmd_y], **plotting.BMD_LINE_FORMAT,
            )

    def to_dict(self) -> Dict:
        return self.serialize.dict()


class BmdModelSchema(BaseModel):
    @classmethod
    def get_subclass(cls, dtype: Dtype) -> "BmdModelSchema":
        from .continuous import BmdModelContinuousSchema
        from .dichotomous import BmdModelDichotomousSchema

        if dtype in DICHOTOMOUS_DTYPES:
            return BmdModelDichotomousSchema
        elif dtype in CONTINUOUS_DTYPES:
            return BmdModelContinuousSchema
        else:
            raise ValueError(f"Invalid dtype: {dtype}")


class BmdModelAveraging:
    """
    Captures modeling configuration for model execution.
    Should save no results form model execution or any dataset-specific settings.
    """

    model_version = "BMDS330"

    def __init__(
        self, dataset: DatasetType, models: List[BmdModel], settings: InputModelSettings = None
    ):
        self.dataset = dataset
        self.models = models
        self.settings = self.get_model_settings(dataset, settings)
        self.results: Optional[BaseModel] = None

    def get_model_settings(self, dataset: DatasetType, settings: InputModelSettings) -> BaseModel:
        raise NotImplementedError("Requires abstract implementation")

    def execute(self) -> BaseModel:
        raise NotImplementedError("Requires abstract implementation")

    def execute_job(self):
        self.results = self.execute()

    @property
    def has_results(self) -> bool:
        return self.results is not None

    def serialize(self, session) -> "BmdModelAveragingSchema":
        raise NotImplementedError("Requires abstract implementation")

    def to_dict(self) -> Dict:
        return self.serialize.dict()


class BmdModelAveragingSchema(BaseModel):
    @classmethod
    def get_subclass(cls, dtype: Dtype) -> "BmdModelAveragingSchema":
        from .ma import BmdModelAveragingDichotomousSchema

        if dtype in (Dtype.DICHOTOMOUS, Dtype.DICHOTOMOUS_CANCER):
            return BmdModelAveragingDichotomousSchema
        elif dtype in (Dtype.CONTINUOUS, Dtype.CONTINUOUS_INDIVIDUAL):
            raise NotImplementedError()
        else:
            raise ValueError(f"Invalid dtype: {dtype}")
