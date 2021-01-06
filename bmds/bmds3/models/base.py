import ctypes
import logging
import platform
from typing import Dict, List, Optional, Union

from pydantic import BaseModel

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

    @property
    def has_output(self) -> bool:
        return self.results is not None

    def get_model_settings(self, dataset: DatasetType, settings: InputModelSettings) -> BaseModel:
        raise NotImplementedError("Requires abstract implementation")

    def execute(self) -> BaseModel:
        raise NotImplementedError("Requires abstract implementation")

    def execute_job(self):
        self.results = self.execute()

    def serialize(self) -> BaseModel:
        raise NotImplementedError("Requires abstract implementation")

    def to_dict(self) -> Dict:
        return self.serialize.dict()


class BaseModelAveraging:
    """
    Captures modeling configuration for model execution.
    Should save no results form model execution or any dataset-specific settings.
    """

    model_version = "BMDS330"

    def __init__(
        self,
        dataset: DatasetType,
        models: List[int],
        settings: InputModelSettings = None,
        results: Optional[BaseModel] = None,
    ):
        self.dataset = dataset
        self.models = models
        self.settings = self.get_model_settings(dataset, settings)
        self.results = results

    def get_model_settings(self, dataset: DatasetType, settings: InputModelSettings) -> BaseModel:
        raise NotImplementedError("Requires abstract implementation")

    def execute(self, session) -> BaseModel:
        raise NotImplementedError("Requires abstract implementation")

    def execute_job(self, session):
        self.results = self.execute(session)

    @property
    def has_output(self) -> bool:
        return self.results is not None

    def serialize(self, model_index: int) -> Dict:
        d = super().dict()
        d.update(
            model_index=model_index, has_output=self.has_output,
        )
        return d

    def to_dict(self) -> Dict:
        return self.serialize.dict()


class BmdModelSchema(BaseModel):
    pass
