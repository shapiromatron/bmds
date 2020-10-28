import ctypes
import logging
import platform
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel as PydanticBaseModel

from ...datasets import Dataset
from ...utils import package_root

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
            filename = base_name
            dll = ctypes.cdll.LoadLibrary(str(path))
            cls._dll_cache[key] = dll

        return dll


InputModelSettings = Optional[Union[Dict, PydanticBaseModel]]


class BaseModel:
    """
    Captures modeling configuration for model execution.
    Should save no results form model execution or any dataset-specific settings.
    """

    model: Any
    model_version = "BMDS330"

    def __init__(
        self,
        dataset: Dataset,
        settings: InputModelSettings = None,
        id: Optional[Union[int, str]] = None,
    ):
        self.id = id
        self.dataset = dataset
        self.execution_start = None
        self.execution_end = None
        self.execution_halted = False
        self.settings = self.get_model_settings(settings)
        self.results = None

    @property
    def output_created(self) -> bool:
        return self.execution_start is not None and self.execution_halted is False

    def get_model_settings(self, settings: InputModelSettings) -> PydanticBaseModel:
        raise NotImplementedError("Requires abstract implementation")

    def execute(self) -> PydanticBaseModel:
        raise NotImplementedError("Requires abstract implementation")

    def execute_job(self):
        self.results = self.execute()

    def to_dict(self, model_index: int) -> Dict:
        """
        Return a summary of the model in a dictionary format for serialization.

        Args:
            model_index (int): numeric model index in a list of models, should be unique

        Returns:
            A dictionary of model inputs, and raw and parsed outputs
        """
        return dict(
            model_index=model_index,
            model_class=self.model.name,
            model_name=self.model.data.verbose,
            model_version=self.model_version,
            has_output=self.output_created,
            execution_halted=self.execution_halted,
            settings=self.settings.dict(),
            results=self.results.dict() if self.results else None,
        )
