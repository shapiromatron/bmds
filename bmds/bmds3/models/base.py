import ctypes
import logging
import platform
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from ...datasets import DatasetBase
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
            dll = ctypes.cdll.LoadLibrary(str(path))
            cls._dll_cache[key] = dll

        return dll


InputModelSettings = Optional[Union[Dict, BaseModel]]


class BmdModel(BaseModel):
    """
    Captures modeling configuration for model execution.
    Should save no results form model execution or any dataset-specific settings.
    """

    model_version: str = "BMDS330"

    id: Optional[Any]
    dataset: DatasetBase
    settings: BaseModel
    results: Optional[BaseModel]
    inputs_struct: Optional[ctypes.Structure]  # used for model averaging
    fit_results_struct: Optional[ctypes.Structure]  # used for model averaging

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True

    def __init__(
        self,
        dataset: DatasetBase,
        settings: InputModelSettings = None,
        id: Optional[Union[int, str]] = None,
        results: Optional[BaseModel] = None,
    ):
        super().__init__(
            id=id,
            dataset=dataset,
            settings=self.get_model_settings(dataset, settings),
            results=results,
        )

    @property
    def bmd_model_class(self) -> Any:
        raise NotImplementedError()

    @property
    def has_output(self) -> bool:
        return self.results is not None

    def get_model_settings(self, dataset: DatasetBase, settings: InputModelSettings) -> BaseModel:
        raise NotImplementedError("Requires abstract implementation")

    def execute(self) -> BaseModel:
        raise NotImplementedError("Requires abstract implementation")

    def execute_job(self):
        self.results = self.execute()

    def dict(self, model_index: int) -> Dict:
        """
        Return a summary of the model in a dictionary format for serialization.

        Args:
            model_index (int): numeric model index in a list of models, should be unique

        Returns:
            A dictionary of model inputs, and raw and parsed outputs
        """
        d = super().dict(exclude={"inputs_struct", "fit_results_struct"})
        d.update(
            bmd_model_class=self.bmd_model_class.dict(),
            model_index=model_index,
            has_output=self.has_output,
        )
        return d


class BaseModelAveraging(BaseModel):
    """
    Captures modeling configuration for model execution.
    Should save no results form model execution or any dataset-specific settings.
    """

    model_version = "BMDS330"

    dataset: Any
    models: List[int]
    settings: InputModelSettings
    id: Optional[Any]
    results: Optional[BaseModel]

    def __init__(
        self,
        dataset: Any,
        models: List[int],
        settings: InputModelSettings = None,
        id: Optional[Union[int, str]] = None,
        results: Optional[BaseModel] = None,
    ):
        super().__init__(
            dataset=dataset,
            models=models,
            settings=self.get_model_settings(dataset, settings),
            id=id,
            results=results,
        )

    def get_model_settings(self, dataset: DatasetBase, settings: InputModelSettings) -> BaseModel:
        raise NotImplementedError("Requires abstract implementation")

    def execute(self, session) -> BaseModel:
        raise NotImplementedError("Requires abstract implementation")

    def execute_job(self, session):
        self.results = self.execute(session)

    @property
    def has_output(self) -> bool:
        return self.results is not None

    def dict(self, model_index: int) -> Dict:
        d = super().dict()
        d.update(
            model_index=model_index, has_output=self.has_output,
        )
        return d
