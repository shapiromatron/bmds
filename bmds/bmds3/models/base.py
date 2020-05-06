import ctypes
import logging
import platform
from typing import Any, Callable, Dict, Optional, Tuple, Union

from ...datasets import Dataset
from ...utils import package_root

logger = logging.getLogger(__name__)


class BmdsFunctionManager:
    def __init__(self):
        raise RuntimeError("Use as a static-class")

    func_cache: Dict[Tuple[str, str, str], Callable] = {}

    @classmethod
    def get_dll_func(cls, bmds_version: str, base_name: str, func_name: str) -> Callable:
        """
        Return a callable function from a dll. The filename will be OS and environment specific.

        Args:
            bmds_version (str): The bmds version, eg., `BMDS312`
            base_name (str): The base-name for the file eg., `bmds_models`
            func_name (str): The callable function from the dll, eg., `run_cmodel`

        Raises:
            EnvironmentError: System could not be determined
            FileNotFoundError: The dll file could not be found

        Returns:
            Callable: the callable function from the dll
        """
        key = (bmds_version, base_name, func_name)
        func = cls.func_cache.get(key)

        if func is None:
            logger.info(f"Loading dll from disk: {key}")
            filename = base_name

            bits = platform.architecture()[0]
            if "64" in bits:
                filename += "_x64"
            elif "32" in bits:
                pass
            else:
                raise EnvironmentError(f"Unknown architecture: {bits}")

            os_ = platform.system()
            if os_ == "Windows":
                filename += ".dll"
            elif os_ in ("Darwin", "Linux"):
                filename += ".so"
            else:
                raise EnvironmentError(f"Unknown OS: {os_}")

            path = package_root / "bin" / bmds_version / filename
            if not path.exists():
                raise FileNotFoundError(f"Path does not exist: {path}")

            dll = ctypes.cdll.LoadLibrary(str(path))
            func = getattr(dll, func_name)

            cls.func_cache[key] = func

        return func


class BaseModel:
    """
    Captures modeling configuration for model execution.
    Should save no results form model execution or any dataset-specific settings.
    """

    def __init__(
        self,
        dataset: Dataset,
        settings: Optional[Dict] = None,
        id: Optional[Union[int, str]] = None,
    ):
        self.id = id
        self.dataset = dataset
        self.execution_start = None
        self.execution_end = None
        self.execution_halted = False
        self.settings = self.get_model_settings(settings or {})

    @property
    def output_created(self) -> bool:
        return self.execution_start is not None and self.execution_halted is False

    def get_model_settings(self, settings: Dict) -> Any:
        raise NotImplementedError("Requires abstract implementation")

    def execute(self):
        raise NotImplementedError("Requires abstract implementation")

    def execute_job(self):
        self.execute()
