from typing import Dict, Optional, Union

from ...datasets import Dataset


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
        self.settings = settings or {}
        self.execution_start = None
        self.execution_end = None
        self.execution_halted = False

    @property
    def output_created(self) -> bool:
        return self.execution_start is not None and self.execution_halted is False

    def execute(self):
        raise NotImplementedError("Requires abstract implementation")

    def execute_job(self):
        self.execute()
