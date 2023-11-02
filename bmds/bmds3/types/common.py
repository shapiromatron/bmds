import inspect
from typing import Any

import numpy as np
from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from ... import bmdscore
from ..constants import BMDS_BLANK_VALUE


def list_t_c(list: list[Any], ctype):
    return (ctype * len(list))(*list)


def residual_of_interest(bmd: float, doses: list[float], residuals: list[float]) -> float:
    if bmd <= 0:
        return BMDS_BLANK_VALUE
    diffs = [abs(bmd - dose) for dose in doses]
    index = diffs.index(min(diffs))
    return residuals[index]


def clean_array(arr: np.ndarray) -> np.ndarray:
    return np.nan_to_num(
        arr, nan=BMDS_BLANK_VALUE, posinf=BMDS_BLANK_VALUE, neginf=BMDS_BLANK_VALUE
    )


class PydanticNumpyArray(np.ndarray):
    # pydantic friendly numpy arrays

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls.validate,
            handler("list"),
            serialization=core_schema.plain_serializer_function_ser_schema(lambda x: x.tolist()),
        )


class NumpyIntArray(PydanticNumpyArray):
    @classmethod
    def validate(cls, v):
        try:
            return np.asarray(v, dtype="int")
        except TypeError:
            raise ValueError("invalid np.ndarray format")


class NumpyFloatArray(PydanticNumpyArray):
    # Numpy arrays, agumented
    @classmethod
    def validate(cls, v):
        try:
            return np.asarray(v, dtype="float")
        except TypeError:
            raise ValueError("invalid np.ndarray format")


def inspect_cpp_obj(lines: list[str], obj: Any, depth: int):
    """Recursively inspect a C++ object.

    Append attributes to the a list of strings, which can be
    transformed into a string representation of the object.

    Args:
        lines (list[str]): a list of strings to append to
        obj (Any): the object to inspect
        depth (int): current depth of recursion
    """
    indent = "  " * depth + "- "
    lines.append(f"{indent}{obj.__class__.__name__}")
    depth += 1
    indent = "  " * depth + "- "
    for attr, value in inspect.getmembers(obj):
        if attr.startswith("__"):
            continue
        elif "bmdscore" in value.__class__.__module__:
            if isinstance(value, bmdscore.cont_model):
                lines.append(f"{indent}{attr}: {value}")
            else:
                inspect_cpp_obj(lines, value, depth)
        else:
            lines.append(f"{indent}{attr}: {value}")
