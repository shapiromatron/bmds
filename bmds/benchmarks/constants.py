from enum import Enum


class BenchmarkDataset(str, Enum):
    TOXREFDB_DICH = "TOXREFDB_DICH"
    TOXREFDB_CONT = "TOXREFDB_CONT"


class BenchmarkVersion(str, Enum):
    BMDS270 = "bmds270"
    BMDS330 = "bmds330"
