from pathlib import Path

HERE = Path(__file__).parent.absolute()

ROOT_DIR = HERE.parents[2].absolute()


class Config:
    DATA = (HERE / "data").absolute()
    DB_URI = f"sqlite:///{ROOT_DIR}/benchmarks.db"
