from pathlib import Path

HERE = Path(__file__).parent.absolute()


class Config:
    DATA = (HERE / "data").absolute()
    DB_URI = f"sqlite:///{HERE}/benchmarks.db"
