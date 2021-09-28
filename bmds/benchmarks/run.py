from ..constants import Version
from .analyses import BenchmarkAnalyses
from .datasets import BenchmarkDataset


def run_analysis(
    dataset: BenchmarkDataset,
    analysis: BenchmarkAnalyses,
    version: Version,
    clear_existing: bool = False,
):
    """Run an analysis for a given combination of datataset, analysis, and BMDS version.

    Args:
        dataset (BenchmarkDataset): The benchmark dataset to use
        analysis (BenchmarkAnalyses): The analysis to perform
        version (Version): The version of BMDS to use
        clear_existing (bool, default False): If True, delete prior results for this
            (dataset, analysis, version) combination
    """
    if not dataset.data_loaded():
        dataset.load_data()

    if clear_existing:
        analysis.executor.clear_results(dataset, version)

    analysis.executor.execute(dataset, version)
