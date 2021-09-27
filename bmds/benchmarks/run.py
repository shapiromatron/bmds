from .datasets import BenchmarkDataset


def run_analysis(dataset: BenchmarkDataset, versions: list[BenchmarkVersion], clear_existing: bool):

    if not dataset.data_loaded():
        dataset.load_data()

    for version in versions:
        print(len(DichotomousResult.query.all()))
        if clear_existing:
            if dataset.value is Dataset.TOXREFDB_CONT.value:
                ContinuousResult.query.filter(ContinuousResult.bmds_version == version).delete()
            if dataset.value is Dataset.TOXREFDB_DICH.value:
                DichotomousResult.query.filter(DichotomousResult.bmds_version == version).delete()
                print(len(DichotomousResult.query.all()))
        if dataset.value is Dataset.TOXREFDB_CONT.value:
            runContinuousModels(version.value)
        if dataset.value is Dataset.TOXREFDB_DICH.value:
            runDichotomousModels(version.value)
            print(len(DichotomousResult.query.all()))
