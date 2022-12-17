# Benchmarks

Benchmarking can be helpful to better understand the impact of code changes to bmds. The bmds package as a built-in benchmarks subpackage which can run routine analyses and datasets. Results from executing benchmarks are saved in a sqlite database for detailed analysis, and also to allow for future comparisons to be made to the existing analysis.

To install the additional dependencies required to run benchmarks, assuming you've setup the developer environment as previously described:

```bash
pip install -e .[benchmarks]
```

An example analysis:

```python
from bmds import benchmarks

# first time-only
benchmarks.setup_db()

# run the toxrefdb_v2 dataset with four permutations:
#  {continuous, dichotomous} + {BMDS270, BMDS330}

benchmarks.run_analysis(
    dataset=benchmarks.BenchmarkDataset.TOXREFDB_V2,
    analysis=benchmarks.BenchmarkAnalyses.FIT_DICHOTOMOUS,
    version=benchmarks.Version.BMDS270,
    clear_existing=True
)

benchmarks.run_analysis(
    dataset=benchmarks.BenchmarkDataset.TOXREFDB_V2,
    analysis=benchmarks.BenchmarkAnalyses.FIT_CONTINUOUS,
    version=benchmarks.Version.BMDS270,
    clear_existing=True
)

benchmarks.run_analysis(
    dataset=benchmarks.BenchmarkDataset.TOXREFDB_V2,
    analysis=benchmarks.BenchmarkAnalyses.FIT_DICHOTOMOUS,
    version=benchmarks.Version.BMDS330,
    clear_existing=True
)

benchmarks.run_analysis(
    dataset=benchmarks.BenchmarkDataset.TOXREFDB_V2,
    analysis=benchmarks.BenchmarkAnalyses.FIT_CONTINUOUS,
    version=benchmarks.Version.BMDS330,
    clear_existing=True
)

# to delete all results in database
benchmarks.reset_db()
```

The sqlite database is a flat file, which is saved in the `bmds.benchmarks` subpackage. You can delete the sqlite database to start fresh, or use the commands describe below to programmatically clean. The sqlite database can be browsed without tools external to python, such as the [SQLite database browser](https://sqlitebrowser.org/).
