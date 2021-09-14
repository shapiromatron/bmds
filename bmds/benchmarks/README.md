9/14/2021

The `benchmarks/toxrefdb` folder creates a test application which executes models and store the results in the database app.db

The current test application can be repeated on future versions of bmds.

Then entrypoint for the testing application is in `benchmarks/main.py`. run_analysis methods executes the test application. below is the example command to execute the run_analysis method

# example command from api.ipynb

from bmds.benchmarks.main import Dataset, Versions, Benchmark
Benchmark.run_analysis(Dataset.TOXREFDB_DICH, [Versions.BMDS270, Versions.BMDS330], True)

if load dataset error message is seen, use example command to load dataset.
Benchmark.load_dataset(Dataset.TOXREFDB_DICH, "dichotomous_tr.csv")

# NO such Table error

if running the analysis gives error message to create db, execute create_db() from main.py file as follows to create tables in the db.
Benchmark.create_db()

# below are the action items for next PR.

# bmds data comparison

Action items:

1. Save code in current bmds

   - Add to bmds package on github, perhaps in a `./bmds/benchmarks/toxrefdb` package?
   - Put data in a `./data/toxrefdb/` folder

   - write a simple python script or jupyter notebook or command line cli using [typer](https://typer.tiangolo.com/) for running bmds2 and bmds3 analyses

   # -bmds run_analysis --dataset=toxrefdb --clear_existing --versions=BMDS27,BMDS33

   --jupyter-notebook (string, bool, list)

   - make sure the sqlite database is in the .gitignore; save toxrefdb datasets in .csv.zip compressed formats (pandas can natively open .csv.zip; see https://stackoverflow.com/a/32993553/906385)
   - add a new pull request

2. Add additional items to output (requested from Matt Wheeler):
   - optimized likelihood, the model degrees of freedom, the BMD, BMDL and BMDU. We should also look at the test values too, but only in a secondary look
   - ask Cody how to get any missing values for bmds3
   - for bmds2 just use -999 for now
   - add a new pull request
3. convert notebook to typer
4. Refactor and simplify code.
   - Add option to pass in other datasets
   - add a new pull request
