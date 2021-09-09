#bmds test suite

The test files are located in toxrefdb/app
jupyter-notebook file to launch bmds test cases are located in /scripts.
The outputs of the test cases are saved into sqlite database.

# bmds data comparison

2021-08-10

The contents of this zip file contain an analysis where we run the same datasets using BMDS2.7 and a development version of BMDS3.3 The goal of the current work is to move this scratch and development code into the main bmds repository, so we're able to re-run this analysis using future versions of BMDS. Because the data are so large, we are saving model outputs into an sqlite database. The `test_db/app` folder creates a test application which executes the models and stores the results.

Our main goal is to cleanup, refactor, and generalize the code where possible, so this analysis can be repeated on current and future versions of the software.

See the bmds/test_db directory, and the main.ipynb as an example entrypoint into this application.

Action items:

1. Redo current analysis as is using the outputs of the zip file. (Windows required for BMDS2). You'll need to be in the bmds python virtual environment. It's fine to run a smaller set of the datsets (just running the first 10 of each data type instead of all 1500; if it works for them then it will work with a little tuning with the larger dataset)
2. Save code in current bmds

   - Add to bmds package on github, perhaps in a `./bmds/benchmarks/toxrefdb` package?
   - Put data in a `./data/toxrefdb/` folder

   - write a simple python script or jupyter notebook or command line cli using [typer](https://typer.tiangolo.com/) for running bmds2 and bmds3 analyses

   # -bmds run_analysis --dataset=toxrefdb --clear_existing --versions=BMDS27,BMDS33

   --jupyter-notebook (string, bool, list)

   - make sure the sqlite database is in the .gitignore; save toxrefdb datasets in .csv.zip compressed formats (pandas can natively open .csv.zip; see https://stackoverflow.com/a/32993553/906385)
   - add a new pull request

3. Add additional items to output (requested from Matt Wheeler):
   - optimized likelihood, the model degrees of freedom, the BMD, BMDL and BMDU. We should also look at the test values too, but only in a secondary look
   - ask Cody how to get any missing values for bmds3
   - for bmds2 just use -999 for now
   - add a new pull request
4. convert notebook to typer
5. Refactor and simplify code.
   - Add option to pass in other datasets
   - add a new pull request
