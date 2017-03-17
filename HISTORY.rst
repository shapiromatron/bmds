=======
History
=======

v0.7.0 (NEXT)
-------------------
* Recommend the most parsimonious model, instead of the first model with target AIC/BMDL
* Add new method to the session, ``execute_and_recommend``, with the option to try dropping doses until a model recommendation exists, or the dataset is exhausted
* Add runtime details to output
* Hide model failure popup when bmds unexpectedly fails  (significant performance boost)
* Log failures by default, including displaying failed input (d) files

v0.6.0 (2017-03-10)
-------------------
* Added new `drop_dose` method to Dataset
* Do not attempt to execute model when there are too few dose-groups
* Remove doses-dropped parameter from dataset init
* Add example notebook running actual data

v0.5.3 (2017-03-02)
-------------------

* Prevent errors when software is run with un-runnable dose-response datasets
* Handle edge-cases for ANOVA calculation failure
* Fix 002 bmds temporary file cleanup

v0.5.2 (2017-02-15)
-------------------

* Add custom exceptions for BMDS package
* Explicitly check that BMDS remote-server authentication is successful
* Hotfix - fix error when running continuous models with 3 dose groups

v0.5.1 (2016-12-23)
-------------------

* hotfix - fix exponential models (they create additional temporary files, had to ensure that they're collected and removed.)

v0.5.0 (2016-12-23)
-------------------

* For multistage and multistage cancer, by default an order 1 polynomial model is also executed (previously started at order 2)
* Update documentation beyond quickstart including API
* Export results as a pandas DataFrame, CSV, or Excel, in addition to JSON, and python dictionaries
* Generate dose-response plots using matplotlib
* Export dose-response plots
* Improve documentation with better describing API and quickstart

v0.4.0 (2016-12-14)
-------------------

* Added Dichotomous-Hill model to list of dichotomous models
* Allowed global overrides (such as BMR settings) as an optional input to ``Session.add_default_models``
* Updated test-logic outputs for individual tests (and added tests)
* For continuous summary datasets, rename ``responses`` array to ``means``
* By default, polynomial-like models are run multiple times with different degrees.
    - Previously, a single polynomial model was added with an order of ``min(n-1, 8)``, where ``n`` is the number of dose-groups. Now, multilpe models are added ranging from ``[3 - min(n-1, 8)]``, inclusive
    - Polynomial like models include: Polynomial, Multistage, and Multistage-Cancer
* Added the ability to use individual continuous data, instead of summary data

v0.3.0 (2016-12-05)
-------------------

* Model recommendation system enabled [`Wignall et al. 2014`_]
* Default continuous variance model now calculated based on dataset using same statistics as BMDS [Thanks Longlong!]
* Default polynomial restriction based on if dataset is increasing or decreasing (previously unrestricted)
* Add new batch dFileRunner to execute multiple dfiles in batch-mode (integration w/ bmds-server)
* Updated Makefile to include with a new tmux developer environment

.. _`Wignall et al. 2014`: https://doi.org/10.1289/ehp.1307539

v0.2.0 (2016-11-23)
-------------------

* Remove older version of BMDS unused in model code
* Updated to working versions of BMDS code

v0.1.0 (2016-10-25)
-------------------

* Allowed for monkeypatch check for executing on linux, since BMDS is Windows-only
* Added model recommendation logic
* Added python 3 support (3.5)
* First PyPI release

v0.0.1 (2016-07-29)
-------------------

* Initial version (github only)
