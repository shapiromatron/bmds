=======
History
=======

v0.4.0 (NEXT)
-------------------

* Added Dichotomous-Hill model to list of dichotomous models
* Allowed global overrides (such as BMR settings) as an optional input to ``Session.add_default_models``
* Updated test-logic outputs for individual tests (and added tests)
* For continuous summary datasets, rename ``responses`` array to ``means``
* By default, polynomial-like models are run multiple times with different degrees.
    - Previously, a single polynomial model was added with an order of ``min(n-1, 8)``, where ``n`` is the number of dose-groups. Now, multilpe models are added ranging from ``[3 - min(n-1, 8)]``, inclusive
    - Polynomial like models include: Polynomial, Multistage, and Multistage-Cancer


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
