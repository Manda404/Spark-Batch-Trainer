Contributing
============

Development workflow
--------------------

1. Create a focused branch:

   .. code-block:: bash

      git checkout -b feature/my-feature

2. Install the locked development environment:

   .. code-block:: bash

      poetry install

3. Run formatting, static checks, and tests through Poetry:

   .. code-block:: bash

      poetry run black --check src tests
      poetry run isort --check-only src tests
      poetry run mypy src
      poetry run pytest

4. Build the documentation in strict mode:

   .. code-block:: bash

      poetry run sphinx-build -W --keep-going -b html docs/source docs/build/html

5. Open a pull request that explains the behavior change and its tests.

Project conventions
-------------------

* Keep public names, docstrings, comments, logs, and documentation in English.
* Use Google-style docstrings for public classes and functions.
* Put backend-independent behavior in focused ``config``, ``data``,
  ``evaluation``, ``observability``, or ``visualization`` modules.
* Keep model-library details inside ``backends``.
* Add unit tests for isolated behavior and integration tests for Spark/backend
  boundaries.
* Preserve compatibility shims only when a deprecation path is intentional and
  documented.

Documentation changes
---------------------

Update the user guide when behavior changes. Add API directives only for
public or intentionally supported objects; exposing every private helper makes
the reference harder to navigate.
