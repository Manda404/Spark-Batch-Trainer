Contributing
============

Merci de votre intérêt pour contribuer à **Spark Batch Trainer**

Étapes pour contribuer
----------------------
1. Forkez le dépôt GitHub.
2. Créez une branche pour vos changements :
   .. code-block:: bash

      git checkout -b feature/ma-feature

3. Installez les dépendances de dev :
   .. code-block:: bash

      poetry install

4. Vérifiez le style et les tests :
   .. code-block:: bash

      black src tests
      isort src tests
      mypy src
      pytest

5. Poussez vos changements et créez une Pull Request.

Bonnes pratiques
----------------
- Respectez le style PEP8.
- Utilisez des docstrings (format NumPy).
- Ajoutez des tests unitaires pour chaque nouvelle fonctionnalité.
