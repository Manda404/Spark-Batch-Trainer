Contributing
============

Thank you for your interest in contributing to **Spark Batch Trainer**!

Contribution Steps
------------------
1. Fork the GitHub repository.  
2. Create a branch for your changes:  

   .. code-block:: bash

      git checkout -b feature/my-feature

3. Install development dependencies:  

   .. code-block:: bash

      poetry install

4. Run style checks and tests:  

   .. code-block:: bash

      black src tests
      isort src tests
      mypy src
      pytest

5. Push your changes and open a Pull Request.

Best Practices
--------------
- Follow the PEP8 coding style.  
- Use docstrings (NumPy format).  
- Add unit tests for every new feature.  
