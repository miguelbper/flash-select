<div align="center">

# Python Template
[![Python](https://img.shields.io/badge/Python-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![Code Quality](https://github.com/miguelbper/python-template/actions/workflows/code-quality.yaml/badge.svg)](https://github.com/miguelbper/python-template/actions/workflows/code-quality.yaml)
[![Unit Tests](https://github.com/miguelbper/python-template/actions/workflows/tests.yaml/badge.svg)](https://github.com/miguelbper/python-template/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/miguelbper/python-template/graph/badge.svg)](https://codecov.io/gh/miguelbper/python-template)
[![License](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](LICENSE)

Template for python projects, using modern tooling and practices

</div>

---

## Description
A template for Python projects, using:
- **[uv](https://docs.astral.sh/uv/)**: A fast Python package installer and resolver
- **[just](https://github.com/casey/just)**: A command runner that makes it easy to run project-specific commands
- **[direnv](https://direnv.net/)**: Automatically loads and unloads environment variables based on the current directory
- **[pre-commit](https://pre-commit.com/)**: Git hooks to ensure code quality before commits
- **[ruff](https://docs.astral.sh/ruff/)**: A fast Python linter and formatter
- **[pyright](https://github.com/microsoft/pyright)**: Static type checker for Python
- **[pytest](https://docs.pytest.org/)**: Testing framework
- **CI/CD**: GitHub Actions workflows for testing, code quality, and publishing
- **Version Management**: Automatic version management using git tags


## Project Structure
```shell
.
├── .github
│   └── workflows
│       ├── code-quality.yaml  # Runs all pre-commit hooks (which includes linting and formatting)
│       ├── coverage.yaml      # Reports test coverage to Codecov
│       ├── publish.yaml       # Whenever there is a new tag, publishes the package to GitHub / PyPI
│       └── tests.yaml         # Runs the test suite
│
├── src
│   └── python_template        # Main package source code
│       ├── __init__.py
│       └── main.py            # Example script
│
├── tests
│   ├── __init__.py
│   └── test_main.py           # Example tests script
│
├── .envrc.example             # Rename to .envrc and include your environment variables here
├── .gitignore
├── .pre-commit-config.yaml    # Configuration for pre-commit hooks
├── .python-version            # Pinned Python version for the project
├── justfile                   # Command runner for project tasks
├── LICENSE
├── project-image.png
├── pyproject.toml             # Python project configuration and dependencies
├── README.md
└── uv.lock                    # Lock file for Python dependencies
```

## Installation
For all functionality in this template to work, you need to install the suggested tooling: [uv](https://docs.astral.sh/uv/), [just](https://github.com/casey/just) and [direnv](https://direnv.net/). See each individual webpage for installation instructions.


## Usage
- Click on [<kbd>Use this template</kbd>](https://github.com/miguelbper/python-template/generate) to create a new repository
- Run `just` in the terminal to view the available project commands, together with a description
- Run `just setup` in the terminal to setup the project. This is equivalent to running the following commands in the shell:
```shell
# Allow direnv to load environment variables
direnv allow

# Create virtual environment and install dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install
```
- Setup any environment variables you may need by adding them to `.envrc`.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
Some helpful links about modern Python practices and tooling:
- https://drivendata.co/blog/python-packaging-2023
- https://rdrn.me/postmodern-python/

---
*Template for your project's `README.md`. Delete text above this line, and edit the code below.*

<div align="center">

# Project Title
[![Python](https://img.shields.io/badge/Python-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![Code Quality](https://github.com/miguelbper/python-template/actions/workflows/code-quality.yaml/badge.svg)](https://github.com/miguelbper/python-template/actions/workflows/code-quality.yaml)
[![Unit Tests](https://github.com/miguelbper/python-template/actions/workflows/tests.yaml/badge.svg)](https://github.com/miguelbper/python-template/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/miguelbper/python-template/graph/badge.svg)](https://codecov.io/gh/miguelbper/python-template)
[![License](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](LICENSE)

*A brief description of what this project does and who it's for.*

![project-image.png](project-image.png)

</div>

---

## Description
*Provide a more detailed explanation about your project here. You can include:*
- *What problem does it solve?*
- *What technologies are used?*
- *What are the key features?*

## Installation
*Provide installation instructions. Write out the commands the user should use to use your project.*

## Usage
- *Explain how to use your project. Include code examples.*
- *You can also include screenshots or GIFs here to demonstrate usage.*

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- *List any resources, libraries, or tools that you used or were inspired by*
- *Credit any collaborators or contributors*
- *Link to any relevant documentation or articles*

## More badges
[![Bash](https://img.shields.io/badge/Bash-293036?logo=gnu-bash&logoColor=white)](https://www.gnu.org/software/bash/)
[![Git](https://img.shields.io/badge/Git-f44d27?logo=git&logoColor=white)](https://git-scm.com/)
[![NumPy](https://img.shields.io/badge/NumPy-4dabcf?logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-130654?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-5c7da2?logo=python&logoColor=white)](https://seaborn.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-013243?logo=scipy&logoColor=white)](https://scipy.org/)
[![SymPy](https://img.shields.io/badge/SymPy-3b5526?logo=sympy&logoColor=white)](https://www.sympy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-f99f44?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyTorch Lightning](https://img.shields.io/badge/-Lightning-7e4fff?logo=lightning&logoColor=white)](https://lightning.ai/docs/pytorch/stable/)
[![hydra](https://img.shields.io/badge/Configs-Hydra-89b8cd)](https://hydra.cc/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
