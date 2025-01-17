<div align="center">

# MeshPy
</div>

<div align="center">

[![website](https://img.shields.io/badge/MeshPy-website?label=website&color=blue)](https://imcs-compsim.github.io/meshpy/)

</div>

<div align="center">

[![Code quality](https://github.com/imcs-compsim/meshpy/actions/workflows/check_code.yml/badge.svg)](https://github.com/imcs-compsim/meshpy/actions/workflows/check_code.yml)
[![Test suite](https://github.com/imcs-compsim/meshpy/actions/workflows/testing.yml/badge.svg)](https://github.com/imcs-compsim/meshpy/actions/workflows/testing.yml)

</div>

<div align="center">

[![pre-commit](https://img.shields.io/badge/enabled-green?logo=pre-commit&label=pre-commit)](https://pre-commit.com/)
[![ruff-formatter](https://img.shields.io/badge/code-ruff-green?logo=ruff&label=code-formatter)](https://docs.astral.sh/ruff/formatter)
[![ruff-linter](https://img.shields.io/badge/code-ruff-green?logo=ruff&label=code-linter)](https://docs.astral.sh/ruff/linter)

</div>

MeshPy is a general purpose 3D beam finite element input generator written in `python`.
It contains basic geometry creation and manipulation functions to create complex beam geometries, including rotational degrees of freedom for the beam nodes.
It can be used to create input files for the following finite element solvers:
- [4C](https://www.4c-multiphysics.org/) (academic finite element solver)
- [Abaqus](https://en.wikipedia.org/wiki/Abaqus) (commercial software package)

MeshPy can easily be adapted to create input files for other solvers.
MeshPy is developed at the [Institute for Mathematics and Computer-Based Simulation (IMCS)](https://www.unibw.de/imcs-en) at the Universität der Bundeswehr München.

## How to cite MeshPy?

Whenever you use or mention MeshPy in some sort of scientific document/publication/presentation, please cite MeshPy as described on the [MeshPy website](https://imcs-compsim.github.io/meshpy).

Feel free to leave a ⭐️ on [GitHub](https://github.com/imcs-compsim/meshpy).

## How to use MeshPy?

Basic tutorials can be found in the directory `tutorial/`.


## Contributing

If you are interested in contributing to MeshPy, we welcome your collaboration.
For general questions, feature request and bug reports please open an [issue](https://github.com/imcs-compsim/meshpy/issues).

If you contribute actual code, fork the repository and make the changes in a feature branch.
Depending on the topic and amount of changes you also might want to open an [issue](https://github.com/imcs-compsim/meshpy/issues).
To merge your changes into the MeshPy repository, create a pull request to the `main` branch.
A few things to keep in mind:
- It is highly encouraged to add tests covering the functionality of your changes, see the test suite in `tests/`.
- MeshPy uses `black` to format python code.
  Make sure to apply `black` to the changed source files.
- Feel free to add yourself to the [CONTRIBUTORS](https://github.com/imcs-compsim/meshpy/blob/main/CONTRIBUTORS) file.

## Installation

MeshPy is developed with `python3.12`.
Other versions of Python might lead to issues.
It is recommended to use a python environment container such as `conda` or `venv`.
- `conda`:
  A [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) environment can be created and loaded with
  ```bash
  conda create -n meshpy python=3.12
  conda activate meshpy
  ```
- `venv`: Chose an appropriate directory for this, e.g., `/home/user/opt`.
  A virtual environment can be setup with
  - On Debian systems the following packages have to be installed:
    ```bash
    sudo apt-get install python3-venv python3-dev
    ```
  - Create and load the environment
    ```bash
    cd <path-to-env-folder>
    python -m venv meshpy-env
    source meshpy-env/bin/activate
    ```

From now on we assume that the previously created environment is loaded.
To install `meshpy` go to the repository root directory
```bash
cd <path_to_meshpy>
```

Install `meshpy` via `pip`
```bash
pip install .
```

If you intend to actively develop `meshpy`, install it in *editable mode*

```bash
pip install -e .
```

If `cython` code is changed, it has to be recompiled. This can be done by running (in the root directory)
```bash
python setup.py build_ext --inplace
```

Optional, a path to the `4C` executable can be given in order to run some combined
tests with 4C
```bash
export MESHPY_FOUR_C_EXE=path_to_4C
```

To check if everything worked as expected, run the standard tests with
```bash
pytest
```

Further tests can be added with the following flags: `--4C`, `--ArborX`, `--CubitPy`, `--performance-tests`.
These can be arbitrarily combined, for example
```bash
pytest --4C --CubityPy
```
executes the standard tests, the 4C tests and the CubitPy tests. Note that the reference time values for the performance tests might not suite your system.

Finally, the base tests can be deactivated with `--exclude-standard-tests`. For example to just run the CubitPy tests execute
```bash
pytest --CubitPy --exclude-standard-tests
```

Before you are ready to contribute to MeshPy, please make sure to install the `pre-commit hook` within the python environment to follow our style guides:
```bash
pre-commit install
```

### Optional dependency on [ArborX](https://github.com/arborx/ArborX)

MeshPy can optionally execute its geometric search functions using the C++ library [ArborX](https://github.com/arborx/ArborX).
First make sure the `pybind11` submodule is loaded
```bash
cd <path_to_meshpy>
git submodule update --init
```
To setup meshpy with ArborX, `cmake` and Kokkos are available on your system (the preferred variant is via [Spack](https://spack.io/)).
Create a build directory
```bash
mkdir -p <path_to_meshpy>/build/geometric_search
```
Configure cmake and build the extension
```bash
cd <path_to_meshpy>/build/geometric_search
cmake ../../meshpy/geometric_search/src/
make -j4
```

If the ArborX extension is working correctly can be checked by running the geometric search tests
```bash
pytest --ArborX
```

## Examples

<p align="center">
  <img src="https://raw.githubusercontent.com/imcs-compsim/meshpy/refs/heads/main/utilities/doc/composite_plate.png" width="350" title="Fiber reinforced composite plate">
</p>
<p align="center" style="font-style: italic; color: gray;">Fiber reinforced composite plate</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/imcs-compsim/meshpy/refs/heads/main/utilities/doc/pressure_pipe.png" width="350" title="Fiber reinforced pipe under pressure">
</p>
<p align="center" style="font-style: italic; color: gray;">Fiber reinforced pipe under pressure</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/imcs-compsim/meshpy/refs/heads/main/utilities/doc/twisted_plate.png" width="350" title="Fiber reinforcements of a twisted plate">
</p>
<p align="center" style="font-style: italic; color: gray;">Fiber reinforcements of a twisted plate</p>

## Contributors

### Main developer
Ivo Steinbrecher (@isteinbrecher)

### Contributors (in alphabetical order)
- Dao Viet Anh
- Nora Hagmeyer (@NoraHagmeyer)
- Matthias Mayr (@mayrmt)
- Gabriela Loera (@eulovi)
- David Rudlstorfer (@davidrudlstorfer)
