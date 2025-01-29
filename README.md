<div align="center">

# MeshPy <!-- omit from toc -->
</div>

<div align="center">

[![website](https://raw.githubusercontent.com/imcs-compsim/meshpy/refs/heads/main/utilities/badges/website.svg)](https://imcs-compsim.github.io/meshpy/)
[![documentation](https://raw.githubusercontent.com/imcs-compsim/meshpy/refs/heads/main/utilities/badges/documentation.svg)](https://imcs-compsim.github.io/meshpy/api-documentation)

</div>

<div align="center">

[![Code quality](https://github.com/imcs-compsim/meshpy/actions/workflows/check_code.yml/badge.svg)](https://github.com/imcs-compsim/meshpy/actions/workflows/check_code.yml?query=event%3Aschedule)
[![Test suite](https://github.com/imcs-compsim/meshpy/actions/workflows/testing.yml/badge.svg)](https://github.com/imcs-compsim/meshpy/actions/workflows/testing.yml?query=event%3Aschedule)
[![Coverage](https://imcs-compsim.github.io/meshpy/coverage-badge/coverage_badge.svg)](https://imcs-compsim.github.io/meshpy/coverage-report/)

</div>

<div align="center">

[![Testing Linux/Ubuntu](https://raw.githubusercontent.com/imcs-compsim/meshpy/refs/heads/main/utilities/badges/testing_linux_ubuntu.svg)](https://github.com/imcs-compsim/meshpy/actions/workflows/testing.yml?query=event%3Aschedule)
[![Testing macOS](https://raw.githubusercontent.com/imcs-compsim/meshpy/refs/heads/main/utilities/badges/testing_macos.svg)](https://github.com/imcs-compsim/meshpy/actions/workflows/testing.yml?query=event%3Aschedule)
[![Testing Windows](https://raw.githubusercontent.com/imcs-compsim/meshpy/refs/heads/main/utilities/badges/testing_windows.svg)](https://github.com/imcs-compsim/meshpy/actions/workflows/testing.yml?query=event%3Aschedule)

</div>

<div align="center">

[![pre-commit](https://raw.githubusercontent.com/imcs-compsim/meshpy/refs/heads/main/utilities/badges/pre-commit.svg)](https://pre-commit.com/)
[![ruff-formatter](https://raw.githubusercontent.com/imcs-compsim/meshpy/refs/heads/main/utilities/badges/ruff-formatter.svg)](https://docs.astral.sh/ruff/formatter)
[![ruff-linter](https://raw.githubusercontent.com/imcs-compsim/meshpy/refs/heads/main/utilities/badges/ruff-linter.svg)](https://docs.astral.sh/ruff/linter)

</div>

MeshPy is a general purpose 3D beam finite element input generator written in `python`.
It contains advanced geometry creation and manipulation functions to create complex beam geometries, including a consistent handling of finite rotations.
It can be used to create input files for the following finite element solvers (adaption to other solvers is easily possibly):
- [4C](https://www.4c-multiphysics.org/) (academic finite element solver)
- [Abaqus](https://en.wikipedia.org/wiki/Abaqus) (commercial software package)
- [AceFEM](http://symech.fgg.uni-lj.si) (Finite element package for automation of the finite element method in [Mathematica](https://www.wolfram.com/mathematica/))

MeshPy is developed at the [Institute for Mathematics and Computer-Based Simulation (IMCS)](https://www.unibw.de/imcs-en) at the Universität der Bundeswehr München.

## Overview <!-- omit from toc -->
- [Examples](#examples)
- [How to use MeshPy?](#how-to-use-meshpy)
- [How to cite MeshPy?](#how-to-cite-meshpy)
- [Work that uses MeshPy](#work-that-uses-meshpy)
- [Installation](#installation)
  - [Optional dependency on ArborX](#optional-dependency-on-arborx)
- [Contributing](#contributing)
- [Contributors](#contributors)

## Examples

<p align="center">
  <img src="https://raw.githubusercontent.com/imcs-compsim/meshpy/refs/heads/main/utilities/doc/honeycomb.png" width="350" title="Honeycomb structure under tension (simulated with 4C)">
</p>
<p align="center" style="font-style: italic; color: gray;">Honeycomb structure under tension (simulated with 4C)</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/imcs-compsim/meshpy/refs/heads/main/utilities/doc/composite_plate.png" width="400" title="Fiber reinforced composite plate (simulated with 4C)">
</p>
<p align="center" style="font-style: italic; color: gray;">Fiber reinforced composite plate (simulated with 4C)</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/imcs-compsim/meshpy/refs/heads/main/utilities/doc/pressure_pipe.png" width="350" title="Fiber reinforced pipe under pressure (simulated with 4C)">
</p>
<p align="center" style="font-style: italic; color: gray;">Fiber reinforced pipe under pressure (simulated with 4C)</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/imcs-compsim/meshpy/refs/heads/main/utilities/doc/twisted_plate.png" width="350" title="Fiber reinforcements of a twisted plate (simulated with 4C)">
</p>
<p align="center" style="font-style: italic; color: gray;">Fiber reinforcements of a twisted plate (simulated with 4C)</p>

## How to use MeshPy?

MeshPy provides example notebooks to showcase its core features and functionality.
The examples can be found in the `examples/` directory.
They can be run locally or directly tested from your browser via the following links:

- Example 1: **Finite rotation framework** [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/imcs-compsim/meshpy/main?labpath=examples%2Fexample_1_finite_rotations.ipynb)
- Example 2: **Core mesh generation functions** [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/imcs-compsim/meshpy/main?labpath=examples%2Fexample_2_core_mesh_generation_functions.ipynb)

You can also interactively test the entire MeshPy framework directly from your browser here [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/imcs-compsim/meshpy/main)


## How to cite MeshPy?

Whenever you use or mention MeshPy in some sort of scientific document/publication/presentation, please cite MeshPy as
> Steinbrecher, I., Popp, A.: MeshPy - A general purpose 3D beam finite element input generator, https://imcs-compsim.github.io/meshpy

This can be done with the following BiBTeX entry:
```TeX
@Misc{MeshPyWebsite,
  author       = {Steinbrecher, I. and Popp, A.},
  howpublished = {\url{https://imcs-compsim.github.io/meshpy}},
  title        = {{M}esh{P}y -- {A} general purpose {3D} beam finite element input generator},
  year         = {2021},
  key          = {MeshPyWebsite},
  url          = {https://imcs-compsim.github.io/meshpy},
}
```

Feel free to leave a ⭐️ on [GitHub](https://github.com/imcs-compsim/meshpy).


## Work that uses MeshPy

### Peer-reviewed articles <!-- omit from toc -->

1. </span><span class="csl-right-inline">Firmbach, M., Steinbrecher, I.,
Popp, A., Mayr, M.: An approximate block factorization preconditioner
for mixed-dimensional beam-solid interaction. Computer Methods in
Applied Mechanics and Engineering. 431, 117256 (2024).
https://doi.org/<https://doi.org/10.1016/j.cma.2024.117256></span>
1. </span><span class="csl-right-inline">Hagmeyer, N., Mayr, M., Popp, A.:
A fully coupled regularized mortar-type finite element approach for
embedding one-dimensional fibers into three-dimensional fluid flow.
International Journal for Numerical Methods in Engineering. 125, e7435
(2024). https://doi.org/<https://doi.org/10.1002/nme.7435></span>
1. </span><span class="csl-right-inline">Steinbrecher, I., Popp, A., Meier,
C.: Consistent coupling of positions and rotations for embedding 1D
Cosserat beams into 3D solid volumes. Computational Mechanics. 69,
701–732 (2022). <https://doi.org/10.1007/s00466-021-02111-4></span>
1. </span><span class="csl-right-inline">Hagmeyer, N., Mayr, M.,
Steinbrecher, I., Popp, A.: One-way coupled fluid-beam interaction:
Capturing the effect of embedded slender bodies on global fluid flow and
vice versa. Advanced Modeling and Simulation in Engineering Sciences. 9,
9 (2022). <https://doi.org/10.1186/s40323-022-00222-y></span>
1. </span><span class="csl-right-inline">Steinbrecher, I., Mayr, M., Grill,
M.J., Kremheller, J., Meier, C., Popp, A.: A mortar-type finite element
approach for embedding 1D beams into 3D solid volumes. Computational
Mechanics. 66, 1377–1398 (2020).
<https://doi.org/10.1007/s00466-020-01907-0></span>

### PhD thesis <!-- omit from toc -->

1. </span><span class="csl-right-inline">Hagmeyer, N.: A computational
framework for balloon angioplasty and stented arteries based on
mixed-dimensional modeling,
<https://athene-forschung.rz.unibw-muenchen.de/146359>, (2023)</span>
1. </span><span class="csl-right-inline">Steinbrecher, I.:
Mixed-dimensional finite element formulations for beam-to-solid
interaction, <https://athene-forschung.unibw.de/143755>, (2022)</span>


## Installation

MeshPy is tested with and supports python versions `3.9-3.12`.
Other versions of Python might lead to issues.
It is recommended to use a python environment container such as `conda` or `venv`.
- `conda`:
  A [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) environment can be created and loaded with
  ```bash
  # Create the environment (this only has to be done once)
  conda create -n meshpy python=3.12
  # Activate the environment
  conda activate meshpy
  ```
- `venv`: A virtual environment can be setup with
  - On Debian systems the following packages might have to be installed:
    ```bash
    sudo apt-get install python3-venv python3-dev
    ```
  - Create and load the environment
    ```bash
    # Create the environment (this only has to be done once)
    python -m venv <path-to-env-folder>/meshpy-env
    # Activate the environment
    source <path-to-env-folder>/meshpy-env/bin/activate
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

To check if everything worked as expected, run the standard tests with
```bash
pytest
```

Further tests can be added with the following flags: `--4C`, `--ArborX`, `--CubitPy`, `--performance-tests`.
These can be arbitrarily combined, for example
```bash
pytest --CubitPy --performance-tests
```
executes the standard tests, the CubitPy tests and the performance tests.
Note: the reference time values for the performance tests might not suite your system.

We can also run tests in combination with 4C
```bash
# 4C Tests require a path to a 4C executable
export MESHPY_FOUR_C_EXE=<path_to_4C>
pytest --4C --performance-tests
```

Finally, the base tests can be deactivated with `--exclude-standard-tests`.
For example to just run the CubitPy tests execute
```bash
pytest --CubitPy --exclude-standard-tests
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

## Contributing

If you are interested in contributing to MeshPy, we welcome your collaboration.
For general questions, feature request and bug reports please open an [issue](https://github.com/imcs-compsim/meshpy/issues).

If you contribute actual code, fork the repository and make the changes in a feature branch.
Depending on the topic and amount of changes you also might want to open an [issue](https://github.com/imcs-compsim/meshpy/issues).
To merge your changes into the MeshPy repository, create a pull request to the `main` branch.
A few things to keep in mind:
- It is highly encouraged to add tests covering the functionality of your changes, see the test suite in `tests/`.
- To maintain high code quality, MeshPy uses a number of different pre-commit hooks to check committed code. Make sure to set up the pre-commit hooks before committing your changes (run in the repository root folder):
  ```bash
  pre-commit install
  ```
- Check that you did not break anything by running the MeshPy tests.
  For most changes it should be sufficient to run the standard test suite (run in the repository root folder):
  ```bash
  pytest
  ```
- Feel free to add yourself to the contributors section in the [README.md](https://github.com/imcs-compsim/meshpy/blob/main/README.md) file.


## Contributors

### Main developer <!-- omit from toc -->
Ivo Steinbrecher (@isteinbrecher)

### Contributors (in alphabetical order) <!-- omit from toc -->
- Dao Viet Anh
- Max Firmbach (@maxfirmbach)
- Martin Frank (@knarfnitram)
- Nora Hagmeyer (@NoraHagmeyer)
- Matthias Mayr (@mayrmt)
- Gabriela Loera (@eulovi)
- David Rudlstorfer (@davidrudlstorfer)
