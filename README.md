<div align="center">

# MeshPy <!-- omit from toc -->
</div>

<div align="center">

[![website](https://raw.githubusercontent.com/imcs-compsim/meshpy/refs/heads/main/doc/badges/website.svg)](https://imcs-compsim.github.io/meshpy/)
[![documentation](https://raw.githubusercontent.com/imcs-compsim/meshpy/refs/heads/main/doc/badges/documentation.svg)](https://imcs-compsim.github.io/meshpy/api-documentation)

</div>

<div align="center">

[![Code quality](https://github.com/imcs-compsim/meshpy/actions/workflows/check_code.yml/badge.svg)](https://github.com/imcs-compsim/meshpy/actions/workflows/check_code.yml?query=event%3Aschedule)
[![Test suite](https://github.com/imcs-compsim/meshpy/actions/workflows/testing.yml/badge.svg)](https://github.com/imcs-compsim/meshpy/actions/workflows/testing.yml?query=event%3Aschedule)
[![Coverage](https://imcs-compsim.github.io/meshpy/coverage-badge/coverage_badge.svg)](https://imcs-compsim.github.io/meshpy/coverage-report/)

</div>

<div align="center">

[![Testing Linux/Ubuntu](https://raw.githubusercontent.com/imcs-compsim/meshpy/refs/heads/main/doc/badges/testing_linux_ubuntu.svg)](https://github.com/imcs-compsim/meshpy/actions/workflows/testing.yml?query=event%3Aschedule)
[![Testing macOS](https://raw.githubusercontent.com/imcs-compsim/meshpy/refs/heads/main/doc/badges/testing_macos.svg)](https://github.com/imcs-compsim/meshpy/actions/workflows/testing.yml?query=event%3Aschedule)
[![Testing Windows](https://raw.githubusercontent.com/imcs-compsim/meshpy/refs/heads/main/doc/badges/testing_windows.svg)](https://github.com/imcs-compsim/meshpy/actions/workflows/testing.yml?query=event%3Aschedule)

</div>

<div align="center">

[![pre-commit](https://raw.githubusercontent.com/imcs-compsim/meshpy/refs/heads/main/doc/badges/pre-commit.svg)](https://pre-commit.com/)
[![ruff-formatter](https://raw.githubusercontent.com/imcs-compsim/meshpy/refs/heads/main/doc/badges/ruff-formatter.svg)](https://docs.astral.sh/ruff/formatter)
[![ruff-linter](https://raw.githubusercontent.com/imcs-compsim/meshpy/refs/heads/main/doc/badges/ruff-linter.svg)](https://docs.astral.sh/ruff/linter)

</div>

MeshPy is a general purpose 3D beam finite element input generator written in Python.
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
  - [Python environment](#python-environment)
  - [Install MeshPy from GitHub (most recent version)](#install-meshpy-from-github-most-recent-version)
  - [Install MeshPy from source](#install-meshpy-from-source)
- [Optional dependencies](#optional-dependencies)
  - [4C](#4c)
  - [CubitPy](#cubitpy)
  - [ArborX geometric search](#arborx-geometric-search)
- [Developing MeshPy](#developing-meshpy)
  - [Coding guidelines](#coding-guidelines)
  - [Testing](#testing)
  - [Cython geometric search](#cython-geometric-search)
- [Contributing](#contributing)
- [Authors](#authors)

## Examples

<p align="center">
  <img src="https://raw.githubusercontent.com/imcs-compsim/meshpy/refs/heads/main/doc/assets/honeycomb.png" width="350" title="Honeycomb structure under tension (simulated with 4C)">
</p>
<p align="center" style="font-style: italic; color: gray;">Honeycomb structure under tension (simulated with 4C)</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/imcs-compsim/meshpy/refs/heads/main/doc/assets/composite_plate.png" width="400" title="Fiber reinforced composite plate (simulated with 4C)">
</p>
<p align="center" style="font-style: italic; color: gray;">Fiber reinforced composite plate (simulated with 4C)</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/imcs-compsim/meshpy/refs/heads/main/doc/assets/pressure_pipe.png" width="350" title="Fiber reinforced pipe under pressure (simulated with 4C)">
</p>
<p align="center" style="font-style: italic; color: gray;">Fiber reinforced pipe under pressure (simulated with 4C)</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/imcs-compsim/meshpy/refs/heads/main/doc/assets/twisted_plate.png" width="350" title="Fiber reinforcements of a twisted plate (simulated with 4C)">
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

1. </span><span class="csl-right-inline">Datz, J.C., Steinbrecher, I.,
Meier, C., Engel, L.C., Popp, A., Pfaller, M.R., Schunkert, H., Wall, W.A.:
Patient-specific coronary angioplasty simulations — A mixed-dimensional
finite element modeling approach. Computers in Biology and Medicine. 189,
109914 (2025).
<https://doi.org/10.1016/j.compbiomed.2025.109914></span>
1. </span><span class="csl-right-inline">Firmbach, M., Steinbrecher, I.,
Popp, A., Mayr, M.: An approximate block factorization preconditioner
for mixed-dimensional beam-solid interaction. Computer Methods in
Applied Mechanics and Engineering. 431, 117256 (2024).
<https://doi.org/10.1016/j.cma.2024.117256></span>
1. </span><span class="csl-right-inline">Hagmeyer, N., Mayr, M., Popp, A.:
A fully coupled regularized mortar-type finite element approach for
embedding one-dimensional fibers into three-dimensional fluid flow.
International Journal for Numerical Methods in Engineering. 125, e7435
(2024). <https://doi.org/10.1002/nme.7435></span>
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

### Python environment

MeshPy is tested with, and supports Python versions 3.9-3.12. It is recommended to use a virtual Python environment such as [Conda](https://anaconda.org/anaconda/conda)/[Miniforge](https://conda-forge.org/download/) or [venv](https://docs.python.org/3/library/venv.html).
- A [Conda](https://anaconda.org/anaconda/conda)/[Miniforge](https://conda-forge.org/download/) environment can be created and loaded with
  ```bash
  # Create the environment (this only has to be done once)
  conda create -n meshpy python=3.12
  # Activate the environment
  conda activate meshpy
  ```
- A [venv](https://docs.python.org/3/library/venv.html) virtual environment can be created and loaded with (on Debian systems the following packages might have to be installed:
    `sudo apt-get install python3-venv python3-dev`)
  ```bash
  # Create the environment (this only has to be done once)
  python -m venv <path-to-env-folder>/meshpy-env
  # Activate the environment
  source <path-to-env-folder>/meshpy-env/bin/activate
  ```

### Install MeshPy from GitHub (most recent version)

If you want to install the current `main` version of MeshPy directly from GitHub, simply run:
```bash
pip install git+https://github.com/imcs-compsim/meshpy.git@main
```

### Install MeshPy from source

You can either install MeshPy directly from the source in a non-editable and editable fashion like:
- Non-editable:
  This allows you to use MeshPy, but changing the source code will not have any effect on the installed package
  ```bash
  git clone git@github.com:imcs-compsim/meshpy.git
  cd meshpy
  pip install .
  ```
- Editable:
  This allows you to change the source code without reinstalling the module
  ```bash
  git clone git@github.com:imcs-compsim/meshpy.git
  cd meshpy
  pip install -e .
  ```
Now you are able to use MeshPy. A good way to get started is by going through the examples
```bash
jupyter notebook examples/
```
If you also want to execute the associated test suite check out our [development](#developing-meshpy) section.


## Optional dependencies

### [4C](https://www.4c-multiphysics.org)

MeshPy can run 4C simulations directly from within a Python script, allowing for full control over arbitrarily complex simulation workflows. Fore more information, please have a look at the `meshpy.four_c.run_four_c` module.

### [CubitPy](https://github.com/imcs-compsim/cubitpy)

CubitPy is a Python library that contains utility functions extending the Cubit/Coreform Python interface. Furthermore, it allows for the easy creation of 4C-compatible input files directly from within Python. MeshPy can import meshes created with CubitPy and allows for further modification and manipulation of them.

CubitPy can be installed as an optional dependency with:
```bash
pip install -e .[cubitpy]
```

### [ArborX](https://github.com/arborx/ArborX) geometric search

MeshPy can optionally execute its geometric search functions using the C++ library [ArborX](https://github.com/arborx/ArborX).
First make sure the [pybind11](https://pybind11.readthedocs.io/en/stable/) submodule is loaded
```bash
cd <path_to_meshpy>
git submodule update --init
```
To setup MeshPy with ArborX, [CMake](https://cmake.org) and [Kokkos](https://kokkos.org) have to be available on your system (the preferred variant is via [Spack](https://spack.io/)).
Create a build directory
```bash
mkdir -p <path_to_meshpy>/src/build/geometric_search
```
Configure cmake and build the extension
```bash
cd <path_to_meshpy>/build/geometric_search
cmake ../../meshpy/geometric_search/src/
make -j4
```
> Note: Currently ArborX only works if MeshPy is installed in _editable_ mode.

## Developing MeshPy

If you want to actively develop MeshPy or run the test suite, you must install MeshPy in _editable_ (`-e`) mode and with our optional developer dependencies (`[dev]`) like
```bash
pip install -e ".[dev]" # Quotation marks are required for some shells
```
You can now run the MeshPy test suite to check that everything worked as expected
```bash
pytest
```

### Coding guidelines

- When working on MeshPy, use a leading underscore (`_`) to indicate functions, classes, and variables that are intended for internal use only. This is a coding convention rather than an enforced rule, so apply it where it improves code clarity, especially for functions that check consistency or modify internal states.
- To avoid ambiguous or incorrect imports when using MeshPy as a library, internal imports must follow a strict aliasing convention as illustrated below:
  <details>

  <summary>Import guidelines</summary>

  ```python
  # Not OK
  import numpy  # No alias
  import numpy as np  # Missing leading underscore

  from numpy import *  # Wildcard imports
  from numpy import _core  # We don't allow the import of private functionality
  from numpy.linalg import norm  # No alias
  from numpy import sin as sin2  # Missing leading underscore
  from meshpy.core.mesh import Mesh as _BeamMesh  # MeshPy imports have to be aliased with the same name, i.e., should be `_Mesh` (imports from third party libraries can be renamed)

  # OK
  import numpy as _np
  import sys as _sys

  from pathlib import Path as _Path

  from math import sin as _math_sin
  from numpy import sin as _np_sin

  import meshpy.core.conf as _conf
  from meshpy.core.mesh import Mesh as _Mesh
  from meshpy.core.node import Node as _Node
  from meshpy.core.node import NodeCosserat as _NodeCosserat
  ```
  </details>

### Testing

MeshPy provides a flexible testing system where additional tests can be enabled using specific flags. The following flags can be used with [pytest](https://pytest-cov.readthedocs.io/en/latest/config.html) to enable specific test sets:
 - `--exclude-standard-tests`: Disables the default test suite
 - `--4C`: Runs tests related to 4C integration
 - `--ArborX`: Enables tests for ArborX-related functionality
 - `--CubitPy`: Runs tests for CubitPy integration
 - `--performance-tests`: Includes performance tests

These flags can be combined arbitrarily; for example, to run the 4C, CubitPy, and ArborX tests but exclude the default test suite, use:
```bash
# 4C Tests require a path to a 4C executable
export MESHPY_FOUR_C_EXE=<path_to_4C>
# CubitPy Tests require a path to a Cubit/Coreform installation
export CUBIT_ROOT=<path_to_Cubit_or_Coreform>

pytest --4C --ArborX --CubitPy --exclude-standard-tests
```

### Cython geometric search

Some performance critical geometric search algorithms in MeshPy are written in [Cython](https://cython.readthedocs.io/en/stable/index.html). If Cython code is changed, it has to be recompiled. This can be done by running
```bash
python setup.py build_ext --inplace
```

## Contributing

If you are interested in contributing to MeshPy, we welcome your collaboration.
For general questions, feature request and bug reports please open an [issue](https://github.com/imcs-compsim/meshpy/issues).

If you contribute actual code, fork the repository and make the changes in a feature branch.
Depending on the topic and amount of changes you also might want to open an [issue](https://github.com/imcs-compsim/meshpy/issues).
To merge your changes into the MeshPy repository, create a pull request to the `main` branch.
A few things to keep in mind:
- Read our [coding guidelines](#coding-guidelines).
- It is highly encouraged to add tests covering the functionality of your changes, see the test suite in `tests/`.
- To maintain high code quality, MeshPy uses a number of different pre-commit hooks to check committed code. Make sure to set up the pre-commit hooks before committing your changes
  ```bash
  pre-commit install
  ```
- Check that you did not break anything by running the MeshPy tests.
  For most changes it should be sufficient to run the standard test suite:
  ```bash
  pytest
  ```
- Feel free to add yourself to the authors section in the [README.md](https://github.com/imcs-compsim/meshpy/blob/main/README.md) file.


## Authors

### Maintainers <!-- omit from toc -->
- Ivo Steinbrecher (@isteinbrecher)
- David Rudlstorfer (@davidrudlstorfer)

### Contributors (in alphabetical order) <!-- omit from toc -->
- Dao Viet Anh
- Max Firmbach (@maxfirmbach)
- Martin Frank (@knarfnitram)
- Nora Hagmeyer (@NoraHagmeyer)
- Matthias Mayr (@mayrmt)
- Gabriela Loera (@eulovi)
