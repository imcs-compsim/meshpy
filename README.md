# MeshPy

MeshPy is a general purpose 3D beam finite element input generator written in `python3`.
It contains basic geometry creation and manipulation functions to create complex beam geometries, including rotational degrees of freedom for the beam nodes.
It can be used to create input files for the following finite element solvers:
- [4C](https://www.4c-multiphysics.org/) (academic finite element solver)
- [Abaqus](https://en.wikipedia.org/wiki/Abaqus) (commercial software package)

MeshPy can easily be adapted to create input files for other solvers.
MeshPy is developed at the [Institute for Mathematics and Computer-Based Simulation (IMCS)](https://www.unibw.de/imcs-en) at the Universität der Bundeswehr München.

## How to cite MeshPy?

Whenever you use or mention MeshPy in some sort of scientific document/publication/presentation, please cite MeshPy as described on the [MeshPy website](https://imcs-compsim.github.io/meshpy).

Feel free to leave a :star: on [GitHub](https://github.com/imcs-compsim/meshpy).

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
- Feel free to add yourself to the [CONTRIBUTORS](CONTRIBUTORS) file.


## Installation

MeshPy is developed with `python3.12`.
Other versions of Python might lead to issues.
It is recommended to use virtual environments with `python`.
On Debian systems the following packages have to be installed
```bash
sudo apt-get install python3-venv python3-dev
```

Now a virtual environment can be created (chose an appropriate directory for this, e.g., `/home/user/opt`)

```bash
python3 -m venv meshpy-env
```

The created virtual environment can be loaded with
```bash
source meshpy-env/bin/activate
```

From now on we assume that the virtual enviroment is loaded.
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
python3 setup.py build_ext --inplace
```

Optional, a path to `baci-release` can be given in order to run some combined
tests with baci
```bash
export BACI_RELEASE=path_to_baci-release
```

To check if everything worked as expected, run the tests
```bash
cd <path_to_meshpy>/tests
python3 testing_main.py
```

Also run the performance tests (the reference time values and host name might have to be adapted in the file `<path_to_meshpy>/tests/performance_testing.py`)
```bash
cd <path_to_meshpy>/tests
python3 performance_testing.py
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
cd <path_to_meshpy>/tests
python3 testing_geometric_search.py
```
