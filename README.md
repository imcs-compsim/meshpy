# Meshpy

Meshpy is a general purpose 3D beam input generator.

## How to cite MeshPy?

Whenever you use or mention MeshPy in some sort of scientific document/publication/presentation, please cite MeshPy as described on the [MeshPy website](https://imcs-compsim.github.io/meshpy).


## Code formating

MeshPy uses the python code formatter [black](https://github.com/psf/black).
The testsuite checks if all files are formatted accordingly.

## Installation

Meshpy is developed with `python3.8`.
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

MeshPy can execute its geometric search functions using the C++ library [ArborX](https://github.com/arborx/ArborX).
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