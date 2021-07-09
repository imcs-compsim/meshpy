# Meshpy

Meshpy is a general purpose 3D beam input generator.

## How to cite MeshPy?

Whenever you use or mention MeshPy in some sort of scientific document/publication/presentation, please cite MeshPy as described on the [MeshPy website](https://compsim.gitlab.io/codes/meshpy/index.html).


## Installation

Meshpy is developed with `python3.8`.
Other versions of Python might lead to issues.
It is recommended to use virtual environments with `python`.
On Debian systems the package `python3-venv` has to be installed.
```bash
sudo apt-get install python3-venv python3-dev
```

Now a virtual environment can be created (for example in the home directory)
```bash
cd ~
mkdir opt
cd opt
python3 -m venv meshpy-env
```

The created virtual environment can be loaded with
```bash
source ~/opt/meshpy-env/bin/activate
```

From now on we assume that the virtual enviroment is loaded.
To install `meshpy` go to the repository directory
```bash
cd path_to_meshpy
```

Run the following command to install the required packages
```bash
pip install -r requirements.txt
```

As a last step the `cython` code within `meshpy` has to be compiled
```bash
cd path_to_meshpy/meshpy
python3 find_close_points_setup.py
```

Add the meshpy path to `PYTHONPATH`
```bash
export PYTHONPATH=path_to_meshpy:$PYTHONPATH
```

Optional, a path to `baci-release` can be given in order to run some combined
tests with baci.
```bash
export BACI_RELEASE=path_to_baci-release
```

To check if everything worked as expected, run the tests
```bash
cd path_to_meshpy/tests
python3 testing_main.py
```

Also run the performance tests (the reference time values and host name might have to be adapted in the file `path_to_meshpy/tests/performance_testing.py`)
```bash
cd path_to_meshpy/tests
python3 performance_testing.py
```
