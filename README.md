# Meshpy

Meshpy is a beam input generator for BACI.

## Installation

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
python find_close_nodes_setup.py
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
python testing_main.py
```
