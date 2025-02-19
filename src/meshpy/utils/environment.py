# MeshPy: A beam finite element input generator
#
# MIT License
#
# Copyright (c) 2018-2025
#     Ivo Steinbrecher
#     Institute for Mathematics and Computer-Based Simulation
#     Universitaet der Bundeswehr Muenchen
#     https://www.unibw.de/imcs-en
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Helper functions to interact with the MeshPy environment."""

import importlib.util
import os


def is_mybinder():
    """Check if the current environment is running on mybinder."""
    return "BINDER_LAUNCH_HOST" in os.environ.keys()


def is_testing():
    """Check if the current environment is a pytest testing run."""
    return "PYTEST_CURRENT_TEST" in os.environ


def is_arborx_available() -> bool:
    """Check if ArborX is installed.

    Returns:
        True if ArborX is installed, False otherwise
    """

    if (
        importlib.util.find_spec("meshpy.geometric_search.geometric_search_arborx_lib")
        is None
    ):
        return False
    return True


def is_cubitpy_available() -> bool:
    """Check if CubitPy is installed.

    Returns:
        True if CubitPy is installed, False otherwise
    """

    if importlib.util.find_spec("cubitpy") is None:
        return False
    return True


def is_cython_available() -> bool:
    """Check if Cython is installed.

    Returns:
        True if Cython is installed, False otherwise
    """

    if importlib.util.find_spec("cython") is None:
        return False
    return True


def get_env_variable(name, *, default="default_not_set"):
    """Return the value of an environment variable.

    Args
    ----
    name: str
        Name of the environment variable
    default:
        Value to be returned if the given named environment variable does
        not exist. If this is not set and the name is not in the env
        variables, then an error will be thrown.
    """
    if name in os.environ.keys():
        return os.environ[name]
    elif default == "default_not_set":
        raise ValueError(f"Environment variable {name} is not set")
    return default
