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
"""Compile the Cython code of MeshPy and move it to the installed location."""

import glob
import importlib.util
import os
import shutil

from meshpy.utils.environment import is_cython_available


def compile_cython() -> bool:  # pragma: no cover
    """Compile the Cython code in the geometric_search_cython_lib.pyx file.

    Returns:
        bool: True if Cython is available and the was compiled. False if Cython is not available
    """

    if not is_cython_available():
        return False

    import numpy as np
    from Cython.Build import cythonize
    from setuptools import Extension, setup

    extensions = [
        Extension(
            "meshpy.geometric_search.geometric_search_cython_lib",
            ["src/meshpy/geometric_search/geometric_search_cython_lib.pyx"],
            include_dirs=[np.get_include()],
        )
    ]

    setup(
        script_args=["build_ext", "--inplace"],
        ext_modules=cythonize(
            extensions,
            build_dir="src/build/cython_generated_code",
            annotate=True,
        ),
    )

    return True


def copy_to_installed_location():  # pragma: no cover
    """Copy the compiled Cython code to the installed location of MeshPy.

    This is crucial in non-editable installs of MeshPy because the
    source files are copied to the site-packages directory.
    """

    # Get the location from where the files are imported
    origin_dir = os.path.dirname(
        importlib.util.find_spec("meshpy.geometric_search").origin
    )

    # Get the compiled cython file
    cython_file = glob.glob(
        "src/meshpy/geometric_search/geometric_search_cython_lib.*.so"
    ) + glob.glob("src/meshpy/geometric_search/geometric_search_cython_lib.*.pyd")

    if len(cython_file) != 1:
        print("No or multiple compiled cython files found in src directory!")
        return

    # Move the newly compiled file to the installed location
    shutil.move(
        cython_file[0], os.path.join(origin_dir, os.path.basename(cython_file[0]))
    )


def main() -> None:  # pragma: no cover
    """Compile the Cython Code of MeshPy."""

    if not compile_cython():
        print(
            "No Cython install found!\n"
            "If you intend to use the compiled Cython code follow these steps:\n"
            "    1. Install MeshPy with Cython: pip install .[cython]\n"
            "    2. Compile the Cython code with this script."
        )

        return

    # copy the compiled code the installed location of MeshPy
    # (for non editable installs this is the site-packages directory)
    copy_to_installed_location()

    # Verify that the import is working
    try:
        import meshpy.geometric_search.geometric_search_cython_lib

        print("\nCython code compiled and imported successfully!")

    except ImportError:
        print("\nCython code compilation or installation failed!")
