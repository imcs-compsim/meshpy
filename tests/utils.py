# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# MeshPy: A beam finite element input generator
#
# MIT License
#
# Copyright (c) 2018-2024
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
# -----------------------------------------------------------------------------
"""
Define utility functions for the testing process.
"""

# Python imports.
import os
import numpy as np
import shutil
import subprocess
import warnings
import xml.etree.ElementTree as ET
import vtk
from vtk.util import numpy_support as vtk_numpy
from pyvista_utils.compare_grids import compare_grids


# MeshPy imports
from meshpy.utility import get_env_variable


def skip_fail_test(self, message):
    """
    Skip or fail the test depending if the test are run in GitHub or not.
    """
    if get_env_variable("TESTING_GITHUB", default="0") == "1":
        self.skipTest(message)
    else:
        self.skipTest(message)


# Define the testing paths
testing_path = os.path.abspath(os.path.dirname(__file__))
testing_input = os.path.join(testing_path, "reference-files")
testing_temp = os.path.join(testing_path, "testing-tmp")

# Check and clean the temporary directory.
os.makedirs(testing_temp, exist_ok=True)


def empty_testing_directory():
    """Delete all files in the testing directory, if it exists."""
    if os.path.isdir(testing_temp):
        for the_file in os.listdir(testing_temp):
            file_path = os.path.join(testing_temp, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(e)


def compare_string_tolerance(
    reference, compare, *, rtol=None, atol=None, split_string=" "
):
    """Compare two strings, all floating point values will be compared with a
    tolerance."""

    def set_tol(tol):
        if tol is None:
            return 0.0
        else:
            return tol

    rtol = set_tol(rtol)
    atol = set_tol(atol)

    lines_reference = reference.strip().split("\n")
    lines_compare = compare.strip().split("\n")
    n_reference = len(lines_reference)
    n_compare = len(lines_compare)
    if n_reference == n_compare:
        # Loop over each line in the file
        for i in range(n_reference):
            line_reference = lines_reference[i].strip().split(split_string)
            line_compare = lines_compare[i].strip().split(split_string)
            n_items_reference = len(line_reference)
            n_items_compare = len(line_compare)
            if n_items_reference == n_items_compare:
                # Loop over each entry in the line
                for j in range(n_items_reference):
                    try:
                        reference_number = float(line_reference[j].strip())
                        compare_number = float(line_compare[j].strip())
                        if np.isclose(
                            reference_number, compare_number, rtol=rtol, atol=atol
                        ):
                            pass
                        else:
                            return False
                    except ValueError:
                        if line_reference[j].strip() != line_compare[j].strip():
                            return False
            else:
                return False
    else:
        return False

    return True


def compare_test_result(
    self, result_string, *, extension="dat", additional_identifier=None, **kwargs
):
    """
    Compare a created string in a test with the reference results. The reference results
    are stored in a file made up of the test name.

    Args
    ----
    additional_identifier: str
        This can be set if there are more than 1 reference files for a single test
    extension: str
        File extension of the reference file
    """

    reference_file_name = self._testMethodName
    if additional_identifier is not None:
        reference_file_name += f"_{additional_identifier}"
    reference_file_name += "_reference"
    if extension is not None:
        reference_file_name += "." + extension
    reference_file_path = os.path.join(testing_input, reference_file_name)

    # Compare the results
    compare_strings(self, reference_file_path, result_string, **kwargs)


def compare_strings(self, reference, compare, *, rtol=None, atol=None, **kwargs):
    """Compare two stings. If they are not identical open a comparison and show the
    differences.
    """

    # Check if the input data is a file that exists.
    reference_is_file = os.path.isfile(reference)
    compare_is_file = os.path.isfile(compare)

    # Get the correct data
    if reference_is_file:
        with open(reference, "r") as myfile:
            reference_string = myfile.read()
    else:
        reference_string = reference

    if compare_is_file:
        with open(compare, "r") as myfile:
            compare_string = myfile.read()
    else:
        compare_string = compare

    if rtol is None and atol is None:
        # Check if the strings are equal, if not compare the differences and
        # fail the test.
        is_equal = reference_string.strip() == compare_string.strip()
    else:
        is_equal = compare_string_tolerance(
            reference_string, compare_string, rtol=rtol, atol=atol, **kwargs
        )
    if not is_equal and not (get_env_variable("TESTING_GITHUB", default="0") == "1"):
        # Check if temporary directory exists, and creates it if necessary.
        os.makedirs(testing_temp, exist_ok=True)

        # Get the paths of the files to compare. If a string was given
        # create a file with the string in it.
        if reference_is_file:
            reference_file = reference
        else:
            reference_file = os.path.join(
                testing_temp, "{}_reference.dat".format(self._testMethodName)
            )
            with open(reference_file, "w") as input_file:
                input_file.write(reference_string)

        if compare_is_file:
            compare_file = compare
        else:
            compare_file = os.path.join(
                testing_temp, "{}_compare.dat".format(self._testMethodName)
            )
            with open(compare_file, "w") as input_file:
                input_file.write(compare_string)

        if shutil.which("code") is not None:
            child = subprocess.Popen(
                ["code", "--diff", reference_file, compare_file], stderr=subprocess.PIPE
            )
            child.communicate()
        else:
            result = subprocess.run(
                ["diff", reference_file, compare_file], stdout=subprocess.PIPE
            )
            self._testMethodName += "\n\nDiff:\n" + result.stdout.decode("utf-8")

    # Check the results.
    self.assertTrue(is_equal, self._testMethodName)


def compare_vtk(self, path_1, path_2, *, tol_float=1e-14):
    """Compare two vtk files and raise an error if they are not equal."""

    def get_vtk(path):
        """
        Return a vtk object for the file at path.
        """
        reader = vtk.vtkXMLGenericDataObjectReader()
        reader.SetFileName(path)
        reader.Update()
        return reader.GetOutput()

    compare = compare_grids(
        get_vtk(path_1), get_vtk(path_2), output=True, tol=tol_float
    )
    self.assertTrue(compare[0], msg="\n".join(compare[1]))
