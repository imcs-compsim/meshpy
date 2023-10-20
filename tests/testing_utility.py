# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# MeshPy: A beam finite element input generator
#
# MIT License
#
# Copyright (c) 2018-2023
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


# Global variable if this test is run by GitHub.
if "TESTING_GITHUB" in os.environ.keys() and os.environ["TESTING_GITHUB"] == "1":
    TESTING_GITHUB = True
else:
    TESTING_GITHUB = False


def skip_fail_test(self, message):
    """
    Skip or fail the test depending if the test are run in GitHub or not.
    """
    if TESTING_GITHUB:
        self.skipTest(message)
    else:
        self.skipTest(message)


def get_baci_path():
    """Look for and return a path to baci-release."""

    if "BACI_RELEASE" in os.environ.keys():
        path = os.environ["BACI_RELEASE"]
    else:
        path = ""

    # Check if the path exists.
    if os.path.isfile(path):
        return path
    else:
        # In the case that no path was found, check if the script is performed
        # by a GitHub runner.
        if TESTING_GITHUB:
            raise ValueError("Path to baci-release not found!")
        else:
            warnings.warn(
                "Path to baci-release not found. Did you set the "
                + "environment variable BACI_RELEASE?"
            )
            return None


# Define the testing paths.
testing_path = os.path.abspath(os.path.dirname(__file__))
testing_input = os.path.join(testing_path, "reference-files")
testing_temp = os.path.join(testing_path, "testing-tmp")
baci_release = get_baci_path()

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


def compare_string_tolerance(reference, compare, tol):
    """Compare two strings, all floating point values will be compared with an
    absolute tolerance."""

    lines_reference = reference.strip().split("\n")
    lines_compare = compare.strip().split("\n")
    n_reference = len(lines_reference)
    n_compare = len(lines_compare)
    if n_reference == n_compare:
        # Loop over each line in the file
        for i in range(n_reference):
            line_reference = lines_reference[i].strip().split(" ")
            line_compare = lines_compare[i].strip().split(" ")
            n_items_reference = len(line_reference)
            n_items_compare = len(line_compare)
            if n_items_reference == n_items_compare:
                # Loop over each entry in the line
                for j in range(n_items_reference):
                    try:
                        reference_number = float(line_reference[j].strip())
                        compare_number = float(line_compare[j].strip())
                        if np.abs(reference_number - compare_number) < tol:
                            pass
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
    compare_strings(self, None, reference_file_path, result_string, **kwargs)


def compare_strings(self, name, reference, compare, *, tol=None):
    """
    Compare two stings. If they are not identical open meld and show the
    differences.
    """

    # TODO: improve the name parameter given to this function, check if it makes sense

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

    if tol is None:
        # Check if the strings are equal, if not compare the differences and
        # fail the test.
        is_equal = reference_string.strip() == compare_string.strip()
    else:
        is_equal = compare_string_tolerance(reference_string, compare_string, tol)
    if not is_equal and not TESTING_GITHUB:

        # Check if temporary directory exists, and creates it if necessary.
        os.makedirs(testing_temp, exist_ok=True)

        # Get the paths of the files to compare. If a string was given
        # create a file with the string in it.
        if reference_is_file:
            reference_file = reference
        else:
            reference_file = os.path.join(testing_temp, "{}_reference.dat".format(name))
            with open(reference_file, "w") as input_file:
                input_file.write(reference_string)

        if compare_is_file:
            compare_file = compare
        else:
            compare_file = os.path.join(testing_temp, "{}_compare.dat".format(name))
            with open(compare_file, "w") as input_file:
                input_file.write(compare_string)

        if shutil.which("meld") is not None:
            child = subprocess.Popen(
                ["meld", reference_file, compare_file], stderr=subprocess.PIPE
            )
            child.communicate()
        else:
            result = subprocess.run(
                ["diff", reference_file, compare_file], stdout=subprocess.PIPE
            )
            name += "\n\nDiff:\n" + result.stdout.decode("utf-8")

    # Check the results.
    self.assertTrue(is_equal, name)


def compare_vtk_data(path1, path2, *, raise_error=False, tol_float=None):
    """
    Compare the vtk files at path1 and path2, by compairing the stored data.

    Args
    ----
    raise_error: bool
        If true, then an error will be raised in case the files do not match.
        Otherwise False will be returned.
    tol_float: None / float
        If given, numbers will be considered equal if the difference between
        them is smaller than tol_float.
    """

    # Check that both arguments are paths and exist.
    if not (os.path.isfile(path1) and os.path.isfile(path2)):
        raise ValueError("The paths given are not OK!")

    # Default value for the numerical tolerance.
    if tol_float is None:
        raise ValueError("Tolerance in compare_vtk_data has to be given!")

    def get_vtk(path):
        """
        Return a vtk object for the file at path.
        """
        reader = vtk.vtkXMLGenericDataObjectReader()
        reader.SetFileName(path)
        reader.Update()
        return reader.GetOutput()

    def compare_arrays(array1, array2, name=None):
        """
        Compare two vtk arrays.
        """

        diff = vtk_numpy.vtk_to_numpy(array1) - vtk_numpy.vtk_to_numpy(array2)
        if not np.max(np.abs(diff)) < tol_float:
            error_string = "VTK array comparison failed!"
            if name is not None:
                error_string += " Name of the array: {}".format(name)
            raise ValueError(error_string)

    def compare_data_sets(data1, data2):
        """
        Compare data sets obtained from vtk objects.
        """

        # Both data sets need to have the same number of arrays.
        if not data1.GetNumberOfArrays() == data2.GetNumberOfArrays():
            raise ValueError("Length of vtk data objects do not match!")

        # Compare each array.
        for i in range(data1.GetNumberOfArrays()):

            # Get the arrays with the same name.
            name = data1.GetArrayName(i)
            array1 = data1.GetArray(name)
            array2 = data2.GetArray(name)
            compare_arrays(array1, array2, name=name)

    # Perform all checks, catch errors.
    try:
        # Load the vtk files.
        data1 = get_vtk(path1)
        data2 = get_vtk(path2)

        # Compare the point positions.
        compare_arrays(
            data1.GetPoints().GetData(),
            data2.GetPoints().GetData(),
            name="point_positions",
        )

        # Compare the cell and point data of the array.
        compare_data_sets(data1.GetCellData(), data2.GetCellData())
        compare_data_sets(data1.GetPointData(), data2.GetPointData())

        # Compare the cell connectivity.
        compare_arrays(
            data1.GetCells().GetData(),
            data2.GetCells().GetData(),
            name="cell_connectivity",
        )

        # Compare the cell types.
        compare_arrays(
            data1.GetCellTypesArray(), data2.GetCellTypesArray(), name="cell_type"
        )

    except Exception as error:
        if raise_error:
            raise error
        return False

    return True


def compare_vtk(self, name, ref_file, vtk_file, *, tol_float=1e-14):
    """Compare two vtk files and raise an error if they are not equal."""
    self.assertTrue(compare_vtk_data(ref_file, vtk_file, tol_float=tol_float), name)
