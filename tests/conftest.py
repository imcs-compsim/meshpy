# The MIT License (MIT)
#
# Copyright (c) 2018-2025 MeshPy Authors
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Testing framework infrastructure."""

import json
import os
import shutil
import subprocess
from difflib import unified_diff
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import pytest
import vtk
import yaml
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from vtk_utils.compare_grids import compare_grids

from meshpy.core.conf import mpy
from meshpy.four_c.input_file import InputFile


def pytest_addoption(parser: Parser) -> None:
    """Add custom command line options to pytest.

    Args:
        parser: Pytest parser
    """

    parser.addoption(
        "--4C",
        action="store_true",
        default=False,
        help="Execute standard and 4C based tests.",
    )

    parser.addoption(
        "--ArborX",
        action="store_true",
        default=False,
        help="Execute standard and ArborX based tests.",
    )

    parser.addoption(
        "--CubitPy",
        action="store_true",
        default=False,
        help="Execute standard and CubitPy based tests.",
    )

    parser.addoption(
        "--performance-tests",
        action="store_true",
        default=False,
        help="Execute standard and performance tests.",
    )

    parser.addoption(
        "--exclude-standard-tests",
        action="store_true",
        default=False,
        help="Exclude standard tests.",
    )


def pytest_collection_modifyitems(config: Config, items: list) -> None:
    """Filter tests based on their markers and provided command line options.

    Currently configured options:
        `pytest`: Execute standard tests with no markers
        `pytest --4C`: Execute standard tests and tests with the `fourc` marker
        `pytest --ArborX`: Execute standard tests and tests with the `arborx` marker
        `pytest --CubitPy`: Execute standard tests and tests with the `cubitpy` marker
        `pytest --performance-tests`: Execute standard tests and tests with the `performance` marker
        `pytest --exclude-standard-tests`: Execute tests with any other marker and exclude the standard unmarked tests

    Args:
        config: Pytest config
        items: Pytest list of tests
    """

    selected_tests = []

    # loop over all collected tests
    for item in items:
        # Get all set markers for current test (e.g. `fourc_arborx`, `cubitpy`, `performance`, ...)
        # We don't care about the "parametrize" marker here
        markers = [
            marker.name
            for marker in item.iter_markers()
            if not marker.name == "parametrize"
        ]

        for flag, marker in zip(
            ["--4C", "--ArborX", "--CubitPy", "--performance-tests"],
            ["fourc", "arborx", "cubitpy", "performance"],
        ):
            if config.getoption(flag) and marker in markers:
                selected_tests.append(item)

        if not markers and not config.getoption("--exclude-standard-tests"):
            selected_tests.append(item)

    deselected_tests = list(set(items) - set(selected_tests))

    items[:] = selected_tests
    config.hook.pytest_deselected(items=deselected_tests)


@pytest.fixture(autouse=True)
def run_before_each_test():
    """Reset the global mpy object before each test."""
    mpy.set_default_values()


@pytest.fixture(scope="session")
def reference_file_directory() -> Path:
    """Provide the path to the reference file directory.

    Returns:
        Path: A Path object representing the full path to the reference file directory.
    """

    testing_path = Path(__file__).resolve().parent
    return testing_path / "reference-files"


@pytest.fixture(scope="function")
def current_test_name(request: pytest.FixtureRequest) -> str:
    """Return the name of the current pytest test.

    Args:
        request: The pytest request object.

    Returns:
        str: The name of the current pytest test.
    """

    return request.node.originalname


@pytest.fixture(scope="function")
def get_string() -> Callable:
    """Return function to get string from different types of input.

    Necessary to enable the function call through pytest fixtures.

    Returns:
        Function to get string from file.
    """

    def _get_string(
        data: Union[Path, str, InputFile, dict], input_file_kwargs: dict = {}
    ) -> str:
        """Get string from file, string or InputFile.

        Args:
            data: Object that should be converted to a string.
            input_file_kwargs: Dictionary which contains the settings when extracting
                the string from the input file.

        Returns:
            String representation of data.
        """

        if isinstance(data, str):
            return data
        elif isinstance(data, Path):
            if not data.is_file():
                raise FileNotFoundError(f"File {data} does not exist!")
            with open(data, "r") as file:
                return file.read()
        elif isinstance(data, InputFile):
            return data.get_string(**input_file_kwargs)
        elif isinstance(data, dict):

            def convert_numpy(obj):
                """Convert a numpy array to a json serializable data
                structure."""
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                raise TypeError(
                    "Object of type {} is not JSON serializable".format(type(obj))
                )

            return json.dumps(data, indent=2, default=convert_numpy, sort_keys=True)

        else:
            raise TypeError(f"Data type {type(data)} not implemented.")

    return _get_string


@pytest.fixture(scope="function")
def get_corresponding_reference_file_path(
    reference_file_directory, current_test_name
) -> Callable:
    """Return function to get path to corresponding reference file for each
    test.

    Necessary to enable the function call through pytest fixtures.
    """

    def _get_corresponding_reference_file_path(
        reference_file_base_name: Optional[str] = None,
        additional_identifier: Optional[str] = None,
        extension: str = "4C.yaml",
    ) -> Path:
        """Get path to corresponding reference file for each test. Also check
        if this file exists. Basename, additional identifier and extension can
        be adjusted.

        Args:
            reference_file_base_name: Basename of reference file, if none is
                provided the current test name is utilized
            additional_identifier: Additional identifier for reference file, by default none
            extension: Extension of reference file, by default ".dat"

        Returns:
            Path to reference file.
        """

        corresponding_reference_file = reference_file_base_name or current_test_name

        if additional_identifier:
            corresponding_reference_file += f"_{additional_identifier}"

        corresponding_reference_file += "." + extension

        corresponding_reference_file_path = (
            reference_file_directory / corresponding_reference_file
        )

        if not os.path.isfile(corresponding_reference_file_path):
            raise AssertionError(
                f"File path: {corresponding_reference_file_path} does not exist"
            )

        return corresponding_reference_file_path

    return _get_corresponding_reference_file_path


@pytest.fixture(scope="function")
def assert_results_equal(get_string, tmp_path, current_test_name) -> Callable:
    """Return function to compare either string or files.

    Necessary to enable the function call through pytest fixtures.

    Args:
        get_string: Function to get string from file.
        tmp_path: Temporary path for testing.
        current_test_name: Name of the current test.

    Returns:
        Function to compare results.
    """

    def _assert_results_equal(
        reference: Union[Path, str, dict],
        result: Union[Path, str, dict, InputFile],
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        input_file_kwargs: dict = {
            "check_nox": False,
            "header": False,
        },
        **kwargs,
    ) -> None:
        """Comparison between reference and result with relative or absolute
        tolerance.

        If the comparison fails, an assertion is raised.

        Args:
            reference: The reference data.
            result: The result data.
            rtol: The relative tolerance.
            atol: The absolute tolerance.
            input_file_kwargs: Dictionary which contains the settings when extracting
                the string from the input file.
        """

        # Per default we do a string comparison of the objects. Some data types, e.g.,
        # vtu files or hdf5 data structures require special comparison functions.
        # We first check if a special case is needed, if not we default to the
        # string comparison.

        if isinstance(reference, Path) and isinstance(result, Path):
            if reference.suffix != result.suffix:
                raise RuntimeError(
                    "Reference and result file must be of same file type!"
                )
            elif reference.suffix in [".vtk", ".vtu"]:
                compare_vtk_files(reference, result, rtol, atol)
                return
            elif reference.suffix == ".dat":
                # Do nothing here as the mechanism below will compare the dat files.
                pass
            else:
                raise NotImplementedError(
                    f"Comparison is not yet implemented for {reference.suffix} files."
                )

        # Dictionary comparison
        if isinstance(reference, dict) or isinstance(result, dict):

            def get_dictionary(data) -> dict:
                """Get the dictionary representation of the data object."""
                if isinstance(data, dict):
                    return data
                elif isinstance(data, Path):
                    return json.loads(get_string(data))
                raise TypeError(
                    f"The comparison for {type(data)} is not yet implemented."
                )

            reference_dict = get_dictionary(reference)
            result_dict = get_dictionary(result)
            compare_dicts(reference_dict, result_dict, rtol=rtol, atol=atol)
            return

        if isinstance(reference, InputFile) or isinstance(result, InputFile):

            def get_dictionary(data) -> dict:
                """Get the dictionary representation of the data object."""
                if isinstance(data, InputFile):
                    return data.dump()
                if isinstance(data, dict):
                    return data
                elif isinstance(data, Path):
                    with open(data) as stream:
                        return yaml.safe_load(stream)
                raise TypeError(
                    f"The comparison for {type(data)} is not yet implemented."
                )

            reference_dict = get_dictionary(reference)
            result_dict = get_dictionary(result)
            compare_dicts(reference_dict, result_dict, rtol=rtol, atol=atol)
            return

        if isinstance(reference, list) or isinstance(result, list):
            compare_lists(reference, result, rtol=rtol, atol=atol)
            return

        # We didn't raise an error or exit this function yet, so we default to a string
        # based comparison.
        [reference_string, result_string] = [
            get_string(data, input_file_kwargs) for data in [reference, result]
        ]

        # compare strings and handle non-matching strings
        try:
            compare_strings(reference_string, result_string, rtol, atol, **kwargs)
        except AssertionError as error:
            if isinstance(reference, Path):
                handle_unequal_strings(
                    tmp_path, current_test_name, result_string, reference
                )
            raise AssertionError(str(error))

    return _assert_results_equal


def compare_strings(
    reference: str,
    result: str,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    string_splitter: str = " ",
) -> None:
    """Compare if two strings are identical, optionally within a given
    tolerance. If the comparison fails, an error is raised.

    Args:
        reference: The reference string.
        result: The result string.
        rtol: The relative tolerance.
        atol: The absolute tolerance.
        string_splitter: With which string the strings are split.
    """

    if rtol is None and atol is None:
        compare_strings_equality_assert(reference, result)
    else:
        compare_strings_with_tolerance_assert(
            reference, result, rtol, atol, string_splitter=string_splitter
        )


def compare_strings_equality_assert(reference: str, result: str) -> None:
    """Check if two strings are exactly equal, if not raise an error.

    Args:
        reference: The reference string.
        result: The result string.
    """
    diff = list(unified_diff(reference.splitlines(), result.splitlines(), lineterm=""))
    if diff:
        raise AssertionError(
            "Exact string comparison failed! Difference between reference and result: \n"
            + "\n".join(list(diff))
        )


def compare_strings_with_tolerance_assert(
    reference: str,
    result: str,
    rtol: Optional[float],
    atol: Optional[float],
    string_splitter=" ",
) -> None:
    """Compare if two strings are identical within a given tolerance.

    Args:
        reference: The reference string.
        result: The result string.
        rtol: The relative tolerance.
        atol: The absolute tolerance.
        string_splitter: With which string the strings are split.
    """

    rtol = 0.0 if rtol is None else rtol
    atol = 0.0 if atol is None else atol

    lines_reference = reference.strip().split("\n")
    lines_result = result.strip().split("\n")

    if len(lines_reference) != len(lines_result):
        raise AssertionError(
            f"String comparison with tolerance failed!\n"
            + f"Number of lines in reference and result differ: {len(lines_reference)} != {len(lines_result)}"
        )

    # Loop over each line in the file
    for line_reference, line_result in zip(lines_reference, lines_result):
        line_reference_splits = line_reference.strip().split(string_splitter)
        line_result_splits = line_result.strip().split(string_splitter)

        if len(line_reference_splits) != len(line_result_splits):
            raise AssertionError(
                f"String comparison with tolerance failed!\n"
                + f"Number of items in reference and result line differ!\n"
                + f"Reference line: {line_reference}\n"
                + f"Result line:    {line_result}"
            )

        # Loop over each entry in the line
        for item_reference, item_result in zip(
            line_reference_splits, line_result_splits
        ):
            try:
                number_reference = float(item_reference.strip())
                number_result = float(item_result.strip())
                if np.isclose(number_reference, number_result, rtol=rtol, atol=atol):
                    pass
                else:
                    raise AssertionError(
                        f"String comparison with tolerance failed!\n"
                        + f"Numbers do not match within given tolerance!\n"
                        + f"Reference line: {line_reference}\n"
                        + f"Result line:    {line_result}"
                    )

            except ValueError:
                if item_reference.strip() != item_result.strip():
                    raise AssertionError(
                        f"String comparison with tolerance failed!\n"
                        + f"Strings do not match in line!\n"
                        + f"Reference line: {line_reference}\n"
                        + f"Result line:    {line_result}"
                    )


def compare_vtk_files(
    reference: Path, result: Path, rtol: Optional[float], atol: Optional[float]
) -> None:
    """Compare two VTK files for equality within a given tolerance.

    Args:
        reference: The path to the reference VTK file.
        result: The path to the result VTK file to be compared.
        rtol: The relative tolerance parameter.
        atol: The absolute tolerance parameter.
    """

    compare = compare_grids(
        get_vtk(reference), get_vtk(result), output=True, rtol=rtol, atol=atol
    )

    if not compare[0]:
        raise AssertionError("\n".join(compare[1]))


def get_vtk(path: Path) -> vtk.vtkDataObject:
    """Return vtk data object for given vtk file.

    Args:
        path: Path to .vtu/.vtk file.

    Returns:
        vtk.vtkDataObject: VTK data object.
    """

    reader = vtk.vtkXMLGenericDataObjectReader()
    reader.SetFileName(path)
    reader.Update()
    return reader.GetOutput()


def handle_unequal_strings(
    tmp_path: Path,
    current_test_name: str,
    result: str,
    reference_path: Path,
) -> None:
    """Handle unequal string comparison. Print error message to console, write
    new result file to temporary pytest directory and open VSCode diff tool if
    local development is used.

    Args:
        tmp_path: Temporary pytest directory
        current_test_name: Name of the current test
        result: "New" result string
        reference_path: Path to "old" reference file
    """

    # save result string to file
    result_path = tmp_path / (current_test_name + "_result.txt")
    with open(result_path, "w") as file:
        file.write(result)
    print(f"Result string saved to: '{result_path}'.")

    # open VSCode diff tool if available
    if shutil.which("code") is not None:
        child = subprocess.Popen(
            ["code", "--diff", result_path, reference_path],
            stderr=subprocess.PIPE,
        )
        child.communicate()


def compare_dicts(
    dict_1: dict,
    dict_2: dict,
    *,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
):
    """Recursively compare two dictionaries.

    For NumPy arrays, use np.allclose. For other types, use direct equality.

    If the dictionaries are not equal, an assertion error is raised.

    Args:
        dict_1: The first dictionary to compare.
        dict_2: The second dictionary to compare.
        rtol: Relative tolerance for np.allclose.
        atol: Absolute tolerance for np.allclose.
    """

    if rtol is None:
        rtol = 1e-10
    if atol is None:
        atol = 1e-10

    if not isinstance(dict_1, dict) or not isinstance(dict_2, dict):
        raise ValueError("Both arguments must be dictionaries")

    if dict_1.keys() != dict_2.keys():
        raise AssertionError(
            "The keys of the dictionary are not equal. "
            f"Got {dict_1.keys()} and {dict_2.keys()}"
        )

    for key in dict_1:
        value_1 = dict_1[key]
        value_2 = dict_2[key]

        if isinstance(value_1, np.ndarray) or isinstance(value_2, np.ndarray):
            try:
                if not np.allclose(value_1, value_2, rtol=rtol, atol=atol):
                    raise AssertionError(
                        f'Comparison of numpy arrays for the key "{key}" failed.'
                    )
            except:
                raise AssertionError(
                    f'Comparison of numpy arrays for the key "{key}" failed.'
                )
        elif isinstance(value_1, dict) and isinstance(value_2, dict):
            # If both values are dictionaries, compare recursively
            compare_dicts(value_1, value_2, rtol=rtol, atol=atol)
        elif (
            isinstance(value_1, float)
            or isinstance(value_2, float)
            or isinstance(value_1, np.generic)
            or isinstance(value_2, np.generic)
        ):
            # If one of the values is a float, we do a float comparison
            try:
                if not np.isclose(value_1, value_2, rtol=rtol, atol=atol):
                    raise AssertionError(
                        f"Comparison of the values {value_1} and {value_2} failed."
                    )
            except:
                raise AssertionError(
                    f"Comparison of the values {value_1} and {value_2} failed."
                )

        elif isinstance(value_1, list) and isinstance(value_2, list):
            # If both values are dictionaries, compare recursively
            compare_lists(value_1, value_2, rtol=rtol, atol=atol)
        elif not value_1 == value_2:
            raise AssertionError(
                f'Comparison of values for the key "{key}" failed. Values: {value_1} and {value_2}'
            )


def compare_lists(
    list_1: list,
    list_2: list,
    *,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
):
    """Recursively compare two dictionaries.

    For NumPy arrays, use np.allclose. For other types, use direct equality.

    If the dictionaries are not equal, an assertion error is raised.

    Args:
        dict_1: The first dictionary to compare.
        dict_2: The second dictionary to compare.
        rtol: Relative tolerance for np.allclose.
        atol: Absolute tolerance for np.allclose.
    """

    if rtol is None:
        rtol = 1e-10
    if atol is None:
        atol = 1e-10

    assert len(list_1) == len(list_2)

    for item_1, item_2 in zip(list_1, list_2):
        if isinstance(item_1, np.ndarray) or isinstance(item_2, np.ndarray):
            if not np.allclose(item_1, item_2, rtol=rtol, atol=atol):
                raise AssertionError("Comparison of numpy arrays for the key failed.")
        elif isinstance(item_1, dict) and isinstance(item_2, dict):
            # If both values are dictionaries, compare recursively
            compare_dicts(item_1, item_2, rtol=rtol, atol=atol)
        elif (
            isinstance(item_1, float)
            or isinstance(item_2, float)
            or isinstance(item_1, np.generic)
            or isinstance(item_2, np.generic)
        ):
            # If one of the values is a float, we do a float comparison
            if not np.isclose(item_1, item_2, rtol=rtol, atol=atol):
                raise AssertionError(
                    f"Comparison of the values {item_1} and {item_2} failed."
                )
        elif isinstance(item_1, list) and isinstance(item_2, list):
            # If both values are dictionaries, compare recursively
            compare_lists(item_1, item_2, rtol=rtol, atol=atol)
        elif isinstance(item_1, str) and isinstance(item_2, str):
            # If both values are dictionaries, compare recursively
            compare_strings(item_1, item_2, rtol=rtol, atol=atol)
        elif not item_1 == item_2:
            raise AssertionError(
                f"Comparison of values for the key failed. {item_1} {item_2}"
            )
