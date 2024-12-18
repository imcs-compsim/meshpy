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
"""Testing framework infrastructure."""

import os
from difflib import unified_diff
from pathlib import Path
from typing import Callable, Union

import numpy as np
import pytest
import vtk
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from vtk_utils.compare_grids import compare_grids


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
        # Get all set markers for current test (e.g. `fourc_arborx`, `cubitpy`, `performance`)
        markers = [marker.name for marker in item.iter_markers()]

        if config.getoption("--4C") and "fourc" in markers:
            selected_tests.append(item)

        if config.getoption("--ArborX") and "arborx" in markers:
            selected_tests.append(item)

        if config.getoption("--CubitPy") and "cubitpy" in markers:
            selected_tests.append(item)

        if config.getoption("--performance-tests") and "performance" in markers:
            selected_tests.append(item)

        if not markers and not config.getoption("--exclude-standard-tests"):
            selected_tests.append(item)

    deselected_tests = list(set(items) - set(selected_tests))

    items[:] = selected_tests
    config.hook.pytest_deselected(items=deselected_tests)


@pytest.fixture(scope="session")
def reference_file_directory() -> Path:
    """Provide the path to the reference file directory.

    Returns:
        Path: A Path object representing the full path to the reference file directory.
    """

    testing_path = os.path.abspath(os.path.dirname(__file__))
    return Path(testing_path) / "reference-files"


@pytest.fixture(scope="function")
def current_test_name(request: pytest.FixtureRequest) -> str:
    """Return the name of the current pytest test.

    Args:
        request: The pytest request object.

    Returns:
        str: The name of the current pytest test.
    """

    return request.node.name


@pytest.fixture(scope="function")
def get_string() -> Callable:
    """Return function to get string from file.

    Necessary to enable the function call through pytest fixtures.
    """

    def _get_string(path: Path) -> str:
        """Get string from file.

        Args:
            path: Path to file.

        Returns:
            String of file
        """

        if not path.is_file():
            raise FileNotFoundError(f"File {path} does not exist!")

        with open(path, "r") as file:
            return file.read()

    return _get_string


@pytest.fixture(scope="function")
def compare_results() -> Callable:
    """Return function to compare either string or files.

    Necessary to enable the function call through pytest fixtures.
    """

    def _compare(
        reference: Union[Path, str],
        result: Union[Path, str],
        rtol: float = 1e-14,
        atol: float = 1e-14,
        **kwargs,
    ) -> bool:
        """Comparison between reference and result with relative or absolute
        tolerance.

        Args:
            reference: The reference string or path to the reference file.
            result: The result string or path to the result file.
            rtol: The relative tolerance.
            atol: The absolute tolerance.

        Returns:
            bool: true if comparison is successful, false otherwise
        """

        if type(reference) is not type(result):
            raise TypeError("Reference and result must be of the same type.")
        elif type(reference) is str and type(result) is str:
            return compare_strings(reference, result, rtol, atol, **kwargs)
        elif isinstance(reference, Path) and isinstance(result, Path):
            if reference.suffix != result.suffix:
                raise RuntimeError(
                    "Reference and result file must be of same file type!"
                )
            elif reference.suffix in [".vtk", ".vtu"]:
                return compare_vtk_files(reference, result, rtol, atol)
            else:
                raise NotImplementedError(
                    f"Comparison is not yet implemented for {reference.suffix} files."
                )
        else:
            raise TypeError("Reference and result must be either string or Path.")

    return _compare


def compare_strings(
    reference: str, result: str, rtol: float, atol: float, string_splitter=" "
) -> bool:
    """Compare if two strings are identical within a given tolerance. If no
    tolerance is given strings are compared for equality.

    Args:
        reference: The reference string.
        result: The result string.
        rtol: The relative tolerance.
        atol: The absolute tolerance.
        string_splitter: With which string the strings are split.

    Returns:
        bool: true if comparison is successful, false otherwise
    """

    # compare strings for equality
    if rtol is None and atol is None:
        diff = list(
            unified_diff(reference.splitlines(), result.splitlines(), lineterm="")
        )
        if diff:
            print("Difference between reference and result:")
            print("\n".join(list(diff)))
            return False
        else:
            return True
    # compare strings with tolerance
    else:
        rtol = 0.0 if rtol is None else rtol
        atol = 0.0 if atol is None else atol

        lines_reference = reference.strip().split("\n")
        lines_result = result.strip().split("\n")

        if len(lines_reference) != len(lines_result):
            raise AssertionError(
                f"Number of lines in reference and result differ: {len(lines_reference)} != {len(lines_result)}"
            )

        # Loop over each line in the file
        for line_reference, line_result in zip(lines_reference, lines_result):
            line_reference_splits = line_reference.strip().split(string_splitter)
            line_result_splits = line_result.strip().split(string_splitter)

            if len(line_reference_splits) != len(line_result_splits):
                raise AssertionError(
                    f"Number of items in reference and result line differ!\n"
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
                    if np.isclose(
                        number_reference, number_result, rtol=rtol, atol=atol
                    ):
                        pass
                    else:
                        raise AssertionError(
                            f"Numbers do not match within given tolerance!\n"
                            + f"Reference line: {line_reference}\n"
                            + f"Result line:    {line_result}"
                        )
                except ValueError:
                    if item_reference.strip() != item_result.strip():
                        raise AssertionError(
                            f"Strings do not match in line!\n"
                            + f"Reference line: {line_reference}\n"
                            + f"Result line:    {line_result}"
                        )

        return True


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


def compare_vtk_files(reference: Path, result: Path, rtol: float, atol: float) -> bool:
    """Compare two VTK files for equality within a given tolerance.

    Args:
        reference: The path to the reference VTK file.
        result: The path to the result VTK file to be compared.
        rtol: The relative tolerance parameter.
        atol: The absolute tolerance parameter.

    Returns:
        bool: true if comparison is successful, false otherwise
    """

    compare = compare_grids(
        get_vtk(reference), get_vtk(result), output=True, rtol=rtol, atol=atol
    )

    if not compare[0]:
        raise AssertionError("\n".join(compare[1]))

    return compare[0]
