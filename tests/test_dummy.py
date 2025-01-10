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
"""Dummy test to demonstrate and ensure working pytest infrastructure."""

from pathlib import Path

import pytest


def test_dummy(
    reference_file_directory: Path,
    tmp_path: Path,
    current_test_name: str,
    get_corresponding_reference_file_path,
) -> None:
    """Dummy test to demonstrate pytest fixtures.

    Args:
        reference_file_directory: path to the reference file directory
        tmp_path: temporary path for testing
        current_test_name: name of the current test
        get_corresponding_reference_file_path: path to the corresponding reference file
    """

    # approach to get reference_file_directory
    print("reference_file_directory: ", reference_file_directory)

    # approach to get temporary testing path
    print(
        "tmp_path: ", tmp_path
    )  # pytest automatically keeps the last three runs for debugging and deletes everything that's older

    # approach to get the current test name
    print("current_test_name: ", current_test_name)

    # approach to get the path to the corresponding reference .dat file
    print(
        "corresponding reference file path: ", get_corresponding_reference_file_path()
    )

    # approach to get the path to a reference file with other base name, additional identifier and extension
    print(
        "corresponding reference file path: ",
        get_corresponding_reference_file_path(
            reference_file_base_name="test_dummy_2",
            additional_identifier="id",
            extension="txt",
        ),
    )

    assert True


@pytest.mark.fourc
def test_4C() -> None:
    """Test with 4C."""

    assert True


@pytest.mark.arborx
def test_ArborX() -> None:
    """Test with ArborX."""

    assert True


@pytest.mark.cubitpy
def test_CubitPy() -> None:
    """Test with CubitPy."""

    assert True


@pytest.mark.performance
def test_performance() -> None:
    """Performance test."""

    assert True
