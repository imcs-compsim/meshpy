# The MIT License (MIT)
#
# Copyright (c) 2018-2025 BeamMe Authors
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
"""Test that the input files created with CubitPy are up to date."""

import pytest

from tests.create_cubit_input import (
    create_block,
    create_solid_shell_meshes,
    create_tube,
    create_tube_tutorial,
)

# Tolerances to be used for the assert_results_equal functions in this file
ASSERT_RESULTS_EQUAL_TOL = {"atol": 1e-13, "rtol": 1e-13}


@pytest.mark.cubitpy
def test_create_cubit_input_tube(
    tmp_path,
    get_corresponding_reference_file_path,
    assert_results_equal,
):
    """Test that the solid tube reference file is up to date."""

    result_path = tmp_path / get_corresponding_reference_file_path().name
    create_tube(result_path)
    assert_results_equal(
        result_path, get_corresponding_reference_file_path(), **ASSERT_RESULTS_EQUAL_TOL
    )


@pytest.mark.cubitpy
def test_create_cubit_input_tutorial(
    tmp_path,
    reference_file_directory,
    assert_results_equal,
):
    """Test the creation of the solid input file for the tutorial."""

    # Create the tube for the tutorial.
    result_path = tmp_path / "tutorial.4C.yaml"
    create_tube_tutorial(result_path)

    tutorial_path = (
        reference_file_directory.parents[1]
        / "tutorial"
        / "4C_input_solid_tutorial.4C.yaml"
    )
    assert_results_equal(tutorial_path, result_path, **ASSERT_RESULTS_EQUAL_TOL)


@pytest.mark.cubitpy
def test_create_cubit_input_block(
    tmp_path,
    get_corresponding_reference_file_path,
    assert_results_equal,
):
    """Test that the solid block reference file is up to date."""

    result_path = tmp_path / get_corresponding_reference_file_path().name
    create_block(result_path)
    assert_results_equal(
        result_path, get_corresponding_reference_file_path(), **ASSERT_RESULTS_EQUAL_TOL
    )


@pytest.mark.cubitpy
def test_create_cubit_input_solid_shell(
    tmp_path,
    get_corresponding_reference_file_path,
    assert_results_equal,
):
    """Test that the solid shell reference files are up to date."""

    reference_path_blocks = get_corresponding_reference_file_path(
        additional_identifier="blocks"
    )
    reference_path_dome = get_corresponding_reference_file_path(
        additional_identifier="dome"
    )
    result_path_blocks = tmp_path / reference_path_blocks.name
    result_path_dome = tmp_path / reference_path_dome.name

    create_solid_shell_meshes(result_path_blocks, result_path_dome)
    assert_results_equal(result_path_blocks, reference_path_blocks)
    assert_results_equal(
        result_path_dome, reference_path_dome, **ASSERT_RESULTS_EQUAL_TOL
    )
