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
"""This script is used to test the functionality of the meshpy.function_utility
module."""

from meshpy.function_utility import (
    create_linear_interpolation_function,
    create_linear_interpolation_string,
)


def test_linear_interpolation_function():
    """Test that linear interpolation functions are created correctly."""

    t = [1.5, 2.5, 3.5, 10.0]
    values = [1.0, -1.0, 3.5, -10.3]

    fun = create_linear_interpolation_function(t, values)
    assert [
        "SYMBOLIC_FUNCTION_OF_SPACE_TIME var\nVARIABLE 0 NAME var TYPE linearinterpolation NUMPOINTS 6 TIMES -1000.0 1.5 2.5 3.5 10.0 1010.0 VALUES 1.0 1.0 -1.0 3.5 -10.3 -10.3"
    ] == fun.get_dat_lines()

    fun = create_linear_interpolation_function(t, values, function_type="My type")
    assert [
        "My type var\nVARIABLE 0 NAME var TYPE linearinterpolation NUMPOINTS 6 TIMES -1000.0 1.5 2.5 3.5 10.0 1010.0 VALUES 1.0 1.0 -1.0 3.5 -10.3 -10.3"
    ] == fun.get_dat_lines()

    fun_string = create_linear_interpolation_string(
        t, values, variable_name="test", variable_index=12
    )
    assert (
        "VARIABLE 12 NAME test TYPE linearinterpolation NUMPOINTS 6 TIMES -1000.0 1.5 2.5 3.5 10.0 1010.0 VALUES 1.0 1.0 -1.0 3.5 -10.3 -10.3"
        == fun_string
    )
