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
"""This module implements utility functions to create 4C space time
function."""

from typing import List

import numpy as np

from meshpy.four_c.function import Function


def create_linear_interpolation_string(
    t, values, *, variable_name="var", variable_index=0
):
    """Create a string that describes a variable that is linear interpolated
    over time.

    Args
    ----
    t, values: list(float)
        Time and values that will be interpolated with piecewise linear functions
    variable_name: str
        Name of the created variable
    variable_index: int
        Index of this created variable
    """

    if not len(t) == len(values):
        raise ValueError(
            f"The dimensions of time ({len(t)}) and values ({len(values)}) do not match"
        )

    t = t.copy()
    f = values.copy()
    t_max = np.max(t)
    t = np.insert(t, 0, -1000.0, axis=0)
    t = np.append(t, [t_max + 1000.0])
    f = np.insert(f, 0, f[0], axis=0)
    f = np.append(f, f[-1])
    times = " ".join(map(str, t))
    values = " ".join(map(str, f))
    return (
        f"VARIABLE {variable_index} NAME {variable_name} TYPE linearinterpolation "
        + f"NUMPOINTS {len(t)} TIMES {times} VALUES {values}"
    )


def create_linear_interpolation_function(
    t, values, *, function_type="SYMBOLIC_FUNCTION_OF_SPACE_TIME"
):
    """Create a function that describes a linear interpolation between the
    given time points and values. Before and after it will be constant.

    Args
    ----
    t, values: list(float)
        Time and values that will be interpolated with piecewise linear functions
    """

    variable_string = create_linear_interpolation_string(t, values, variable_name="var")
    return Function(f"{function_type} var\n" + variable_string)


def ensure_length_of_function_array(
    function_array: List[Function], length: int = 3
) -> List[Function]:
    """Performs size check of a function array and appends the function array to the given length, if a list with only one item is provided.
    Args:
        function_array: list with functions
        length: expected length of function array

    Returns:
        function_array: list with functions with provided length
    """

    # extend items of function automatically if it is only provided once
    if len(function_array) == 1:
        return function_array * length

    if len(function_array) != length:
        raise ValueError(
            f"The function array must have length {length} not {len(function_array)}."
        )
    return function_array
