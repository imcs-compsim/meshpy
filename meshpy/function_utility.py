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
This module implements utility functions to create 4C space time function
"""

# Python modules
import numpy as np

# Meshpy modules
from .function import Function


def create_linear_interpolation_string(
    t, values, *, variable_name="var", variable_index=0
):
    """Create a string that describes a variable that is linear interpolated over time

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
    """Create a function that describes a linear interpolation between the given time points and
    values. Before and after it will be constant.

    Args
    ----
    t, values: list(float)
        Time and values that will be interpolated with piecewise linear functions
    """

    variable_string = create_linear_interpolation_string(t, values, variable_name="var")
    return Function(f"{function_type} var\n" + variable_string)


def linear_time_transformation(
    time, values, time_span, flip, valid_start_and_end_point=False
):
    """Performs a transformation of the time to a new intervall with an appropriate value vector

    Args
    ----
    time: np.array
        array with time values
    values: np.array
        corresponding values to time
    time_span: [list] with 2 or 3 entries:
        time_span[0:2] defines the time intervall to which the initial time intervall should be scaled.
        time_span[3] repeats the final entry
    flip: Bool
        Flag if the values should be reversed
    valid_start_and_end_point: Bool
        optionally adds a valid starting point at t=0 and timespan[3] if provided
    """

    # flip values if desired and adjust time
    if flip is True:
        values = np.flipud(values)
        time = np.flip(-time) + time[-1]

    # transform time to intervall
    min_t = np.min(time)
    max_t = np.max(time)

    # scaling/transforming the time into the user defined time
    time = time_span[0] + (time - min_t) * (time_span[1] - time_span[0]) / (
        max_t - min_t
    )

    # ensure that start time is valid
    if valid_start_and_end_point and time[0] > 0.0:

        # add starting time 0
        time = np.append(0.0, time)

        # add first coordinate again at the beginning of the array
        if len(values.shape) == 1:
            values = np.append(values[0], values)
        else:
            values = np.append(values[0], values).reshape(
                values.shape[0] + 1, values.shape[1]
            )

    # repeat last value at provided time point
    if valid_start_and_end_point and len(time_span) > 2:
        if time_span[2] > time_span[1]:
            time = np.append(time, time_span[2])
            values = np.append(values, values[-1]).reshape(
                values.shape[0] + 1, values.shape[1]
            )

    return time, values
