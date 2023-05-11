# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# MeshPy: A beam finite element input generator
#
# MIT License
#
# Copyright (c) 2021 Ivo Steinbrecher
#                    Institute for Mathematics and Computer-Based Simulation
#                    Universitaet der Bundeswehr Muenchen
#                    https://www.unibw.de/imcs-en
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
This script is used to test the functionality of the meshpy.function_utility module.
"""

# Python imports
import unittest

# Meshpy imports
from meshpy.function_utility import (
    create_linear_interpolation_function,
    create_linear_interpolation_string,
)


class TestFunctionUtilitiesGeometricSearch(unittest.TestCase):
    """Test various stuff from the meshpy.function_utility module."""

    def test_linear_interpolation_function(self):
        """Test that linear interpolation functions are created correctly"""

        t = [1.5, 2.5, 3.5, 10.0]
        values = [1.0, -1.0, 3.5, -10.3]

        fun = create_linear_interpolation_function(t, values)
        self.assertEqual(
            [
                "SYMBOLIC_FUNCTION_OF_SPACE_TIME var\nVARIABLE 0 NAME var TYPE linearinterpolation NUMPOINTS 6 TIMES -1000.0 1.5 2.5 3.5 10.0 1010.0 VALUES 1.0 1.0 -1.0 3.5 -10.3 -10.3"
            ],
            fun.get_dat_lines(),
        )

        fun_string = create_linear_interpolation_string(
            t, values, variable_name="test", variable_index=12
        )
        self.assertEqual(
            "VARIABLE 12 NAME test TYPE linearinterpolation NUMPOINTS 6 TIMES -1000.0 1.5 2.5 3.5 10.0 1010.0 VALUES 1.0 1.0 -1.0 3.5 -10.3 -10.3",
            fun_string,
        )


if __name__ == "__main__":
    # Execution part of script.
    unittest.main()
