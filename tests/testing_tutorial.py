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
This script is used to test the tutorial.
"""

# Import python modules.
import unittest
import os

# Import tutorial function.
from tutorial import meshpy_tutorial

# Import testing utilities.
from tests.testing_utility import (skip_fail_test, testing_temp, testing_input,
    compare_strings, compare_vtk)


class Testtutorial(unittest.TestCase):
    """This class tests the headers in the repository."""

    def test_tutorial(self):
        """
        Test that the tutorial works.
        """

        input_file = meshpy_tutorial(testing_temp)
        input_file.write_input_file(os.path.join(testing_temp, 'tutorial.dat'),
            header=False, dat_header=False)

        tutorial_file = os.path.join(testing_temp,
            'tutorial.dat')
        ref_file = os.path.join(testing_input,
            'test_tutorial_reference.dat')
        compare_strings(self,
            'test_tutorial',
            ref_file,
            tutorial_file)


if __name__ == '__main__':
    # Execution part of script.
    unittest.main()
