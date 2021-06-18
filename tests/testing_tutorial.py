# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# MeshPy: A beam finite element input generator.
#
# Copyright (c) 2021 Ivo Steinbrecher
#                    Institute for Mathematics and Computer-Based Simulation
#                    Universitaet der Bundeswehr Muenchen
#                    https://www.unibw.de/imcs-en
#
# TODO: Add license.
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
