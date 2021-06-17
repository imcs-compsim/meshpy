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
This script is used to call all unit test modules in python.
"""

# Python imports.
import os
import sys
import unittest

# Import testing functions.
from tests.testing_utility import empty_testing_directory

# Set path to find meshpy.
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))


if __name__ == '__main__':
    # Execution part of script.

    empty_testing_directory()

    # Load the test cases.
    testsuite = unittest.TestLoader().discover('.', pattern='testing_*.py')

    # Perform the tests
    run = unittest.TextTestRunner(verbosity=1).run(testsuite)
    sys.exit(not (run.wasSuccessful()))
