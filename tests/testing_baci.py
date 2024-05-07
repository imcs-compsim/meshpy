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
This script is used to test the functionality of MeshPy for creating BACI input files.
TODO: A lot of this functionality is covered in testing_meshpy.py this should be moved
to this file at some point.
"""

# Python imports.
import unittest

# Testing imports.
from utilities import compare_test_result

# Meshpy imports.
from meshpy import (
    mpy,
    InputFile,
    MaterialReissner,
)


class TestBaci(unittest.TestCase):
    """
    Test BACI related functionality in MeshPy
    """

    def setUp(self):
        """
        This method is called before each test and sets the default MeshPy
        values for each test. The values can be changed in the individual
        tests.
        """

        # Set default values for global parameters.
        mpy.set_default_values()

    def test_baci_material_numbering(self):
        """Test that materials can be added as strings to an input file (as is done when
        importing dat files) and that the numbering with other added materials does not
        lead to materials with double IDs."""

        input_file = InputFile()
        input_file.add(
            """
            --MATERIALS
            // some comment
            MAT 1 MAT_ViscoElastHyper NUMMAT 4 MATIDS 10 11 12 13 DENS 1.3e-6   // density (kg/mm^3), young (N/mm^2)
            MAT 10 ELAST_CoupNeoHooke YOUNG 0.16 NUE 0.45  // 0.16 (MPa)
            MAT 11 VISCO_GenMax TAU 0.1 BETA 0.4 SOLVE OST
            MAT 12 ELAST_CoupAnisoExpo K1 2.4e-03 K2 0.14 GAMMA 0.0 K1COMP 0 K2COMP 1 ADAPT_ANGLE No INIT 3 STR_TENS_ID 100 FIBER_ID 1
            MAT 13 ELAST_CoupAnisoExpo K1 5.4e-03 K2 1.24 GAMMA 0.0 K1COMP 0 K2COMP 1 ADAPT_ANGLE No INIT 3 STR_TENS_ID 100 FIBER_ID 2
            MAT 100 ELAST_StructuralTensor STRATEGY Standard

            // other comment
            MAT 2 MAT_ElastHyper NUMMAT 3 MATIDS 20 21 22 DENS 1.3e-6                                            // density (kg/mm^3), young (N/mm^2)
            MAT 20 ELAST_CoupNeoHooke YOUNG 1.23 NUE 0.45                                                 // MPa
            MAT 21 ELAST_CoupAnisoExpo K1 0.4e-03 K2 12.0 GAMMA 0.0 K1COMP 0 K2COMP 1 ADAPT_ANGLE No INIT 3 STR_TENS_ID 200 FIBER_ID 1
            MAT 22 ELAST_CoupAnisoExpo K1 50.2e-03 K2 10.0 GAMMA 0.0 K1COMP 0 K2COMP 1 ADAPT_ANGLE No INIT 3 STR_TENS_ID 200 FIBER_ID 2
            MAT 200 ELAST_StructuralTensor STRATEGY Standard
            """
        )
        input_file.add(MaterialReissner(youngs_modulus=1.0, radius=2.0))
        compare_test_result(self, input_file.get_string(header=False, dat_header=False))


if __name__ == "__main__":
    # Execution part of script.
    unittest.main()
