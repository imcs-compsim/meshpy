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
This script is used to test the header functions.
"""

# Python imports.
import unittest
import os

# Meshpy imports.
from meshpy import mpy, InputFile

# Header functions.
from meshpy.header_functions import (
    set_header_static,
    set_runtime_output,
    set_beam_to_solid_meshtying,
    get_comment,
)

# Testing imports.
from testing_utility import testing_input, compare_strings


class TestHeaderFunctions(unittest.TestCase):
    """
    Test the header functions.
    """

    def test_static(self):
        """
        Test the default static header function.
        """

        # Set default values for global parameters.
        mpy.set_default_values()

        # Create input file.
        input_file = InputFile(maintainer="Ivo Steinbrecher")

        # Set the header.
        set_header_static(input_file, time_step=0.1, n_steps=17, load_lin=True)
        set_runtime_output(input_file, output_triad=False)
        set_beam_to_solid_meshtying(
            input_file,
            mpy.beam_to_solid.volume_meshtying,
            contact_discretization="mortar",
            binning_bounding_box=[1, 2, 3, 4, 5, 6],
            binning_cutoff_radius=0.69,
        )

        set_beam_to_solid_meshtying(
            input_file,
            mpy.beam_to_solid.surface_meshtying,
            contact_discretization="gp",
            binning_bounding_box=[1, 2, 3, 4, 5, 6],
            binning_cutoff_radius=0.69,
            segmentation_search_points=6,
            coupling_type="consistent_fad",
        )

        set_beam_to_solid_meshtying(
            input_file,
            mpy.beam_to_solid.surface_meshtying,
            contact_discretization="gp",
            segmentation=False,
            option_overwrite=True,
        )

        input_file.add(
            "--Test\n{}on\n{}off".format(get_comment(True), get_comment(False))
        )

        # Check the output.
        ref_file = os.path.join(testing_input, "test_header_static_reference.dat")
        compare_strings(
            self,
            "test_header_static",
            ref_file,
            input_file.get_string(header=False, check_nox=False),
        )

    def test_static_prestress(self):
        """
        Test the static header function with non default prestressing
        parameter.
        """

        # Set default values for global parameters.
        mpy.set_default_values()

        # Create input file.
        input_file = InputFile(maintainer="Ivo Steinbrecher")

        # Set the header.
        set_header_static(
            input_file,
            time_step=0.1,
            n_steps=17,
            load_lin=True,
            prestress="mulf",
            prestress_time=1,
        )
        set_beam_to_solid_meshtying(
            input_file,
            mpy.beam_to_solid.volume_meshtying,
            contact_discretization="mortar",
            binning_bounding_box=[1, 2, 3, 4, 5, 6],
            binning_cutoff_radius=0.69,
            couple_restart=True,
        )

        set_beam_to_solid_meshtying(
            input_file,
            mpy.beam_to_solid.surface_meshtying,
            contact_discretization="gp",
            segmentation=False,
            couple_restart=False,
        )

        # Check the output.
        ref_file = os.path.join(
            testing_input, "test_header_static_prestress_reference.dat"
        )
        compare_strings(
            self,
            "test_header_static_prestress",
            ref_file,
            input_file.get_string(header=False, check_nox=False),
        )

    def test_stress_output(self):
        """
        Test the static header function with non default stress output
        parameter.
        """

        # Set default values for global parameters.
        mpy.set_default_values()

        # Create input file.
        input_file = InputFile(maintainer="Ivo Steinbrecher")

        # Set the header.
        set_header_static(
            input_file,
            time_step=0.1,
            n_steps=17,
            load_lin=True,
            write_stress="cauchy",
            write_strain="gl",
        )
        set_runtime_output(input_file, output_stress_strain=True)

        # Check the output.
        ref_file = os.path.join(
            testing_input, "test_header_stress_output_reference.dat"
        )
        compare_strings(
            self,
            "test_header_stress_output",
            ref_file,
            input_file.get_string(header=False, check_nox=False),
        )


if __name__ == "__main__":
    # Execution part of script.
    unittest.main()
