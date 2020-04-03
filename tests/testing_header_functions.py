# -*- coding: utf-8 -*-
"""
This script is used to test the header functions.
"""

# Python imports.
import unittest
import os

# Meshpy imports.
from meshpy import (mpy, InputFile)

# Header functions.
from meshpy.header_functions import (set_header_static, set_runtime_output,
    set_beam_to_solid_meshtying)

# Testing imports.
from tests.testing_utility import testing_input, compare_strings


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
        input_file = InputFile(maintainer='Ivo Steinbrecher')

        # Set the header.
        set_header_static(input_file, time_step=0.1, n_steps=17, load_lin=True)
        set_runtime_output(input_file, output_triad=False)
        set_beam_to_solid_meshtying(input_file,
            mpy.beam_to_solid.volume_meshtying,
            contact_discretization='mortar',
            binning_bounding_box=[1, 2, 3, 4, 5, 6],
            binning_cutoff_radius=0.69)

        set_beam_to_solid_meshtying(input_file,
            mpy.beam_to_solid.surface_meshtying,
            contact_discretization='gp',
            binning_bounding_box=[1, 2, 3, 4, 5, 6],
            binning_cutoff_radius=0.69,
            segmentation_search_points=6)

        set_beam_to_solid_meshtying(input_file,
            mpy.beam_to_solid.surface_meshtying,
            contact_discretization='gp',
            binning_bounding_box=[1, 2, 3, 4, 5, 6],
            binning_cutoff_radius=0.69,
            option_overwrite=True)

        # Check the output.
        ref_file = os.path.join(testing_input,
            'test_header_static_reference.dat')
        compare_strings(
            self,
            'test_header_static',
            ref_file,
            input_file.get_string(header=False, check_nox=False))


if __name__ == '__main__':
    # Execution part of script.
    unittest.main()
