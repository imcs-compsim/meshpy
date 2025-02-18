# -*- coding: utf-8 -*-
# MeshPy: A beam finite element input generator
#
# MIT License
#
# Copyright (c) 2018-2025
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
"""This script is used to test the header functions."""

import pytest

from meshpy.core.conf import mpy
from meshpy.core.header_functions import (
    get_comment,
    set_beam_to_solid_meshtying,
    set_header_static,
    set_runtime_output,
)
from meshpy.four_c.inputfile import InputFile


def test_header_functions_static(
    get_corresponding_reference_file_path,
    assert_results_equal,
):
    """Test the default static header function."""

    # Create input file.
    input_file = InputFile()

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

    input_file.add("--Test\n{}on\n{}off".format(get_comment(True), get_comment(False)))

    # Check the output.
    assert_results_equal(get_corresponding_reference_file_path(), input_file)


@pytest.mark.parametrize(
    "additional_identifier, time_step, n_steps, total_time",
    [
        ("all", 0.01, 17, 2.1),
        ("no_total_time", 0.01, 17, None),
        ("no_n_steps", 0.01, None, 2.1),
        ("no_time_step", None, 17, 2.1),
    ],
)
def test_header_functions_static_time(
    get_corresponding_reference_file_path,
    assert_results_equal,
    additional_identifier,
    time_step,
    n_steps,
    total_time,
):
    """Test the time setting options in the static header functions."""

    input_file = InputFile()

    set_header_static(
        input_file, time_step=time_step, n_steps=n_steps, total_time=total_time
    )

    assert_results_equal(
        get_corresponding_reference_file_path(
            additional_identifier=additional_identifier
        ),
        input_file,
    )


def test_header_functions_static_prestress(
    get_corresponding_reference_file_path,
    assert_results_equal,
):
    """Test the static header function with non default prestressing
    parameter."""

    # Create input file.
    input_file = InputFile()

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
    assert_results_equal(get_corresponding_reference_file_path(), input_file)


def test_header_functions_stress_output(
    get_corresponding_reference_file_path,
    assert_results_equal,
):
    """Test the static header function with non default stress output
    parameter."""

    # Create input file.
    input_file = InputFile()

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
    assert_results_equal(get_corresponding_reference_file_path(), input_file)
