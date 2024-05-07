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
This script is used to test the functionality of MeshPy for creating 4C input files.
"""

# Python imports.
import unittest

import numpy as np

# Testing imports.
from utilities import compare_test_result

# Meshpy imports.
from meshpy import (
    Beam3rHerm2Line3,
    BoundaryCondition,
    Function,
    GeometrySet,
    InputFile,
    MaterialReissner,
    mpy,
    set_header_static,
)
from meshpy.four_c.beam_potential import BeamPotential
from meshpy.mesh_creation_functions.beam_basic_geometry import create_beam_mesh_helix
from meshpy.utility import is_node_on_plane


class Test4C(unittest.TestCase):
    """
    Test 4C related functionality in MeshPy
    """

    def test_four_c_simulation_beam_potential_helix(self):
        """Test the correct creation of input files for simulations including beam to
        beam potential interactions."""

        def create_model():
            """Create the beam to beam potential interaction model."""

            input_file = InputFile()
            mat = MaterialReissner(
                youngs_modulus=1000, radius=0.5, shear_correction=1.0
            )

            # define function for line charge density
            fun = Function("COMPONENT 0 SYMBOLIC_FUNCTION_OF_SPACE_TIME t")

            # define the beam potential
            beampotential = BeamPotential(
                input_file,
                pot_law_prefactor=[-1.0e-3, 12.45e-8],
                pot_law_exponent=[6.0, 12.0],
                pot_law_line_charge_density=[1.0, 2.0],
                pot_law_line_charge_density_funcs=[fun, "none"],
            )

            # set headers for static case and beam potential
            beampotential.add_header(
                cutoff_radius=10.0,
                evaluation_strategy="SingleLengthSpecific_SmallSepApprox_Simple",
                potential_reduction_length=15.0,
            )
            beampotential.add_runtime_output(every_iteration=True)

            # create helix
            helix_set = create_beam_mesh_helix(
                input_file,
                Beam3rHerm2Line3,
                mat,
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                twist_angle=np.pi / 4,
                height_helix=10,
                n_el=4,
            )

            # add potential charge conditions to helix
            beampotential.add_potential_charge_condition(geometry_set=helix_set["line"])

            ### Add boundary condition to bottom node
            input_file.add(
                BoundaryCondition(
                    GeometrySet(
                        input_file.get_nodes_by_function(
                            is_node_on_plane,
                            normal=[0, 0, 1],
                            origin_distance=0.0,
                            tol=0.1,
                        )
                    ),
                    "NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0",
                    bc_type=mpy.bc.dirichlet,
                )
            )

            return input_file

        input_file = create_model()

        compare_test_result(self, input_file.get_string(header=False))


if __name__ == "__main__":
    # Execution part of script.
    unittest.main()
