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
Utility functions to create all different kind of meshes used within the tests.
"""
from meshpy import (
    mpy,
    set_header_static,
    InputFile,
    Function,
    MaterialReissner,
    Beam3rHerm2Line3,
)
from meshpy.mesh_creation_functions import create_beam_mesh_line


def create_cantilver_model(n_steps, time_step=0.5):
    """
    Create a simple cantilver model.

    Args
    ----
    n_steps: int
        Number of simulation steps.
    time_step: float
        Time step size.
    """

    mpy.set_default_values()
    input_file = InputFile()
    set_header_static(input_file, time_step=time_step, n_steps=n_steps)
    input_file.add("--IO\nOUTPUT_BIN yes\nSTRUCT_DISP yes", option_overwrite=True)
    ft = Function("COMPONENT 0 SYMBOLIC_FUNCTION_OF_SPACE_TIME t")
    input_file.add(ft)
    mat = MaterialReissner(youngs_modulus=100.0, radius=0.1)
    beam_set = create_beam_mesh_line(
        input_file, Beam3rHerm2Line3, mat, [0, 0, 0], [2, 0, 0], n_el=10
    )

    return input_file, beam_set
