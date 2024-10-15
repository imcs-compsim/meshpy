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
This file has functions to creates a flow diverter
"""
from meshpy.mesh_creation_functions import create_beam_mesh_helix
import numpy as np


def create_beam_flow_diverter(
    mesh,
    beam_object,
    material,
    length,
    radius,
    *,
    n_turns=1,
    n_wire=1,
    n_el=10,
    interwoven=False,
):

    """create a mesh consisting of multiple helix to represent a fd"""
    fibers = []
    # Create a wire clockwise and counterclockwise
    for clockwise in [1.0, -1.0]:
        for i in range(1, n_wire + 1):

            # create beams according the flow diverter function
            beam_set = create_beam_mesh_helix(
                mesh,
                beam_object,
                material,
                axis_vector=[0.0, 0.0, clockwise],
                axis_point=[0.0, 0.0, 0.0],
                # calculate starting point based on cylinder
                start_point=[
                    radius * np.cos(i * 2 * np.pi / n_wire),
                    radius * np.sin(i * 2 * np.pi / n_wire),
                    0.0,
                ],
                helix_angle=-clockwise
                * np.arctan(length / (2 * radius * np.pi * n_turns)),
                height_helix=length,
                n_el=n_el,
            )

            # add all single fibers as geometry set to mesh
            fibers.append(beam_set["line"].get_all_nodes())

    if interwoven == "z-cylinder":
        mesh.interwove_nodes_of_z_cylinder(fibers=fibers, beam_radius=material.radius)
