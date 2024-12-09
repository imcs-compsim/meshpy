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
"""This file contains functionality to warp an existing mesh along a 1D curve"""


# Import user modules
from .cosserat_curve import CosseratCurve
from meshpy import Mesh
from meshpy.node import Node, NodeCosserat
from meshpy.rotation import Rotation


def warp_mesh_along_curve(
    mesh: Mesh,
    curve: CosseratCurve,
    *,
    origin=[0.0, 0.0, 0.0],
    reference_rotation=Rotation(),
):
    """Warp an existing mesh along the given curve. The reference coordinates for the
    transformation are defined by the given origin and rotation, where the first basis
    vector of the triad defines the centerline axis."""

    # Loop over nodes and map them to the new configuration
    for node in mesh.nodes:
        if not isinstance(node, Node):
            raise TypeError(
                "All nodes in the mesh have to be derived from the base Node object"
            )

        # Get the coordinates in the cylindrical coordinate system, so we can transform
        # the node along the centerline curve.
        node_pos = node.coordinates
        coordinates = reference_rotation.inv() * (node_pos - origin)
        centerline_position = coordinates[0]
        cross_section_coordinates = [0.0, *coordinates[1:]]

        pos, rot = curve.get_centerline_position_and_rotation(centerline_position)
        rot = Rotation(rot)

        new_pos = pos + rot * cross_section_coordinates
        node.coordinates = new_pos
        if isinstance(node, NodeCosserat):
            node.rotation = rot * reference_rotation.inv() * node.rotation
