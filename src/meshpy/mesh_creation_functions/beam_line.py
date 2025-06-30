# The MIT License (MIT)
#
# Copyright (c) 2018-2025 BeamMe Authors
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Functions to create beam meshes along straight lines."""

import numpy as _np

from meshpy.core.conf import mpy as _mpy
from meshpy.core.rotation import Rotation as _Rotation
from meshpy.mesh_creation_functions.beam_generic import (
    create_beam_mesh_generic as _create_beam_mesh_generic,
)


def create_beam_mesh_line(mesh, beam_class, material, start_point, end_point, **kwargs):
    """Generate a straight line of beam elements.

    Args
    ----
    mesh: Mesh
        Mesh that the line will be added to.
    beam_class: Beam
        Class of beam that will be used for this line.
    material: Material
        Material for this line.
    start_point, end_point: _np.array, list
        3D-coordinates for the start and end point of the line.

    **kwargs (for all of them look into create_beam_mesh_function)
    ----
    n_el: int
        Number of equally spaced beam elements along the line. Defaults to 1.
        Mutually exclusive with l_el.
    l_el: float
        Desired length of beam elements. Mutually exclusive with n_el.
        Be aware, that this length might not be achieved, if the elements are
        warped after they are created.
    start_node: Node, GeometrySet
        Node to use as the first node for this line. Use this if the line
        is connected to other lines (angles have to be the same, otherwise
        connections should be used). If a geometry set is given, it can
        contain one, and one node only.
    add_sets: bool
        If this is true the sets are added to the mesh and then displayed
        in eventual VTK output, even if they are not used for a boundary
        condition or coupling.

    Return
    ----
    return_set: GeometryName
        Set with the 'start' and 'end' node of the line. Also a 'line' set
        with all nodes of the line.
    """

    # Get geometrical values for this line.
    start_point = _np.asarray(start_point)
    end_point = _np.asarray(end_point)
    direction = end_point - start_point
    line_length = _np.linalg.norm(direction)
    t1 = direction / line_length

    # Check if the z or y axis are larger projected onto the direction.
    # The tolerance is used here to ensure that round-off changes in the last digits of
    # the floating point values don't switch the case. This increases the robustness in
    # testing.
    if abs(_np.dot(t1, [0, 0, 1])) < abs(_np.dot(t1, [0, 1, 0])) - _mpy.eps_quaternion:
        t2 = [0, 0, 1]
    else:
        t2 = [0, 1, 0]
    rotation = _Rotation.from_basis(t1, t2)

    def get_beam_geometry(parameter_a, parameter_b):
        """Return a function for the position along the beams axis."""

        def beam_function(xi):
            """Return a point on the beams axis for a given parameter
            coordinate xi."""
            point_a = start_point + parameter_a * direction
            point_b = start_point + parameter_b * direction
            pos = 0.5 * (1 - xi) * point_a + 0.5 * (1 + xi) * point_b
            arc_length = (
                0.5 * (1 - xi) * parameter_a + 0.5 * (1 + xi) * parameter_b
            ) * line_length
            return (pos, rotation, arc_length)

        return beam_function

    # Create the beam in the mesh
    return _create_beam_mesh_generic(
        mesh,
        beam_class=beam_class,
        material=material,
        function_generator=get_beam_geometry,
        interval=[0.0, 1.0],
        interval_length=line_length,
        **kwargs,
    )
