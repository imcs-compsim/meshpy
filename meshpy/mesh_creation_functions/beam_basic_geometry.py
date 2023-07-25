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
This file has functions to create basic geometry items with meshpy.
"""

# Python packages.
import numpy as np

# Meshpy modules.
from ..conf import mpy
from ..rotation import Rotation
from ..utility import get_node
from .beam_generic import create_beam_mesh_function


def create_beam_mesh_line(
    mesh, beam_object, material, start_point, end_point, **kwargs
):
    """
    Generate a straight line of beam elements.

    Args
    ----
    mesh: Mesh
        Mesh that the line will be added to.
    beam_object: Beam
        Class of beam that will be used for this line.
    material: Material
        Material for this line.
    start_point, end_point: np.array, list
        3D-coordinates for the start and end point of the line.

    **kwargs (for all of them look into create_beam_mesh_function)
    ----
    n_el: int
        Number of equally spaced beam elements along the line. Defaults to 1.
        Mutually exclusive with l_el.
    l_el: float
        Desired length of beam elements. This requires the option interval_length
        to be set. Mutually exclusive with n_el. Be aware, that this length
        might not be achieved, if the elements are warped after they are
        created.
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
    direction = np.array(end_point) - np.array(start_point)
    line_length = np.linalg.norm(direction)
    t1 = direction / line_length

    # Check if the z or y axis are larger projected onto the direction.
    if abs(np.dot(t1, [0, 0, 1])) < abs(np.dot(t1, [0, 1, 0])):
        t2 = [0, 0, 1]
    else:
        t2 = [0, 1, 0]
    rotation = Rotation.from_basis(t1, t2)

    def get_beam_geometry(parameter_a, parameter_b):
        """
        Return a function for the position and rotation along the beam
        axis.
        """

        def beam_function(xi):
            point_a = start_point + parameter_a * direction
            point_b = start_point + parameter_b * direction
            return (0.5 * (1 - xi) * point_a + 0.5 * (1 + xi) * point_b, rotation)

        return beam_function

    # Create the beam in the mesh
    return create_beam_mesh_function(
        mesh,
        beam_object=beam_object,
        material=material,
        function_generator=get_beam_geometry,
        interval=[0.0, 1.0],
        interval_length=line_length,
        **kwargs,
    )


def create_beam_mesh_arc_segment(
    mesh, beam_object, material, center, axis_rotation, radius, angle, **kwargs
):
    """
    Generate a circular segment of beam elements.

    Args
    ----
    mesh: Mesh
        Mesh that the arc segment will be added to.
    beam_object: Beam
        Class of beam that will be used for this line.
    material: Material
        Material for this segment.
    center: np.array, list
        Center of the arc.
    axis_rotation: Rotation
        This rotation defines the spatial orientation of the arc.
        The 3rd base vector of this rotation is the rotation axis of the arc
        segment. The segment starts in the direction of the 1st basis vector
        and the starting point is along the 2nd basis vector.
    radius: float
        The radius of the segment.
    angle: float
        The central angle of this segment in radians.

    **kwargs (for all of them look into create_beam_mesh_function)
    ----
    n_el: int
        Number of equally spaced beam elements along the line. Defaults to 1.
        Mutually exclusive with l_el.
    l_el: float
        Desired length of beam elements. This requires the option interval_length
        to be set. Mutually exclusive with n_el. Be aware, that this length
        might not be achieved, if the elements are warped after they are
        created.

    Return
    ----
    return_set: GeometryName
        Set with the 'start' and 'end' node of the line. Also a 'line' set
        with all nodes of the line.
    """

    # The angle can not be negative with the current implementation.
    if angle <= 0.0:
        raise ValueError("The angle for a beam segment has to be a positive number!")

    def get_beam_geometry(alpha, beta):
        """
        Return a function for the position and rotation along the beam
        axis.
        """

        def beam_function(xi):
            phi = 0.5 * (xi + 1) * (beta - alpha) + alpha
            rot = axis_rotation * Rotation([0, 0, 1], phi)
            pos = center + radius * (rot * [0, -1, 0])
            return (pos, rot)

        return beam_function

    # Create the beam in the mesh
    return create_beam_mesh_function(
        mesh,
        beam_object=beam_object,
        material=material,
        function_generator=get_beam_geometry,
        interval=[0.0, angle],
        interval_length=angle * radius,
        **kwargs,
    )


def create_beam_mesh_arc_segment_2d(
    mesh, beam_object, material, center, radius, phi_start, phi_end, **kwargs
):
    """
    Generate a circular segment of beam elements in the x-y plane.

    Args
    ----
    mesh: Mesh
        Mesh that the arc segment will be added to.
    beam_object: Beam
        Class of beam that will be used for this line.
    material: Material
        Material for this segment.
    center: np.array, list
        Center of the arc. If the z component is not 0, an error will be
        thrown.
    radius: float
        The radius of the segment.
    phi_start, phi_end: float
        The start and end angles of the arc w.r.t the x-axis. If the start
        angle is larger than the end angle the beam faces in counter-clockwise
        direction, and if the start angle is smaller than the end angle, the
        beam faces in clockwise direction.

    **kwargs (for all of them look into create_beam_mesh_function)
    ----
    n_el: int
        Number of equally spaced beam elements along the line. Defaults to 1.
        Mutually exclusive with l_el.
    l_el: float
        Desired length of beam elements. This requires the option interval_length
        to be set. Mutually exclusive with n_el. Be aware, that this length
        might not be achieved, if the elements are warped after they are
        created.

    Return
    ----
    return_set: GeometryName
        Set with the 'start' and 'end' node of the line. Also a 'line' set
        with all nodes of the line.
    """

    # The center point has to be on the x-y plane.
    if np.abs(center[2]) > mpy.eps_pos:
        raise ValueError("The z-value of center has to be 0!")

    # Check if the beam is in clockwise or counter clockwise direction.
    angle = phi_end - phi_start
    clockwise = not np.sign(angle) == 1

    # Create rotation for the general arc segment function.
    axis_rotation = Rotation([0, 0, 1], np.pi * 0.5 + phi_start)
    if clockwise:
        # If the beam is in counter clockwise direction, we have to rotate the
        # rotation axis around its first basis vector - this will result in the
        # arc facing in the other direction. Additionally we have to rotate
        # around the z-axis for the angles to fit.
        t1 = [-np.sin(phi_start), np.cos(phi_start), 0.0]
        axis_rotation = Rotation([0, 0, 1], np.pi) * Rotation(t1, np.pi) * axis_rotation

    return create_beam_mesh_arc_segment(
        mesh,
        beam_object,
        material,
        center,
        axis_rotation,
        radius,
        np.abs(angle),
        **kwargs,
    )


def create_beam_mesh_line_at_node(
    mesh, beam_object, material, start_node, length, **kwargs
):
    """
    Generate a straight line at a given node. The tangent will be the same as at that node.

    Args
    ----
    mesh: Mesh
        Mesh that the arc segment will be added to.
    beam_object: Beam
        Class of beam that will be used for this line.
    material: Material
        Material for this segment.
    start_node: np.array, list
        Point where the arc will continue.
    length: float
        Length of the line.

    **kwargs (for all of them look into create_beam_mesh_function)
    ----
    n_el: int
        Number of equally spaced beam elements along the line. Defaults to 1.
        Mutually exclusive with l_el.
    l_el: float
        Desired length of beam elements. This requires the option interval_length
        to be set. Mutually exclusive with n_el. Be aware, that this length
        might not be achieved, if the elements are warped after they are
        created.

    Return
    ----
    return_set: GeometryName
        Set with the 'start' and 'end' node of the line. Also a 'line' set
        with all nodes of the line.
    """

    if length < 0:
        raise ValueError("Length has to be positive!")

    # Create the line starting from the given node
    start_node = get_node(start_node, check_cosserat_node=True)
    tangent = start_node.rotation * [1, 0, 0]
    start_position = start_node.coordinates
    end_position = start_position + tangent * length

    return create_beam_mesh_line(
        mesh,
        beam_object,
        material,
        start_position,
        end_position,
        start_node=start_node,
        **kwargs,
    )


def create_beam_mesh_arc_at_node(
    mesh, beam_object, material, start_node, arc_axis_normal, radius, angle, **kwargs
):
    """
    Generate a circular segment starting at a given node. The arc will be tangent to
    the given node.

    Args
    ----
    mesh: Mesh
        Mesh that the arc segment will be added to.
    beam_object: Beam
        Class of beam that will be used for this line.
    material: Material
        Material for this segment.
    start_node: np.array, list
        Point where the arc will continue.
    arc_axis_normal: 3d-vector
        Rotation axis for the created arc.
    radius: float
        The radius of the arc segment.
    angle: float
        Angle of the arc. If the angle is negative, the arc will point in the
        opposite direction, i.e., as if the arc_axis_normal would change sign.

    **kwargs (for all of them look into create_beam_mesh_function)
    ----
    n_el: int
        Number of equally spaced beam elements along the line. Defaults to 1.
        Mutually exclusive with l_el.
    l_el: float
        Desired length of beam elements. This requires the option interval_length
        to be set. Mutually exclusive with n_el. Be aware, that this length
        might not be achieved, if the elements are warped after they are
        created.

    Return
    ----
    return_set: GeometryName
        Set with the 'start' and 'end' node of the line. Also a 'line' set
        with all nodes of the line.
    """

    # If the angle is negative, the normal is switched
    arc_axis_normal = np.array(arc_axis_normal)
    if angle < 0:
        arc_axis_normal = -1.0 * arc_axis_normal

    # The normal has to be perpendicular to the start point tangent
    start_node = get_node(start_node, check_cosserat_node=True)
    tangent = start_node.rotation * [1, 0, 0]
    if np.abs(np.dot(tangent, arc_axis_normal)) > mpy.eps_pos:
        raise ValueError(
            "The normal has to be perpendicular to the tangent in the start node!"
        )

    # Get the center of the arc
    center_direction = np.cross(tangent, arc_axis_normal)
    center_direction *= 1.0 / np.linalg.norm(center_direction)
    center = start_node.coordinates - center_direction * radius

    # Create rotation for the general arc segment function
    axis_rotation = Rotation.from_rotation_matrix(
        np.transpose(np.array([tangent, -center_direction, arc_axis_normal]))
    )

    return create_beam_mesh_arc_segment(
        mesh,
        beam_object,
        material,
        center,
        axis_rotation,
        radius,
        np.abs(angle),
        start_node=start_node,
        **kwargs,
    )
