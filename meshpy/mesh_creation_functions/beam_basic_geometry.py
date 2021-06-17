# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# MeshPy: A beam finite element input generator.
#
# Copyright (c) 2021 Ivo Steinbrecher
#                    Institute for Mathematics and Computer-Based Simulation
#                    Universitaet der Bundeswehr Muenchen
#                    https://www.unibw.de/imcs-en
#
# TODO: Add license.
# -----------------------------------------------------------------------------
"""
This file has functions to create basic geometry items with meshpy.
"""

# Python packages.
import numpy as np

# Meshpy modules.
from ..conf import mpy
from ..rotation import Rotation


def create_beam_mesh_line(mesh, beam_object, material, start_point,
        end_point, **kwargs):
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

    **kwargs (for all of them look into Mesh().create_beam_mesh_function)
    ----
    n_el: int
        Number of equally spaces beam elements along the line.
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
    t1 = direction / np.linalg.norm(direction)

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
            return (
                0.5 * (1 - xi) * point_a + 0.5 * (1 + xi) * point_b,
                rotation
                )
        return beam_function

    # Create the beam in the mesh
    return mesh.create_beam_mesh_function(beam_object=beam_object,
        material=material, function_generator=get_beam_geometry,
        interval=[0., 1.], **kwargs)


def create_beam_mesh_arc_segment(mesh, beam_object, material, center,
        axis_rotation, radius, angle, **kwargs):
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
        The 3rd base vector of this rotation is the rotation axis of the arc
        segment. The segment starts on the 2nd basis vector.
    radius: float
        The radius of the segment.
    angle: float
        The central angle of this segment in radians.

    **kwargs (for all of them look into Mesh().create_beam_mesh_function)
    ----
    n_el: int
        Number of equally spaces beam elements along the segment.

    Return
    ----
    return_set: GeometryName
        Set with the 'start' and 'end' node of the line. Also a 'line' set
        with all nodes of the line.
    """

    # The angle can not be negative with the current implementation.
    if angle <= 0.:
        raise ValueError('The angle for a beam segment has to be a ' +
            'positive number!')

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
    return mesh.create_beam_mesh_function(beam_object=beam_object,
        material=material, function_generator=get_beam_geometry,
        interval=[0., angle], **kwargs)


def create_beam_mesh_arc_segment_2d(mesh, beam_object, material, center,
        radius, phi_start, phi_end, **kwargs):
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

    **kwargs (for all of them look into Mesh().create_beam_mesh_function)
    ----
    n_el: int
        Number of equally spaces beam elements along the segment.

    Return
    ----
    return_set: GeometryName
        Set with the 'start' and 'end' node of the line. Also a 'line' set
        with all nodes of the line.
    """

    # The center point has to be on the x-y plane.
    if np.abs(center[2]) > mpy.eps_pos:
        raise ValueError('The z-value of center has to be 0!')

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
        axis_rotation = (Rotation([0, 0, 1], np.pi) * Rotation(t1, np.pi) *
            axis_rotation)

    return create_beam_mesh_arc_segment(mesh, beam_object, material, center,
            axis_rotation, radius, np.abs(angle), **kwargs)
