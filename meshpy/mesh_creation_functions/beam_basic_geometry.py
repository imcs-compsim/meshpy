# -*- coding: utf-8 -*-
"""
This file has functions to create basic geometry items with meshpy.
"""

# Python packages.
import numpy as np

# Meshpy modules.
from .. import Rotation


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
        The rotation of the segment axis in 3D and the segment starts on
        the local y-axis.
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