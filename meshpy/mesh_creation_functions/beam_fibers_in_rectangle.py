# -*- coding: utf-8 -*-
"""
This file has functions to create a honeycomb beam mesh.
"""

# Python packages.
import numpy as np

# Meshpy modules.
from .. import mpy, GeometrySet
from ..container import GeometryName
from ..utility import check_node_by_coordinate
from .beam_basic_geometry import create_beam_mesh_line


def _intersect_line_with_rectangle(length, width, start_line, direction_line,
        fail_if_no_intersection=True):
    """
    Calculate the intersection points between a line and a rectangle.

    Args
    ----
    length: scalar
        Rectangle length in x direction.
    width: scalar
        Rectangle width in y direction.
    start_line: 2d-list
        Point on the line.
    direction_line: 2d-list
        Direction of the line.
    fail_if_no_intersection: bool
        If this is true and no intersections are found, an error will be
        thrown.

    Return
    ----
    (start, end, projection_found)
    start: 2D vector
        Start point of intersected line.
    end: 2D vector
        End point of intersected line.
    projection_found: bool
        True if intersection is valid.
    """

    # Convert the input values to np.arrays.
    start_line = np.array(start_line)
    direction_line = np.array(direction_line)

    # Set definition for the boundary lines of the rectangle. The director is
    # chosen in a way, that the values [0, 1] for the line parameters alpha are
    # valid.
    # boundary_lines = [..., [start, dir], ...]
    boundary_lines = [
        [[0, 0], [length, 0]],
        [[0, 0], [0, width]],
        [[0, width], [length, 0]],
        [[length, 0], [0, width]]
        ]
    # Convert fo numpy arrays.
    boundary_lines = [[np.array(item) for item in boundary] for boundary
        in boundary_lines]

    # Loop over the boundaries.
    alpha_list = []
    for start_boundary, direction_boundary in boundary_lines:
        # Set up the linear system to solve the intersection problem.
        A = np.transpose(np.array([direction_line, -direction_boundary]))

        # Check if the system is solvable.
        if np.abs(np.linalg.det(A)) > 1e-10:
            alpha = np.linalg.solve(A, start_boundary - start_line)
            if (0 <= alpha[1] and alpha[1] <= 1):
                alpha_list.append(alpha[0])

    # Check that intersections were found.
    if len(alpha_list) < 2:
        if fail_if_no_intersection:
            raise ValueError('No intersections found!')
        return (None, None, False)

    # Return the start and end point on the line.
    return (
        start_line + min(alpha_list) * direction_line,
        start_line + max(alpha_list) * direction_line,
        True
        )


def create_fibers_in_rectangle(mesh, beam_object, material, length, width,
        angle, fiber_distance, fiber_element_length, *, offset=0.0):
    """
    Create multiple fibers in a rectangle.

    Args
    ----
    mesh: Mesh
        Mesh that the fibers will be added to.
    beam_object: Beam
        Object that will be used to create the beam elements.
    material: Material
        Material for the beam.
    length: float
        Length of the rectangle in x direction (starting at x=0)
    width: float
        Width of the rectangle in y direction (starting at y=0)
    angle: float
        Angle of the fibers in degree.
    fiber_distance: float
        Perpendicular distance between the fibers.
    fiber_element_length: float
        Length of a single beam element. In general it will not be possible to
        exactly achieve this length. If a line at a corner is shorter than this
        value, it will not be meshed.
    offset: double
        Fibers will be offset by this value from the symmetric layout.
    """

    if offset < 0:
        raise ValueError('The offset has to be positive!')
    if np.abs(offset) >= fiber_distance:
        raise ValueError('The offset has to smaller than the fiber distance!')

    # Get the fiber angle in rad.
    fiber_angle = angle * np.pi / 180.
    sin = np.sin(fiber_angle)
    cos = np.cos(fiber_angle)

    # The cosinus has to be positive for the algorithm to work.
    if cos < 0:
        cos = -cos
        sin = -sin

    # Direction and normal vector of the fibers.
    fiber_direction = np.array([cos, sin])
    fiber_normal = np.array([-sin, cos])

    # Get starting point for the creation of the fibers.
    if sin >= 0:
        fiber_start_point = np.array([length, 0])
        plate_diagonal = np.array([-length, width])
    else:
        fiber_start_point = np.array([0, 0])
        plate_diagonal = np.array([length, width])

    # Get the number of fibers in this layer.
    fiber_diagonal_distance = np.dot(fiber_normal, plate_diagonal)
    fiber_n = int(fiber_diagonal_distance // fiber_distance)

    # Calculate the offset, so the fibers are placed in the 'middle' of the
    # diagonal.
    fiber_offset = ((fiber_diagonal_distance / fiber_distance - fiber_n)
        * fiber_distance * 0.5 * fiber_normal)

    # Loop over each fiber and create the beam element.
    for n in range(-1, fiber_n + 1):

        # Get the start and end point of the line.
        start, end, projection_found = _intersect_line_with_rectangle(
            length, width,
            (fiber_offset + fiber_start_point
                + (n * fiber_distance + offset) * fiber_normal),
            fiber_direction,
            fail_if_no_intersection=False)

        if projection_found:
            # Calculate the length of the line.
            fiber_length = np.linalg.norm(end - start)

            # Create the beams if the length is not smaller than the fiber
            # distance.
            if fiber_length > fiber_distance:

                # Calculate the number of elements in this fiber.
                fiber_nel = int(fiber_length // fiber_element_length)
                fiber_nel = np.max([fiber_nel, 1])
                create_beam_mesh_line(mesh, beam_object, material,
                    np.append(start, 0.),
                    np.append(end, 0.),
                    n_el=fiber_nel)

    return_set = GeometryName()
    return_set['north'] = GeometrySet(mpy.geo.point,
        nodes=mesh.get_nodes_by_function(check_node_by_coordinate, 1, width))
    return_set['east'] = GeometrySet(mpy.geo.point,
        nodes=mesh.get_nodes_by_function(check_node_by_coordinate, 0, length))
    return_set['south'] = GeometrySet(mpy.geo.point,
        nodes=mesh.get_nodes_by_function(check_node_by_coordinate, 1, 0))
    return_set['west'] = GeometrySet(mpy.geo.point,
        nodes=mesh.get_nodes_by_function(check_node_by_coordinate, 0, 0))
    return_set['all'] = GeometrySet(mpy.geo.line,
        nodes=mesh.get_global_nodes())
    return return_set
