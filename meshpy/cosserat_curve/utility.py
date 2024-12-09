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
This file contains functionality to warp an existing mesh along a 1D curve
"""

# Import python modules
import numpy as np
from scipy import interpolate

# Import user modules
from meshpy import mpy, Mesh, BoundaryCondition, GeometrySet
from meshpy.node import Node
from meshpy.rotation import Rotation, smallest_rotation
from meshpy.function_utility import create_linear_interpolation_function
from meshpy.geometric_search import find_close_points


def get_piecewise_linear_arc_length_along_points(coordinates: np.array):
    """Return the accumulated distance between the points

    Args
    ----
    coordinates:
        Array containing the point coordinates
    """

    n_points = len(coordinates)
    point_distance = np.linalg.norm(coordinates[1:] - coordinates[:-1], axis=1)
    point_arc_length = np.zeros(n_points)
    for i in range(1, n_points):
        point_arc_length[i] = point_arc_length[i - 1] + point_distance[i - 1]
    return point_arc_length


def get_spline_interpolation(coordinates: np.array, point_arc_length: np.array):
    """Get a spline interpolation of the given points

    Args
    ----
    coordinates:
        Array containing the point coordinates
    point_arc_length:
        Arc length for each coordinate

    Return
    ----
    centerline_interpolation:
        The spline interpolation object
    """

    # Interpolate coordinates along arc length
    centerline_interpolation = interpolate.make_interp_spline(
        point_arc_length, coordinates
    )
    return centerline_interpolation


def get_quaternions_along_curve(centerline, point_arc_length):
    """Get the quaternions along the curve based on smallest rotation mappings.

    The initial rotation will be calculated based on the largest projection of the initial tangent
    onto the cartesian basis vectors.

    Args
    ----
    centerline:
        A function that returns the centerline position for a parameter coordinate t
    point_arc_length:
        Array of parameter coordinates for which the quaternions should be calculated
    """

    centerline_interpolation_derivative = centerline.derivative()

    def basis(i):
        """Retrun the i-th Cartesian basis vector"""
        basis = np.zeros([3])
        basis[i] = 1.0
        return basis

    # Get the reference rotation
    t0 = centerline_interpolation_derivative(point_arc_length[0])
    min_projection = np.argmin(np.abs([np.dot(basis(i), t0) for i in range(3)]))
    last_rotation = Rotation.from_basis(t0, basis(min_projection))

    # Get the rotation vectors along the curve. They are calculated with smallest rotation mappings.
    n_points = len(point_arc_length)
    quaternions = np.zeros([n_points, 4])
    quaternions[0] = last_rotation.q
    for i in range(1, n_points):
        rotation = smallest_rotation(
            last_rotation,
            centerline_interpolation_derivative(point_arc_length[i]),
        )
        quaternions[i] = rotation.q
        last_rotation = rotation
    return quaternions


def create_transform_boundary_conditions(
    mesh: Mesh,
    curve,
    *,
    nodes: list[Node] = None,
    n_steps: int = 10,
    t_end: float = 1.0,
    n_dof_per_node=3,
):
    """Create the Dirichlet boundary conditions that enforce the warping.
    The warped object is assumed to align with the x-axis in the reference configuration
    """

    # If no nodes are given, use all that are in the mesh
    if nodes is None:
        nodes = mesh.nodes

    time_values = np.linspace(0.0, t_end, n_steps + 1)

    # Get all positions that need to be warped
    positions_x = np.array([[node.coordinates[0], 0, 0] for node in nodes])

    # Get unique points along x-axis
    has_partner, n_partner = find_close_points(positions_x)
    positions_x_unique = [None for i in range(n_partner)]
    n_total = n_partner
    has_partner_total = [None for i in range(len(positions_x))]
    for i in range(len(positions_x)):
        partner_id = has_partner[i]
        if partner_id == -1:
            positions_x_unique[n_total] = positions_x[i][0]
            has_partner_total[i] = n_total
            n_total += 1
        else:
            if positions_x_unique[partner_id] is None:
                positions_x_unique[partner_id] = positions_x[i][0]
            has_partner_total[i] = partner_id

    positions_x_unique = np.array(positions_x_unique)
    positions_x_sorted_index = np.argsort(positions_x_unique)
    positions_x_sorted = positions_x_unique[positions_x_sorted_index]
    positions_x_sorted_index_inv = [None for i in range(n_total)]
    for i in range(n_total):
        positions_x_sorted_index_inv[positions_x_sorted_index[i]] = i
    point_to_unique = []
    for partner in has_partner_total:
        point_to_unique.append(positions_x_sorted_index_inv[partner])

    # Get all configurations for the unique points
    positions_for_all_steps = []
    quaternions_for_all_steps = []
    for fac in np.linspace(0, 1, n_steps + 1):
        sol_r, sol_q = curve.get_centerline_position_and_rotation_factor(
            positions_x_sorted, fac
        )
        positions_for_all_steps.append(sol_r)
        quaternions_for_all_steps.append(sol_q)

    # Loop over nodes and map them to the new configuration
    for i_node, node in enumerate(nodes):
        if not isinstance(node, Node):
            raise TypeError(
                "All nodes in the mesh have to be derived from the base Node object"
            )

        node_unique_id = point_to_unique[i_node]
        node_pos_ref = node.coordinates
        cross_section_position = node_pos_ref * np.array([0.0, 1.0, 1.0])

        # Check that the coordinates along the x-axis match
        if np.abs(node_pos_ref[0] - positions_x_sorted[node_unique_id]) > 1e-4:
            raise ValueError("Positions along x axis do not match")

        # Create the functions that describe the deformation
        displacement_values = np.zeros([n_steps + 1, 3])
        for i_step, fac in enumerate(np.linspace(0.0, 1.0, n_steps + 1)):
            centerline_pos = positions_for_all_steps[i_step][node_unique_id]
            q = quaternions_for_all_steps[i_step][node_unique_id]

            displacement_values[i_step] = (
                centerline_pos + Rotation(q) * cross_section_position - node_pos_ref
            )

        fun_pos = [
            create_linear_interpolation_function(
                time_values, displacement_values[:, i_dir]
            )
            for i_dir in range(3)
        ]
        for fun in fun_pos:
            mesh.add(fun)
        additional_dof = "0 " * (n_dof_per_node - 3)
        mesh.add(
            BoundaryCondition(
                GeometrySet(mpy.geo.point, nodes=[node]),
                "NUMDOF {4} ONOFF 1 1 1 {3}VAL 1.0 1.0 1.0 {3}FUNCT {0} {1} {2} {3}TAG monitor_reaction",
                format_replacement=fun_pos + [additional_dof, n_dof_per_node],
                bc_type=mpy.bc.dirichlet,
            )
        )

        # Print a status update every for every 10% of done work
        if i_node % round(len(nodes) / 10) == 0 or i_node + 1 == len(nodes):
            print(f"Done {i_node + 1}/{len(nodes)}")
