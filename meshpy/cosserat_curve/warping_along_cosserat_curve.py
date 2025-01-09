# MeshPy: A beam finite element input generator
#
# MIT License
#
# Copyright (c) 2018-2025
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
"""This file contains functionality to warp an existing mesh along a 1D
curve."""

from typing import Optional, Tuple

import numpy as np
import quaternion
from numpy.typing import NDArray

from .. import mpy
from ..boundary_condition import BoundaryCondition
from ..function_utility import create_linear_interpolation_function
from ..geometric_search import find_close_points
from ..geometry_set import GeometrySet
from ..mesh import Mesh
from ..node import Node, NodeCosserat
from ..rotation import Rotation
from .cosserat_curve import CosseratCurve


def get_arc_length_and_cross_section_coordinates(
    coordinates: np.ndarray, origin: np.ndarray, reference_rotation: Rotation
) -> Tuple[float, np.ndarray]:
    """Return the arc length and the cross section coordinates for a coordinate
    system defined by the reference rotation and the origin.

    Args
    ----
    coordinates:
        Point coordinates in R3
    origin:
        Origin of the coordinate system
    reference_rotation:
        Rotation of the coordinate system. The first basis vector is the arc
        length direction.
    """

    transformed_coordinates = reference_rotation.inv() * (coordinates - origin)
    centerline_position = transformed_coordinates[0]
    cross_section_coordinates = [0.0, *transformed_coordinates[1:]]
    return centerline_position, cross_section_coordinates


def get_mesh_transformation(
    curve: CosseratCurve,
    nodes: list[Node],
    *,
    origin=[0.0, 0.0, 0.0],
    reference_rotation=Rotation(),
    n_steps: int = 10,
    initial_configuration: bool = True,
    **kwargs,
) -> Tuple[np.ndarray, NDArray[quaternion.quaternion]]:
    """Generate a list of positions for each node that describe the
    transformation of the nodes from the given configuration to the Cosserat
    curve.

    Args
    ----
    curve:
        Curve to warp the mesh to
    nodes:
        Optional, if this is given only warp the given nodes. Per default all nodes in
        the mesh are warped.
    origin:
        Origin of the coordinate system
    reference_rotation:
        Rotation of the coordinate system. The first basis vector is the arc
        length direction.
    n_steps:
        Number of steps to get from the unwrapped configuration to the final configuration.
    initial_configuration:
        If the initial, unwrapped configuration (factor=0) should also be added to the
        results.
    kwargs:
        Keyword arguments passed to CosseratCurve.get_centerline_positions_and_rotations

    Return
    ----
    positions: list(np.array(n_nodes x 3))
        A list for each time step containing the position of all nodes for that time step
    relative_rotations: list(list(Rotation))
        A list for each time step containing the relative rotations for all nodes at that
        time step
    """

    # Define the factors for which we will generate the positions and rotations
    factors = np.linspace(0.0, 1.0, n_steps + 1)
    if initial_configuration:
        n_output_steps = n_steps + 1
    else:
        n_output_steps = n_steps
        factors = np.delete(factors, 0)

    # Create output arrays
    n_nodes = len(nodes)
    positions = np.zeros((n_output_steps, n_nodes, 3))
    relative_rotations = np.zeros(
        (n_output_steps, n_nodes), dtype=quaternion.quaternion
    )

    # Get all arc lengths and cross section positions
    arc_lengths = np.zeros((n_nodes, 1))
    cross_section_coordinates = [None] * n_nodes
    for i_node, node in enumerate(nodes):
        (
            arc_lengths[i_node],
            cross_section_coordinates[i_node],
        ) = get_arc_length_and_cross_section_coordinates(
            node.coordinates, origin, reference_rotation
        )

    # Get unique arc length points
    has_partner, n_partner = find_close_points.find_close_points(arc_lengths)
    arc_lengths_unique = [None] * n_partner
    has_partner_total = [-2] * len(arc_lengths)
    for i in range(len(arc_lengths)):
        partner_id = has_partner[i]
        if partner_id == -1:
            has_partner_total[i] = len(arc_lengths_unique)
            arc_lengths_unique.append(arc_lengths[i][0])
        else:
            if arc_lengths_unique[partner_id] is None:
                arc_lengths_unique[partner_id] = arc_lengths[i][0]
            has_partner_total[i] = partner_id

    n_total = len(arc_lengths_unique)
    arc_lengths_unique = np.array(arc_lengths_unique)
    arc_lengths_sorted_index = np.argsort(arc_lengths_unique)
    arc_lengths_sorted = arc_lengths_unique[arc_lengths_sorted_index]
    arc_lengths_sorted_index_inv = [-2 for i in range(n_total)]
    for i in range(n_total):
        arc_lengths_sorted_index_inv[arc_lengths_sorted_index[i]] = i
    point_to_unique = []
    for partner in has_partner_total:
        point_to_unique.append(arc_lengths_sorted_index_inv[partner])

    # Get all configurations for the unique points
    positions_for_all_steps = []
    quaternions_for_all_steps = []

    for factor in factors:
        sol_r, sol_q = curve.get_centerline_positions_and_rotations(
            arc_lengths_sorted, factor=factor, **kwargs
        )
        positions_for_all_steps.append(sol_r)
        quaternions_for_all_steps.append(sol_q)

    # Get data required for the rigid body motion
    curve_start_pos, curve_start_rot = curve.get_centerline_position_and_rotation(0.0)
    rigid_body_translation = curve_start_pos - origin
    rigid_body_rotation = curve_start_rot

    # Loop over nodes and map them to the new configuration
    for i_node, node in enumerate(nodes):
        if not isinstance(node, Node):
            raise TypeError(
                "All nodes in the mesh have to be derived from the base Node object"
            )

        node_unique_id = point_to_unique[i_node]
        cross_section_position = cross_section_coordinates[i_node]

        # Check that the arc length coordinates match
        if (
            np.abs(arc_lengths[i_node] - arc_lengths_sorted[node_unique_id])
            > mpy.eps_pos
        ):
            raise ValueError("Arc lengths do not match")

        # Create the functions that describe the deformation
        for i_step, factor in enumerate(factors):
            centerline_pos = positions_for_all_steps[i_step][node_unique_id]
            centerline_relative_pos = quaternion.rotate_vectors(
                curve_start_rot.conjugate(), centerline_pos - curve_start_pos
            )
            centerline_rotation = quaternions_for_all_steps[i_step][node_unique_id]
            centerline_relative_rotation = (
                curve_start_rot.conjugate() * centerline_rotation
            )

            rigid_body_rotation_for_factor = quaternion.slerp_evaluate(
                quaternion.from_float_array(reference_rotation.q),
                rigid_body_rotation,
                factor,
            )

            current_pos = (
                quaternion.rotate_vectors(
                    rigid_body_rotation_for_factor,
                    (
                        centerline_relative_pos
                        + quaternion.rotate_vectors(
                            centerline_relative_rotation, cross_section_position
                        )
                    ),
                )
                + origin
                + factor * rigid_body_translation
            )

            positions[i_step, i_node] = current_pos
            relative_rotations[i_step, i_node] = (
                rigid_body_rotation_for_factor
                * centerline_relative_rotation
                * quaternion.from_float_array(reference_rotation.q).conjugate()
            )

    return positions, relative_rotations


def create_transform_boundary_conditions(
    mesh: Mesh,
    curve: CosseratCurve,
    *,
    nodes: Optional[list[Node]] = None,
    t_end: float = 1.0,
    n_steps: int = 10,
    n_dof_per_node: int = 3,
    **kwargs,
) -> None:
    """Create the Dirichlet boundary conditions that enforce the warping. The
    warped object is assumed to align with the z-axis in the reference
    configuration.

    Args
    ----
    mesh:
        Mesh to be warped
    curve:
        Curve to warp the mesh to
    nodes:
        Optional, if this is given only warp the given nodes. Per default all nodes in
        the mesh are warped.
    n_steps:
        Number of steps to apply the warping condition
    t_end:
        End time for applying the warping boundary conditions
    n_dof_per_node:
        Number of DOF per node in 4C (is needed to correctly define the boundary conditions)
    kwargs:
        Keyword arguments passed to get_mesh_transformation
    """

    # If no nodes are given, use all nodes in the mesh
    if nodes is None:
        nodes = mesh.nodes

    time_values = np.linspace(0.0, t_end, n_steps + 1)

    # Get positions and rotations for each step
    positions, _ = get_mesh_transformation(curve, nodes, n_steps=n_steps, **kwargs)

    # Loop over nodes and map them to the new configuration
    for i_node, node in enumerate(nodes):
        # Create the functions that describe the deformation
        reference_position = node.coordinates
        displacement_values = np.array(
            [
                positions[i_step][i_node] - reference_position
                for i_step in range(n_steps + 1)
            ]
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
                GeometrySet(node),
                "NUMDOF {4} ONOFF 1 1 1 {3}VAL 1.0 1.0 1.0 {3}FUNCT {0} {1} {2} {3}TAG monitor_reaction",
                format_replacement=fun_pos + [additional_dof, n_dof_per_node],
                bc_type=mpy.bc.dirichlet,
            )
        )


def warp_mesh_along_curve(
    mesh: Mesh,
    curve: CosseratCurve,
    *,
    origin=[0.0, 0.0, 0.0],
    reference_rotation=Rotation(),
) -> None:
    """Warp an existing mesh along the given curve.

    The reference coordinates for the transformation are defined by the
    given origin and rotation, where the first basis vector of the triad
    defines the centerline axis.
    """

    pos, rot = get_mesh_transformation(
        curve,
        mesh.nodes,
        origin=origin,
        reference_rotation=reference_rotation,
        n_steps=1,
        initial_configuration=False,
    )

    # Loop over nodes and map them to the new configuration
    for i_node, node in enumerate(mesh.nodes):
        if not isinstance(node, Node):
            raise TypeError(
                "All nodes in the mesh have to be derived from the base Node object"
            )

        new_pos = pos[0, i_node]
        node.coordinates = new_pos
        if isinstance(node, NodeCosserat):
            node.rotation = (
                Rotation.from_quaternion(quaternion.as_float_array(rot[0, i_node]))
                * node.rotation
            )
