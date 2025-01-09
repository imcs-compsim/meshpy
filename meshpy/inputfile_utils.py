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
"""This module defines utility functions for creating input files from MeshPy
meshes."""

from typing import Dict, List, Tuple

from .conf import mpy
from .mesh import Mesh
from .node import Node


def get_coupled_nodes_to_master_map(
    mesh: Mesh, *, assign_n_global: bool = False
) -> Tuple[Dict[Node, Node], List[Node]]:
    """Get a mapping of nodes in a mesh that should be "replaced" because they
    are coupled via a joint.

    In some finite element (FE) solvers, nodes coupled via joints are resolved
    by assigning a "master" node to represent the joint. This function identifies
    such nodes and creates a mapping where each coupled node is mapped to its
    master node.

    Args
    ----
    mesh:
        Input mesh
    assign_n_global:
        If this flag is set, the global indices are set in the node objects.

    Return
    ----
    replaced_node_to_master_map:
        A dictionary mapping each "replaced" node to its "master" node.
    unique_nodes:
        A list containing all unique nodes in the mesh, i.e., all nodes which
        are not coupled and the master nodes.
    """

    # Get a dictionary that maps the "replaced" nodes to the "master" ones
    replaced_node_to_master_map = {}
    for coupling in mesh.boundary_conditions[mpy.bc.point_coupling, mpy.geo.point]:
        if coupling.coupling_dof_type is not mpy.coupling_dof.fix:
            raise ValueError(
                "This function is only implemented for rigid joints at the DOFs"
            )
        coupling_nodes = coupling.geometry_set.get_points()
        for node in coupling_nodes[1:]:
            replaced_node_to_master_map[node] = coupling_nodes[0]

    # Check that no "replaced" node is a "master" node
    master_nodes = set(replaced_node_to_master_map.values())
    for replaced_node in replaced_node_to_master_map.keys():
        if replaced_node in master_nodes:
            raise ValueError(
                "A replaced node is also a master nodes. This is not supported"
            )

    # Get all unique nodes
    unique_nodes = [
        node for node in mesh.nodes if node not in replaced_node_to_master_map
    ]

    # Optionally number the nodes
    if assign_n_global:
        for i_node, node in enumerate(unique_nodes):
            node.n_global = i_node
        for replaced_node, master_node in replaced_node_to_master_map.items():
            replaced_node.n_global = master_node.n_global

    # Return the mapping
    return replaced_node_to_master_map, unique_nodes
