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
This module implements a basic class to manage geometry in the input file.
"""

# Meshpy modules.
from .conf import mpy
from .base_mesh_item import BaseMeshItem
from .node import Node


class GeometrySet(BaseMeshItem):
    """This object represents a geometry set. The set is defined by nodes."""

    # Node set names for the input file file.
    geometry_set_names = {
        mpy.geo.point: 'DNODE',
        mpy.geo.line: 'DLINE',
        mpy.geo.surface: 'DSURFACE',
        mpy.geo.volume: 'DVOL'
        }

    def __init__(self, geometry_type, nodes=None, fail_on_double_nodes=True,
            **kwargs):
        """
        Initialize the geometry set.

        Args
        ----
        geometry_type: mpy.geo
            Type of geometry. This is  necessary, as the boundary conditions
            and input file depend on that type.
        nodes: Node, list(Nodes)
            Node(s) or list of nodes to be added to this geometry set.
        fail_on_double_nodes: bool
            If True, an error will be thrown if the same node is added twice.
            If False, the node will only be added once.
        """
        super().__init__(is_dat=None, **kwargs)

        self.geometry_type = geometry_type
        self.nodes = []

        if nodes is not None:
            self._add(nodes, fail_on_double_nodes)

    @classmethod
    def from_dat(cls, geometry_key, lines, comments=None):
        """
        Get a geometry set from an input line in a dat file. The geometry set
        is passed as integer (0 based index) and will be connected after the
        whole input file is parsed.
        """

        # Split up the input line.
        nodes = []
        for line in lines:
            nodes.append(int(line.split()[1]) - 1)

        # Set up class with values for solid mesh import
        return cls(geometry_key, nodes=nodes, comments=comments)

    def _add(self, value, fail_on_double_nodes):
        """
        Add nodes to this object.

        Args
        ----
        value: Node, list(Nodes)
            Node(s) or list of nodes to be added to this geometry set.
        fail_on_double_nodes: bool
            If True, an error will be thrown if the same node is added twice.
            If False, the node will only be added once.
        """

        if isinstance(value, list):
            # Loop over items and check if they are either Nodes or integers.
            # This improves the performance considerably when large list of
            # Nodes are added.
            for item in value:
                self._add(item, fail_on_double_nodes)
        elif isinstance(value, Node) or isinstance(value, int):
            if value not in self.nodes:
                self.nodes.append(value)
            elif fail_on_double_nodes:
                raise ValueError('The node already exists in this set!')
        elif isinstance(value, GeometrySet):
            # Add all nodes from this geometry set.
            for node in value.nodes:
                self._add(node, fail_on_double_nodes)
        else:
            raise TypeError('Expected Node or list, but got {}'.format(
                type(value)
                ))

    def check_replaced_nodes(self):
        """Check if nodes in this set have been replaced."""

        for node in self.nodes:
            if node.master_node is not None:
                self.replace_node(node, node.get_master_node())

    def replace_node(self, old_node, new_node):
        """Replace old_node with new_node."""

        # Check if the new node is in the set.
        has_new_node = new_node in self.nodes

        for i, node in enumerate(self.nodes):
            if node == old_node:
                if has_new_node:
                    del self.nodes[i]
                else:
                    self.nodes[i] = new_node
                break
        else:
            raise ValueError('The node that should be replaced is not in the '
                + 'current node set')

    def __iter__(self):
        for node in self.nodes:
            yield node

    def _get_dat(self):
        """Get the lines for the input file."""
        return ['NODE {} {} {}'.format(
            node.n_global,
            self.geometry_set_names[self.geometry_type], self.n_global
            ) for node in self.nodes]
