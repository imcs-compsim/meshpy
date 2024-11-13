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
This module implements a basic class to manage geometry in the input file.
"""

# Python modules.
import numpy as np

# Meshpy modules.
from .conf import mpy
from .base_mesh_item import BaseMeshItemFull
from .node import Node
from .element_beam import Beam


class GeometrySetBase(BaseMeshItemFull):
    """Base class for a geometry set."""

    # Node set names for the input file file.
    geometry_set_names = {
        mpy.geo.point: "DNODE",
        mpy.geo.line: "DLINE",
        mpy.geo.surface: "DSURFACE",
        mpy.geo.volume: "DVOL",
    }

    def __init__(self, geometry_type, name=None, **kwargs):
        """
        Initialize the geometry set.

        Args
        ----
        geometry_type: mpy.geo
            Type of geometry. MeshPy only supports geometry sets of a single
            specified geometry type.
        name: str
            Optional name to identify this geometry set.
        """
        super().__init__(**kwargs)

        self.geometry_type = geometry_type
        self.name = name

    def link_to_nodes(self):
        """Set a link to this object in the all contained nodes of this geometry set.
        This will also set a link to nodes connected to an edge in a node based line
        set.
        """
        for node in self.get_all_nodes():
            node.node_sets_link.append(self)

    def check_replaced_nodes(self):
        """
        Check if nodes in this set have to be replaced.
        We need to do this for all nodes to correctly represent node-based sets.
        """
        for node in self.get_all_nodes():
            if node.master_node is not None:
                self.replace_node(node, node.get_master_node())

    def replace_node(self, old_node, new_node):
        """Replace old_node with new_node. This is only done for point sets."""

        # Check if the new node is in the set.
        my_nodes = self.get_all_nodes()
        has_new_node = new_node in my_nodes

        for i, node in enumerate(my_nodes):
            if node == old_node:
                if has_new_node:
                    del my_nodes[i]
                else:
                    my_nodes[i] = new_node
                break
        else:
            raise ValueError(
                "The node that should be replaced is not in the current node set"
            )

    def get_all_nodes(self):
        """
        Return all nodes associated with this set. This includes nodes contained within
        the geometry added to this set.
        """
        raise NotImplementedError(
            'The "get_all_nodes" method has to be overwritten in the derived class'
        )

    def _get_dat(self):
        """Get the lines for the input file."""

        # Sort the nodes based on the node GID.
        nodes = self.get_all_nodes()
        if len(nodes) == 0:
            raise ValueError("Writing empty geometry sets is not supported")
        nodes_id = [node.n_global for node in nodes]
        sort_indices = np.argsort(nodes_id)
        nodes = [nodes[i] for i in sort_indices]

        return [
            f"NODE {node.n_global} {self.geometry_set_names[self.geometry_type]} {self.n_global}"
            for node in nodes
        ]


class GeometrySet(GeometrySetBase):
    """Geometry set which is defined by geometric entries."""

    def __init__(self, geometry, **kwargs):
        """
        Initialize the geometry set.

        Args
        ----
        geometry: List or single Geometry/GeometrySet
            Geometries associated with this set. Empty geometries (i.e., no given)
            are not supported.
        """

        # This is ok, we check every single type in the add method
        if isinstance(geometry, list):
            geometry_type = self._get_geometry_type(geometry[0])
        else:
            geometry_type = self._get_geometry_type(geometry)

        super().__init__(geometry_type, **kwargs)

        self.geometry_objects = {}
        for geo in mpy.geo:
            self.geometry_objects[geo] = []
        self.add(geometry)

    @staticmethod
    def _get_geometry_type(item):
        """Return the geometry type of a given item."""

        if isinstance(item, Node):
            return mpy.geo.point
        elif isinstance(item, Beam):
            return mpy.geo.line
        elif isinstance(item, GeometrySet):
            return item.geometry_type
        raise TypeError(f"Got unexpected type {type(item)}")

    def add(self, item):
        """
        Add a geometry item to this object.
        """

        if isinstance(item, list):
            for sub_item in item:
                self.add(sub_item)
        elif isinstance(item, GeometrySet):
            if item.geometry_type is self.geometry_type:
                for geometry in item.geometry_objects[self.geometry_type]:
                    self.add(geometry)
            else:
                raise TypeError(
                    "You tried to add a {item.geometry_type} set to a {self.geometry_type} set. "
                    "This is not possible"
                )
        elif self._get_geometry_type(item) is self.geometry_type:
            if item not in self.geometry_objects[self.geometry_type]:
                self.geometry_objects[self.geometry_type].append(item)
        else:
            raise TypeError(f"Got unexpected geometry type {type(item)}")

    def get_points(self):
        """
        Return nodes explicitly associated with this set. Only in case this is a
        point set something is returned here.
        """
        if self.geometry_type is mpy.geo.point:
            return self.geometry_objects[mpy.geo.point]
        else:
            raise TypeError(
                "The function get_points can only be called for point sets."
                f" The present type is {self.geometry_type}"
            )

    def get_all_nodes(self):
        """
        Return all nodes associated with this set. This includes nodes contained within
        the geometry added to this set.
        """

        if self.geometry_type is mpy.geo.point:
            return self.geometry_objects[mpy.geo.point]
        elif self.geometry_type is mpy.geo.line:
            nodes = []
            for element in self.geometry_objects[mpy.geo.line]:
                nodes.extend(element.nodes)
            # Remove duplicates while preserving order
            return list(dict.fromkeys(nodes))
        else:
            raise TypeError(
                "Currently GeometrySet are only implemented for points and lines"
            )


class GeometrySetNodes(GeometrySetBase):
    """Geometry set which is defined by nodes and not explicit geometry."""

    def __init__(self, geometry_type, nodes=None, **kwargs):
        """
        Initialize the geometry set.

        Args
        ----
        geometry_type: mpy.geo
            Type of geometry. This is  necessary, as the boundary conditions
            and input file depend on that type.
        nodes: Node, GeometrySetNodes, list(Nodes), list(GeometrySetNodes)
            Node(s) or list of nodes to be added to this geometry set.
        """

        super().__init__(geometry_type, **kwargs)
        self.nodes = []
        if nodes is not None:
            self.add(nodes)

    @classmethod
    def from_dat(cls, geometry_key, lines, comments=None):
        """
        Get a geometry set from an input line in a dat file. The geometry set
        is passed as integer (0 based index) and will be connected after the
        whole input file is parsed.
        """
        nodes = []
        for line in lines:
            nodes.append(int(line.split()[1]) - 1)
        return cls(geometry_key, nodes, comments=comments)

    def replace_indices_with_nodes(self, nodes):
        """After the set is imported from a dat file, replace the node indices
        with the corresponding nodes objects."""
        for i, node in enumerate(self.nodes):
            self.nodes[i] = nodes[node]

    def add(self, value):
        """
        Add nodes to this object.

        Args
        ----
        nodes: Node, GeometrySetNodes, list(Nodes), list(GeometrySetNodes)
            Node(s) or list of nodes to be added to this geometry set.
        """

        if isinstance(value, list):
            # Loop over items and check if they are either Nodes or integers.
            # This improves the performance considerably when large list of
            # Nodes are added.
            for item in value:
                self.add(item)
        elif isinstance(value, (int, Node)):
            if value not in self.nodes:
                self.nodes.append(value)
        elif isinstance(value, GeometrySetNodes):
            # Add all nodes from this geometry set.
            if self.geometry_type == value.geometry_type:
                for node in value.nodes:
                    self.add(node)
            else:
                raise TypeError(
                    f"You tried to add a {value.geometry_type} set to a {self.geometry_type} set. "
                    "This is not possible"
                )
        else:
            raise TypeError(f"Expected Node or list, but got {type(value)}")

    def get_points(self):
        """Return nodes explicitly associated with this set."""
        if self.geometry_type is mpy.geo.point:
            return self.nodes
        else:
            raise TypeError(
                "The function get_points can only be called for point sets."
                f" The present type is {self.geometry_type}"
            )

    def get_all_nodes(self):
        """Return all nodes associated with this set."""
        return self.nodes
