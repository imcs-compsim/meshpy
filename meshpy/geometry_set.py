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
"""This module implements a basic class to manage geometry in the input
file."""

# Python modules.
import numpy as np

from .base_mesh_item import BaseMeshItemFull

# Meshpy modules.
from .conf import mpy
from .element_beam import Beam
from .node import Node


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
        """Initialize the geometry set.

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

    def link_to_nodes(self, *, link_to_nodes="explicitly_contained_nodes"):
        """Set a link to this object in the all contained nodes of this
        geometry set.

        link_to_nodes: str
            "explicitly_contained_nodes":
                A link will be set for all nodes that are explicitly part of the geometry set
            "all_nodes":
                A link will be set for all nodes that are part of the geometry set, i.e., also
                nodes connected to elements of an element set. This is mainly used for vtk
                output so we can color the nodes which are part of element sets.
        """
        if link_to_nodes == "explicitly_contained_nodes":
            node_list = self.get_node_dict().keys()
        elif link_to_nodes == "all_nodes":
            node_list = self.get_all_nodes()
        else:
            raise ValueError(f'Got unexpected value link nodes="{link_to_nodes}"')
        for node in node_list:
            node.node_sets_link.append(self)

    def check_replaced_nodes(self):
        """Check if nodes in this set have to be replaced.

        We need to do this for explicitly contained nodes in this set.
        """
        # Don't iterate directly over the keys as the dict changes during this iteration
        for node in list(self.get_node_dict().keys()):
            if node.master_node is not None:
                self.replace_node(node, node.get_master_node())

    def replace_node(self, old_node, new_node):
        """Replace old_node with new_node."""

        explicit_nodes_in_this_set = self.get_node_dict()
        explicit_nodes_in_this_set[new_node] = None
        del explicit_nodes_in_this_set[old_node]

    def get_node_dict(self):
        """Return the dictionary containing the explicitly added nodes for this
        set."""
        raise NotImplementedError(
            'The "get_node_dict" method has to be overwritten in the derived class'
        )

    def get_points(self):
        """Return nodes explicitly associated with this set."""
        raise NotImplementedError(
            'The "get_points" method has to be overwritten in the derived class'
        )

    def get_all_nodes(self):
        """Return all nodes associated with this set.

        This includes nodes contained within the geometry added to this
        set.
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
        """Initialize the geometry set.

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
            self.geometry_objects[geo] = {}
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
        """Add a geometry item to this object."""

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
            self.geometry_objects[self.geometry_type][item] = None
        else:
            raise TypeError(f"Got unexpected geometry type {type(item)}")

    def get_node_dict(self):
        """Return the dictionary containing the explicitly added nodes for this
        set.

        For non-point sets an empty dict is returned.
        """
        if self.geometry_type is mpy.geo.point:
            return self.geometry_objects[mpy.geo.point]
        else:
            return {}

    def get_points(self):
        """Return nodes explicitly associated with this set.

        Only in case this is a point set something is returned here.
        """
        if self.geometry_type is mpy.geo.point:
            return list(self.geometry_objects[mpy.geo.point].keys())
        else:
            raise TypeError(
                "The function get_points can only be called for point sets."
                f" The present type is {self.geometry_type}"
            )

    def get_all_nodes(self):
        """Return all nodes associated with this set.

        This includes nodes contained within the geometry added to this
        set.
        """

        if self.geometry_type is mpy.geo.point:
            return list(self.geometry_objects[mpy.geo.point].keys())
        elif self.geometry_type is mpy.geo.line:
            nodes = []
            for element in self.geometry_objects[mpy.geo.line].keys():
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
        """Initialize the geometry set.

        Args
        ----
        geometry_type: mpy.geo
            Type of geometry. This is  necessary, as the boundary conditions
            and input file depend on that type.
        nodes: Node, GeometrySetNodes, list(Nodes), list(GeometrySetNodes)
            Node(s) or list of nodes to be added to this geometry set.
        """

        super().__init__(geometry_type, **kwargs)
        self.nodes = {}
        if nodes is not None:
            self.add(nodes)

    @classmethod
    def from_dat(cls, geometry_key, lines, comments=None):
        """Get a geometry set from an input line in a dat file.

        The geometry set is passed as integer (0 based index) and will
        be connected after the whole input file is parsed.
        """
        nodes = []
        for line in lines:
            nodes.append(int(line.split()[1]) - 1)
        return cls(geometry_key, nodes, comments=comments)

    def replace_indices_with_nodes(self, nodes):
        """After the set is imported from a dat file, replace the node indices
        with the corresponding nodes objects."""
        node_dict = {}
        for node_id in self.nodes.keys():
            node_dict[nodes[node_id]] = None
        self.nodes = node_dict

    def add(self, value):
        """Add nodes to this object.

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
            self.nodes[value] = None
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

    def get_node_dict(self):
        """Return the dictionary containing the explicitly added nodes for this
        set."""
        return self.nodes

    def get_points(self):
        """Return nodes explicitly associated with this set."""
        if self.geometry_type is mpy.geo.point:
            return self.get_all_nodes()
        else:
            raise TypeError(
                "The function get_points can only be called for point sets."
                f" The present type is {self.geometry_type}"
            )

    def get_all_nodes(self):
        """Return all nodes associated with this set."""
        return list(self.nodes.keys())
