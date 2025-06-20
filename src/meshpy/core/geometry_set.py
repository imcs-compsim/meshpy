# The MIT License (MIT)
#
# Copyright (c) 2018-2025 MeshPy Authors
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""This module implements a basic class to manage geometry in the input
file."""

from meshpy.core.base_mesh_item import BaseMeshItem as _BaseMeshItem
from meshpy.core.conf import mpy as _mpy
from meshpy.core.container import ContainerBase as _ContainerBase
from meshpy.core.element_beam import Beam as _Beam
from meshpy.core.node import Node as _Node


class GeometrySetBase(_BaseMeshItem):
    """Base class for a geometry set."""

    # Node set names for the input file file.
    geometry_set_names = {
        _mpy.geo.point: "DNODE",
        _mpy.geo.line: "DLINE",
        _mpy.geo.surface: "DSURFACE",
        _mpy.geo.volume: "DVOL",
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

    def dump_to_list(self):
        """Return a list with the legacy strings of this geometry set."""

        # Sort nodes based on their global index
        nodes = sorted(self.get_all_nodes(), key=lambda n: n.i_global)

        if not nodes:
            raise ValueError("Writing empty geometry sets is not supported")

        return [
            {
                "type": "NODE",
                "node_id": node.i_global,
                "d_type": self.geometry_set_names[self.geometry_type],
                "d_id": self.i_global,
            }
            for node in nodes
        ]

    def __add__(self, other):
        """Allow to add two geometry sets to each other and return a geometry
        set with the geometries from both sets."""
        combined_set = self.copy()
        combined_set.add(other)
        return combined_set


class GeometrySet(GeometrySetBase):
    """Geometry set which is defined by geometric entries."""

    def __init__(self, geometry, **kwargs):
        """Initialize the geometry set.

        Args
        ----
        geometry: _List or single Geometry/GeometrySet
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
        for geo in _mpy.geo:
            self.geometry_objects[geo] = {}
        self.add(geometry)

    @staticmethod
    def _get_geometry_type(item):
        """Return the geometry type of a given item."""

        if isinstance(item, _Node):
            return _mpy.geo.point
        elif isinstance(item, _Beam):
            return _mpy.geo.line
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
        if self.geometry_type is _mpy.geo.point:
            return self.geometry_objects[_mpy.geo.point]
        else:
            return {}

    def get_points(self):
        """Return nodes explicitly associated with this set.

        Only in case this is a point set something is returned here.
        """
        if self.geometry_type is _mpy.geo.point:
            return list(self.geometry_objects[_mpy.geo.point].keys())
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

        if self.geometry_type is _mpy.geo.point:
            return list(self.geometry_objects[_mpy.geo.point].keys())
        elif self.geometry_type is _mpy.geo.line:
            nodes = []
            for element in self.geometry_objects[_mpy.geo.line].keys():
                nodes.extend(element.nodes)
            # Remove duplicates while preserving order
            return list(dict.fromkeys(nodes))
        else:
            raise TypeError(
                "Currently GeometrySet are only implemented for points and lines"
            )

    def get_geometry_objects(self):
        """Return a list of the objects with the specified geometry type."""
        return list(self.geometry_objects[self.geometry_type].keys())

    def copy(self):
        """Return a shallow copy of this object, the reference to the nodes
        will be the same, but the containers storing them will be copied."""
        return GeometrySet(list(self.geometry_objects[self.geometry_type].keys()))


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

        if geometry_type not in _mpy.geo:
            raise TypeError(f"Expected geometry enum, got {geometry_type}")

        super().__init__(geometry_type, **kwargs)
        self.nodes = {}
        if nodes is not None:
            self.add(nodes)

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
        elif isinstance(value, (int, _Node)):
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
        if self.geometry_type is _mpy.geo.point:
            return self.get_all_nodes()
        else:
            raise TypeError(
                "The function get_points can only be called for point sets."
                f" The present type is {self.geometry_type}"
            )

    def get_all_nodes(self):
        """Return all nodes associated with this set."""
        return list(self.nodes.keys())

    def copy(self):
        """Return a shallow copy of this object, the reference to the nodes
        will be the same, but the containers storing them will be copied."""
        return GeometrySetNodes(
            geometry_type=self.geometry_type,
            nodes=list(self.nodes.keys()),
        )


class GeometryName(dict):
    """Group node geometry sets together.

    This is mainly used for export from mesh functions. The sets can be
    accessed by a unique name. There is no distinction between different
    types of geometry, every name can only be used once -> use
    meaningful names.
    """

    def __setitem__(self, key, value):
        """Set a geometry set in this container."""

        if not isinstance(key, str):
            raise TypeError(f"Expected string, got {type(key)}!")
        if isinstance(value, GeometrySetBase):
            super().__setitem__(key, value)
        else:
            raise NotImplementedError("GeometryName can only store GeometrySets")


class GeometrySetContainer(_ContainerBase):
    """A class to group geometry sets together with the key being the geometry
    type."""

    def __init__(self, *args, **kwargs):
        """Initialize the container and create the default keys in the map."""
        super().__init__(*args, **kwargs)

        self.item_types = [GeometrySetBase]

        for geometry_key in _mpy.geo:
            self[geometry_key] = []

    def copy(self):
        """When creating a copy of this object, all lists in this object will
        be copied also."""

        # Create a new geometry set container.
        copy = GeometrySetContainer()

        # Add a copy of every list from this container to the new one.
        for geometry_key in _mpy.geo:
            copy[geometry_key] = self[geometry_key].copy()

        return copy
