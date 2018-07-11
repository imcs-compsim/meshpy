# -*- coding: utf-8 -*-
"""
This module implements a basic class to manage geometry in the input file.
"""

# Meshpy modules.
from . import mpy, BaseMeshItem, Node


class GeometrySet(BaseMeshItem):
    """This object represents a geometry set. The set is defined by nodes."""

    # Node set names for the input file file.
    geometry_set_names = {
        mpy.point: 'DNODE',
        mpy.line: 'DLINE',
        mpy.surface: 'DSURFACE',
        mpy.volume: 'DVOLUME'
        }

    def __init__(self, geometry_type, nodes=None):
        BaseMeshItem.__init__(self, is_dat=None)

        self.geometry_type = geometry_type
        self.nodes = []

        if nodes is not None:
            self.add(nodes)

    def add(self, value):
        """
        Add nodes to this object.

        Args
        ----
        value: Node, list(Nodes)
            Node(s) to be added to this geometry
        """

        if isinstance(value, list):
            for item in value:
                self.add(item)
        elif isinstance(value, Node):
            if value not in self.nodes:
                self.nodes.append(value)
            else:
                raise ValueError('The node already exists in this set!')
        else:
            raise TypeError('Expected Node or list, but got {}'.format(
                type(value)
                ))

    def __iter__(self):
        for node in self.nodes:
            yield node

    def _get_dat(self):
        """Get the lines for the input file."""
        return ['NODE {} {} {}'.format(
            node.n_global,
            self.geometry_set_names[self.geometry_type], self.n_global
            ) for node in self.nodes]
