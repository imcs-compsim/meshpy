# -*- coding: utf-8 -*-
"""
This module implements the class that represents one node in the Mesh.
"""

# Python modules.
import numpy as np

# Meshpy modules.
from . import mpy, BaseMeshItem


class Node(BaseMeshItem):
    """
    This object represents one node in the mesh. The node can have a rotation
    and can be rotated moved and so on.
    """

    def __init__(self, coordinates, rotation=None, is_middle_node=False,
                is_dat=False):
        BaseMeshItem.__init__(self, data=None, is_dat=is_dat)

        # Coordinates and rotation of this node.
        self.coordinates = np.array(coordinates)
        if rotation is None:
            self.rotation = None
        else:
            self.rotation = rotation.copy()

        # If this node is at the end of a line or curve (by default only those
        # nodes are checked for overlapping nodes).
        self.is_end_node = False

        # If the node is in the middle of a beam element.
        self.is_middle_node = is_middle_node

        # Lists with the node sets that are connected to this node.
        self.node_sets_link = []

    @classmethod
    def from_dat(cls, input_line):
        """Create the Node object from a line in the input file."""

        # Split up the input line.
        line_split = input_line[0].split()

        # Convert the node coordinates into a Node object.
        return cls([float(line_split[i]) for i in range(3, 6)], is_dat=True)

    def rotate(self, rotation, origin=None, only_rotate_triads=False):
        """
        Rotate this node. By default the node is rotated around the origin
        (0,0,0), if the keyword argument origin is given, it is rotated around
        that point.
        If only_rotate_triads is True, then only the rotation is affected,
        the position of the node stays the same.
        """

        # If the node has a rotation, rotate it.
        if self.rotation is not None:
            self.rotation = rotation * self.rotation

            # Rotate the positions (around origin).
            if not only_rotate_triads:
                if origin is not None:
                    self.coordinates = self.coordinates - origin
                self.coordinates = rotation * self.coordinates
                if origin is not None:
                    self.coordinates = self.coordinates + origin

    def _get_dat(self):
        """
        Return the line that represents this node in the input file.
        """
        node_string = 'NODE {} COORD ' + ' '.join(
            [mpy.dat_precision for i in range(3)])
        return node_string.format(
            self.n_global,
            self.coordinates[0],
            self.coordinates[1],
            self.coordinates[2]
            )
