# -*- coding: utf-8 -*-
"""
This module implements the class that represents one element in the Mesh.
"""

# Meshpy modules.
from . import BaseMeshItem


class Element(BaseMeshItem):
    """A base class for an FEM element in the mesh."""

    def __init__(self, nodes=None, material=None, is_dat=False, **kwargs):
        BaseMeshItem.__init__(self, data=None, is_dat=is_dat, **kwargs)

        # List of nodes that are connected to the element.
        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes

        # Material of this element.
        self.material = material

    @classmethod
    def from_dat(cls, input_line):
        """Check the string to decide which element to use."""

        # Import solid element classes for creation of the element.
        from . import SolidHEX8, SolidRigidSphere

        # Split up input line and get pre node string.
        line_split = input_line[0].split()
        dat_pre_nodes = ' '.join(line_split[1:3])

        # Get a list of the element nodes.
        element_nodes = []
        for i, item in enumerate(line_split[3:]):
            if item.isdigit():
                element_nodes.append(int(item) - 1)
            else:
                break
        else:
            raise ValueError(('The input line:\n"{}"\ncould not be converted '
                + 'to a solid element!').format(input_line))

        # Get the post node string
        dat_post_nodes = ' '.join(line_split[3 + i:])

        # Depending on the number of nodes chose which solid element to return.
        if len(element_nodes) == 8:
            return SolidHEX8(nodes=element_nodes, dat_pre_nodes=dat_pre_nodes,
                dat_post_nodes=dat_post_nodes, comments=input_line[1])
        elif len(element_nodes) == 1:
            return SolidRigidSphere(nodes=element_nodes,
                dat_pre_nodes=dat_pre_nodes, dat_post_nodes=dat_post_nodes)
        else:
            raise TypeError('Could not find a element type for {}'.format(
                dat_pre_nodes))
