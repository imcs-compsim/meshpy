# -*- coding: utf-8 -*-
"""
This module implements solid elements for the mesh.
"""

# Python modules.
import numpy as np

# Meshpy modules.
from . import Element, add_point_data_node_sets


class SolidElement(Element):
    """A base class for a solid element."""

    def __init__(self, nodes=None, dat_pre_nodes='', dat_post_nodes='',
            **kwargs):
        Element.__init__(self, nodes=nodes, material=None, is_dat=True,
            **kwargs)
        self.dat_pre_nodes = dat_pre_nodes
        self.dat_post_nodes = dat_post_nodes

    def _get_dat(self, **kwargs):
        """Return the dat line for this element."""

        # String with the node ids.
        nodes_string = ''
        for node in self.nodes:
            nodes_string += '{} '.format(node.n_global)

        # Return the dat line.
        return '{} {} {} {}'.format(
            self.n_global,
            self.dat_pre_nodes,
            nodes_string,
            self.dat_post_nodes
            )


class SolidHEX8(SolidElement):
    """A HEX8 solid element."""

    def get_vtk(self, vtkwriter_beam, vtkwriter_solid):
        """
        Add the representation of this element to the VTK writer as a quad.
        """

        # Dictionary with cell data.
        cell_data = {}

        # Dictionary with point data.
        point_data = {}

        # Array with nodal coordinates.
        coordinates = np.zeros([len(self.nodes), 3])
        for i, node in enumerate(self.nodes):
            coordinates[i, :] = node.coordinates

        # Add the node sets connected to this element.
        add_point_data_node_sets(point_data, self.nodes)

        # Add hex8 line to writer.
        vtkwriter_solid.add_hex8(coordinates, cell_data=cell_data,
            point_data=point_data)


class SolidRigidSphere(SolidElement):
    """A rigid sphere solid element."""
    pass
