# -*- coding: utf-8 -*-
"""
This module implements solid elements for the mesh.
"""

# Python modules.
import numpy as np
import vtk

# Meshpy modules.
from .element import Element
from .vtk_writer import add_point_data_node_sets


class SolidElement(Element):
    """A base class for a solid element."""

    # This class variables stores the information about the element shape in
    # vtk. And the connectivity to the nodes.
    vtk_cell_type = None
    vtk_topology = None

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

    def get_vtk(self, vtkwriter_beam, vtkwriter_solid):
        """
        Add the representation of this element to the VTK writer as a quad.
        """

        # Check that the element has a valid vtk cell type.
        if self.vtk_cell_type is None:
            raise TypeError('vtk_cell_type for {} not set!'.format(type(self)))

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
        vtkwriter_solid.add_cell(self.vtk_cell_type, coordinates,
            self.vtk_topology, cell_data=cell_data, point_data=point_data)


class SolidHEX8(SolidElement):
    """A HEX8 solid element."""
    vtk_cell_type = vtk.vtkHexahedron


class SolidTET4(SolidElement):
    """A TET4 solid element."""
    vtk_cell_type = vtk.vtkTetra


class SolidTET10(SolidElement):
    """A TET10 solid element."""
    vtk_cell_type = vtk.vtkQuadraticTetra


class SolidHEX20(SolidElement):
    """A HEX20 solid element."""
    vtk_cell_type = vtk.vtkQuadraticHexahedron
    vtk_topology = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 12,
        13, 14, 15]


class SolidHEX27(SolidElement):
    """A HEX27 solid element."""
    vtk_cell_type = vtk.vtkTriQuadraticHexahedron
    vtk_topology = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 12,
        13, 14, 15, 24, 22, 21, 23, 20, 25, 26]


class SolidRigidSphere(SolidElement):
    """A rigid sphere solid element."""

    def __init__(self, **kwargs):
        """Initialize solid sphere object."""
        SolidElement.__init__(self, **kwargs)

        # Set radius of sphere from input file.
        arg_name = self.dat_post_nodes.split()[0] 
        if not arg_name == 'RADIUS':
            raise ValueError('The first argument after the node should be '
                + 'RADIUS, but it is "{}"!'.format(arg_name))
        self.radius = float(self.dat_post_nodes.split()[1])
