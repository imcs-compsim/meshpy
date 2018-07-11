# -*- coding: utf-8 -*-
"""
This module implements beam elements for the mesh.
"""

# Python modules.
import numpy as np

# Meshpy modules.
from . import mpy, Element, Node


class Beam(Element):
    """A base class for a beam element."""

    # A array that defines the creation of the nodes for this beam:
    #    self.nodes_create = [
    #        [ xi, create_rotation, is_middle_node ], # fist node
    #        ...
    #        ]    
    nodes_create = []
    
    def __init__(self, material=None, nodes=None):
        Element.__init__(self, nodes=nodes, material=material)
    
    def create_beam(self, beam_function, start_node=None):
        """
        Create the nodes for this beam element. The function returns a list with
        the created nodes.
        
        Args
        ----
        beam_function: function(xi)
            Returns the position and rotation of the beam along the local
            coordinate xi.
        start_node: Node
            If this argument is given, this is the node of the beam at xi=-1.
        """
        
        if len(self.nodes) > 0:
            raise ValueError('The beam should not have any local nodes yet!')
        
        if not start_node is None:
            pos, rot = beam_function(-1)
            if np.linalg.norm(pos
                    - start_node.coordinates) > mpy.eps_pos:
                raise ValueError('Start node does not match with function!')
            if not start_node.rotation == rot:
                raise ValueError('Start rotation does not match with function!')
            self.nodes = [start_node]
        
        # Loop over local nodes.
        for i, [xi, create_rot, middle_node] in enumerate(self.nodes_create):
            if i > 0 or start_node is None:
                pos, rot = beam_function(xi)
                if create_rot:
                    self.nodes.append(Node(
                        pos,
                        rotation=rot,
                        is_middle_node=middle_node
                        ))
                else:
                    self.nodes.append(Node(
                        pos,
                        rotation=None,
                        is_middle_node=middle_node
                        ))
        
        # Return the created nodes.
        if start_node is None:
            return self.nodes
        else:
            return self.nodes[1:]
    
    
    def preview_python(self, ax):
        """Plot the beam in matplotlib, by connecting the nodes."""
        
        coordinates = np.array([node.coordinates for node in self.nodes])
        ax.plot(coordinates[:,0], coordinates[:,1], coordinates[:,2], '-x')
    
    def get_vtk(self, vtk_writer):
        """
        Add the representation of this element to the VTK writer as a poly line.
        """
        
        # Dictionary with cell data.
        cell_data = {}
        cell_data['cross_section_radius'] = self.material.diameter / 2
        
        # Dictionary with point data.
        point_data = {}
        point_data['node_value'] = []
        
        # Array with nodal coordinates.
        coordinates = np.zeros([len(self.nodes), 3])
        for i, node in enumerate(self.nodes):
            xi = self.nodes_create[i][0]
            coordinates[i, :] = node.coordinates
            if xi == -1 or xi == 1:
                point_data['node_value'].append(1.)
            elif xi == 0:
                point_data['node_value'].append(0.5)
            else:
                point_data['node_value'].append(0)
                
        # Add poly line to writer.
        vtk_writer.add_poly_line(coordinates, cell_data=cell_data,
            point_data=point_data)


class Beam3rHerm2Lin3(Beam):
    """Represents a BEAM3R HERM2LIN3 element."""
    
    nodes_create = [
        [-1, True, False],
        [0, True, True],
        [1, True, False]
        ]
    
    def _get_dat(self):
        """ Return the line for the input file. """
        
        string_nodes = ''
        string_triads = ''
        for i in [0,2,1]:
            node = self.nodes[i]
            string_nodes += '{} '.format(node.n_global)
            string_triads += ' ' + node.rotation.get_dat()
        
        return '{} BEAM3R HERM2LIN3 {}MAT {} TRIADS{}'.format(
            self.n_global,
            string_nodes,
            self.material.n_global,
            string_triads
            )
