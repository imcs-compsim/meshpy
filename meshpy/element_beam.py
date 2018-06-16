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
    
    def create_beam(self, position_function, rotation_function=None,
            start_node=None):
        """
        Create the nodes for this beam element. The function returns a list with
        the created nodes.
        
        Args
        ----
        position_function: function(xi)
            Returns the position of the beam along the local coordinate xi
        rotation_function: function(xi)
            Returns the rotation of the beam along the local coordinate xi
        start_node: Node
            If this argument is given, this is the node of the beam at xi=-1
        """
        
        if len(self.nodes) > 0:
            raise ValueError('The beam should not have any local nodes yet!')
        
        if not start_node is None:
            if np.linalg.norm(position_function(-1)
                    - start_node.coordinates) > mpy.eps_pos:
                raise ValueError('Start node does not match with function!')
            if not start_node.rotation == rotation_function(-1):
                raise ValueError('Start rotation does not match with function!')
            self.nodes = [start_node]
        
        # Loop over local nodes.
        for i, [xi, create_rot, middle_node] in enumerate(self.nodes_create):
            if i > 0 or start_node is None:
                if create_rot:
                    rotation = rotation_function(xi)
                else:
                    rotation = None
                self.nodes.append(Node(
                    position_function(xi),
                    rotation=rotation,
                    is_middle_node=middle_node
                    ))
        
        # Return the created nodes.
        if start_node is None:
            return self.nodes
        else:
            return self.nodes[1:]
    


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
