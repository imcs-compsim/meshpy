# python modules
import numpy as np

# meshpy modules
from . import Element
from . import Node



class Beam(Element):
    """
    A base class for a beam element. Derived from the base element.
    """
    
    def __init__(self, nodes_create=None, material=None):
        Element.__init__(self, nodes=None, material=material)
        
        # A array that defines the creation of the nodes for this beam along a
        # line. Details are in the docstring for create_beam.
        self.node_create = nodes_create
        
    
    def create_beam(self,
                    position_function,
                    rotation_function=None,
                    start_node=None
                    ):
        """
        Create the nodes for this beam element.
        If a start_node is given this will be the first node for this beam
        (therefore the first node should have the local coordinate xi = -1)
        
        The functions position_function and rotation_function are called to give
        the position and rotation along the beam in the local coordinates xi.
        
        Creation is based on the self.node_create parameter:
            self.node_create = [
                [ xi, create_rotation ], # fist node
                ...
                ] 
        """
        
        if len(self.nodes) > 0:
            raise ValueError('The beam should not have any local nodes yet!')
        
        self.nodes = [None for i in range(len(self.node_create))]
        
        if not start_node is None:
            
            # check if start node has the same position and rotation as the
            # given functions
            if np.linalg.norm(position_function(-1)-start_node.coordinates) > 1e-10:
                print(position_function(-1))
                print(start_node.coordinates)
                raise ValueError('Coordinates of function and start nodes do not match!')
            
            # TODO: check rotation
            self.nodes[0] = start_node
        
        # Loop over local nodes.
        for i, [xi, create_rotation, middle_node] in enumerate(self.node_create):
            if i > 0 or start_node is None:
                if create_rotation:
                    rotation = rotation_function(xi)
                else:
                    rotation = None
                self.nodes[i] = Node(
                    position_function(xi),
                    rotation=rotation,
                    is_middle_node=middle_node
                    )
        
        if start_node is None:
            return self.nodes
        else:
            return self.nodes[1:]
    


class Beam3rHerm2Lin3(Beam):
    """ Represents a BEAM3R HERM2LIN3 element. """
    
    def __init__(self, material=None):
        
        nodes_create = [
            [-1, True, False],
            [0, True, True],
            [1, True, False]
            ]
        Beam.__init__(self, nodes_create=nodes_create, material=material)    
    
    
    def get_dat_line(self):
        """ Return the line for the input file. """
        
        string_nodes = ''
        string_triads = ''
        for i in [0,2,1]:
            node = self.nodes[i]
            string_nodes += '{} '.format(node.n_global)
            string_triads += node.rotation.get_dat()
        
        return '{} BEAM3R HERM2LIN3 {}MAT {} TRIADS{}'.format(
            self.n_global,
            string_nodes,
            self.material.n_global,
            string_triads
            )
