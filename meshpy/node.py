# import python modules
import numpy as np

class Node(object):
    """ A class that represents one node of the mesh in the input file. """
    
    
    def __init__(self, coordinates, rotation=None):
        """
        Each node has a position and an optional rotation object.
        The n_global value is set when the model is writen to a dat file.
        """
        
        self.coordinates = np.array(coordinates)
        self.rotation = rotation
        self.n_global = None
        self.is_dat = False
        self.connected_elements = []
        self.connected_couplings = []
        # for the end nodes of a line
        self.is_end_node = False

    
    def rotate(self, rotation, only_rotate_triads=False):
        """
        Rotate the node.
        Default values is that the nodes is rotated around the origin.
        If only_rotate_triads is True, then only the triads are rotated,
        the position of the node stays the same.
        """
        
        # do not do anything if the node does not have a rotation
        if self.rotation:
            # apply the rotation to the triads
            self.rotation = rotation * self.rotation
        
        # rotate the positions (around origin)
        if not only_rotate_triads:
            self.coordinates = rotation * self.coordinates


    def get_dat_line(self):
        """ Return the line for the dat file for this element. """
        
        return 'NODE {} COORD {} {} {}'.format(
            self.n_global,
            self.coordinates[0],
            self.coordinates[1],
            self.coordinates[2]
            )
