# python modules
import numpy as np

# meshpy modules
from . import BaseMeshItem


class Node(BaseMeshItem):
    """
    This object represents one node in the mesh. The node can have a rotation
    and can be rotated moved and so on.
    """
    
    def __init__(self, coordinates, rotation=None, is_middle_node=False):
        BaseMeshItem.__init__(self, data=None, is_dat=False, is_referenced=False)
        
        # Coordinates and rotation of this node.
        self.coordinates = np.array(coordinates, dtype=np.float)
        self.rotation = rotation
        
        # If this node is at the end of a line or curve created with multiple
        # elements.
        self.is_end_node = False
        self.is_middle_node = is_middle_node

    
    def rotate(self, rotation, origin=None, only_rotate_triads=False):
        """
        Rotate this node. By default the node is rotated around the origin
        (0,0,0), if the keyword argument origin is given, it is rotated around
        that point.
        If only_rotate_triads is True, then only the rotation is affected,
        the position of the node stays the same.
        """
        
        # If the node has a rotation, rotate it.
        if self.rotation:
            self.rotation = rotation * self.rotation
        
        # Rotate the positions (around origin).
        if not only_rotate_triads:
            if not origin is None:
                self.coordinates -= origin
            self.coordinates = rotation * self.coordinates
            if not origin is None:
                self.coordinates += origin


    def get_dat_line(self):
        """
        Return the line that represents this node in the input file.
        """
        
        return 'NODE {} COORD {} {} {}'.format(
            self.n_global,
            self.coordinates[0],
            self.coordinates[1],
            self.coordinates[2]
            )