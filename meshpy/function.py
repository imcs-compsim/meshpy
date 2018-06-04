# import meshpy modules
from . import BaseMeshItem


class Function(BaseMeshItem):
    """ Holds information for a function. """
    
    def __init__(self, data):
        BaseMeshItem.__init__(self, data=data, is_dat=False)
    
    
    def __str__(self):
        """ Check if the function has a global index. """
        if self.n_global:
            return str(self.n_global)
        else:
            raise IndexError('The function does not have a global index!')
