

from . import BaseMeshItem


class BC(BaseMeshItem):
    """ This object is one BC. """
    
    def __init__(self, geometry_set, bc_string, format_replacement=None, bc_type=None):
        """
        Set the default values. Format_replacement will be called on string.
        """
        
        BaseMeshItem.__init__(self, is_dat=False)
        self.bc_string = bc_string
        self.bc_type = bc_type
        self.format_replacement = format_replacement  
        self.geometry_set = geometry_set
            
    
    def get_dat_line(self):
        """ Line in the input file for the BC. """
        
        if self.format_replacement:
            dat_string = self.bc_string.format(*self.format_replacement)
        else:
            dat_string = self.bc_string
        
        return 'E {} - {}'.format(
            self.geometry_set.n_global,
            dat_string
            )