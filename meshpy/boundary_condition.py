


class BC(object):
    """ This object is one BC. """
    
    def __init__(self, set_item, bc_string, format_replacement=None, bc_type=None):
        """
        Set the default values. Format_replacement will be called on string.
        """
        
        self.bc_string = bc_string
        self.type = bc_type
        self.format_replacement = format_replacement
        
        if set_item.is_referenced:
            print('Error, each set can only have one BC!')        
        self.set = set_item
        self.set.is_referenced = True
        
        
        self.is_dat = False
        self.n_global = None
        self.is_referenced = False
    
    def get_dat_line(self):
        """ Line in the input file for the BC. """
        
        if self.format_replacement:
            dat_string = self.bc_string.format(*self.format_replacement)
        else:
            dat_string = self.bc_string
        
        return 'E {} - {}'.format(
            self.set.n_global,
            dat_string
            )