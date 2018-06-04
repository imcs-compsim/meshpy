

class BaseMeshItem(object):
    """
    Base class for all objects that are related to a mesh.
    """
    
    def __init__(self, data=None, is_dat=True):
        
        # Data for this object. Should be string or list of strings.
        self.data = data
        
        # If this object is imported from a .dat file.
        self.is_dat = is_dat
        
        # Global number of this object. Relevant for derived classes for
        # generation of global input file.
        self.n_global = None
        
        # If the item is referenced by another item. For example sets and BCs.
        self.is_referenced = False
    
    
    def get_dat_line(self):
        """ By default return the data string. """
        if isinstance(self.data, str):
            return self.data
        else:
            raise TypeError('Expected string, got {}!'.format(type(self.data)))
    
    
    def get_dat_lines(self):
        """ By default return the data list. """
        if isinstance(self.data, list):
            return self.data
        else:
            raise TypeError('Expected list, got {}!'.format(type(self.data)))
