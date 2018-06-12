
# import meshpy modules
from . import BaseMeshItem, Node, mpy


class NodeSet(BaseMeshItem):
    """
    To group nodes together. The variable referenced_type holds a boolean value
    if the item is referenced to a certain geometry.
    """
    
    def __init__(self, geo_type, nodes=None):
        """
        Initialize object. It is possible to give nodes at this point.
        """
        BaseMeshItem.__init__(self, is_dat=None, is_referenced=False)
        self.geo_type = geo_type
        self.nodes = []
        
        if not nodes is None:
            self.add(nodes)
        
        # node set names
        self.geo_set_names = {
            mpy.point: 'DNODE',
            mpy.line: 'DLINE',
            mpy.surface: 'DSURFACE',
            mpy.volume: 'DVOLUME'
            }
        
    def add(self, value):
        """ Add Nodes. """
        
        if isinstance(value, list):
            for item in value:
                self.add(item)
        elif isinstance(value, Node):
            if not value in self.nodes:
                self.nodes.append(value)
            else:
                raise ValueError('The node already exists in this set!')
        else:
            raise TypeError('Expected Node or list, but got {}'.format(
                type(value)
                ))
    
    def __iter__(self):
        for node in self.nodes:
            yield node
    
    def get_dat_lines(self):
        """ Print the data stuff. """
        return ['NODE {} {} {}'.format(node.n_global, self.geo_set_names[self.geo_type], self.n_global) for node in self.nodes]



class NodeSetContainer(dict):
    """ Group node sets together. Mainly for export from mesh functions. """
    
    def __setitem__(self, key, value):
        """ Set nodes to item. """
        
        if not isinstance(key, str):
            raise TypeError('Expected string, got {}!'.format(type(key)))
        elif isinstance(value, NodeSet):
            dict.__setitem__(self, key, value)
        elif isinstance(value, list):
            dict.__setitem__(self, key, NodeSet(nodes=value))
        elif isinstance(value, Node):
            dict.__setitem__(self, key, NodeSet(nodes=[value]))
        else:
            raise ValueError('Not expected {}'.format(type(value)))




















