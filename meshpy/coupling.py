
from . import flatten, Node
 
class Coupling(object):
    """
    Represents a coupling between dof in BACI.
    """
     
    def __init__(self, nodes, coupling_type, name=None):
         
        # flatten out nodes
        self.nodes = []
        self._add_nodes(flatten(nodes))
        self.name = name
        self.coupling_type = coupling_type
        self.node_set = None
     
    def _add_nodes(self, nodes):
        # check type
        if type(nodes) == list:
            for node in nodes:
                self._add_nodes(node)
        elif type(nodes) == Node:
            self.nodes.append(nodes)
        elif type(nodes) == GeometrySet:
            self.nodes.extend(nodes.nodes)
        else:
            print('Error! not node or list')
     
    def get_dat_line(self):
        if self.coupling_type == 'joint':
            string = 'NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0'
        elif self.coupling_type == 'fix':
            string = 'NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0' 
        return 'E {} - {}'.format(self.node_set.n_global, string)

