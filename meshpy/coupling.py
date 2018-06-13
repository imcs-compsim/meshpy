
from . import Node, NodeSet, mpy, BaseMeshItem
 
class Coupling(BaseMeshItem):
    """
    Represents a coupling between dof in BACI.
    """
     
    def __init__(self, nodes, coupling_type):
        BaseMeshItem.__init__(self, is_dat=False)
        self.node_set = NodeSet(mpy.point, nodes=nodes) 
        self.coupling_type = coupling_type

    
    def _get_dat(self):
        if self.coupling_type == 'joint':
            string = 'NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0'
        elif self.coupling_type == 'fix':
            string = 'NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0' 
        return 'E {} - {}'.format(self.node_set.n_global, string)

