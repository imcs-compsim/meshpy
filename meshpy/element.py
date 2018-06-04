from . import BaseMeshItem

class Element(BaseMeshItem):
    """
    A base class for an FEM element in the mesh.
    """
    
    def __init__(self, nodes=None, material=None):
        BaseMeshItem.__init__(self, data=None, is_dat=False)

        # List of nodes that are connected to the element.
        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes 
        
        # Material of this element.
        self.material = material