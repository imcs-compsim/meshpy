class Coupling(object):
    """
    Represents a coupling between dof in BACI.
    """
    
    pass


class GeometrySet(object):
    """
    Represents a set of nodes, for points or lines
    """
    
    def __init__(self, name, nodes=None):
        """
        Define the type of the set
        """
        
        self.name = [name]
        self.nodes = []
        if nodes:
            self.add_node(nodes)
    
    
    def add_node(self, add):
        """
        Check if the object or list of objects given is a node and add it to self.
        """
        
        if type(add) == Node:
            self.nodes.append(add)
        elif type(add) == list:
            for item in add:
                self.add_node(item)
        else:
            print('ERROR only Nodes and list of Nodes can be added to this object.')


class Node(object):
    """
    A class that represents one node in the simulation
    """
    
    def __init__(self, coordinates, rotation=None):
        """
        Each node has a position and an optional rotation object.
        The n_global value is set when the model is writen to a dat file.
        """
        
        self.coordinates = coordinates
        self.rotation = rotation
        self.n_global = None
    
    
    def rotate(self, rotation, only_rotate_triads=False):
        """
        Rotate the node.
        Default values is that the nodes is rotated around the origin.
        If only_rotate_triads is True, then only the triads are rotated,
        the position of the node stays the same.
        """
        
        # apply the roation to the triads
        self.rotation = rotation * self.rotation
        
        # rotate the positions
        if not only_rotate_triads:
            self.coordinates = rotation * self.coordinates


    def get_dat_line(self):
        """
        Return the line for the dat file for this element.
        """
        
        return 'NODE {} COORD {} {} {}'.format(
            self.n_global,
            self.coordinates[0],
            self.coordinates[1],
            self.coordinates[2]
            )


class Beam(object):
    """
    TODO
    """
    
    def __init__(self, material,
                 nodes=None,
                 element_name=None,
                 node_create=None
                 ):
        """
        TODO
        """
        
        # name that will be displayed in the dat file
        self.element_name = element_name
        
        # list with nodes from this element
        if not nodes:
            self.nodes = []
        else:
            self.nodes = nodes
        
        # material of this beam
        self.material = material
        
        # default node creation rules
        self.node_create = node_create
        
        # global element number
        self.n_global = None
        
    
    def create_beam(self, nodes, position_function, rotation_function, create_first=False):
        """
        Create the nodes and links for the beam element.
        If the node list is empty or create_first is true, the first node is created.
        Otherwise the last node of nodes is taken as the first node of this element.
        
        The functions position_function and rotation_function are called to give the
        position and rotation along the beam in the local coordinates xi.
        
        Creation is based on the self.node_create parameter:
            self.node_create = [
                [ xi, local_index, create_rotation ], # fist node
                ...
                ] 
        """
        
        # create local node list
        self.nodes = [None for i in range(len(self.node_create))]
        
        # loop over nodes
        for i, [xi, local_index, create_rotation] in enumerate(self.node_create):
            # if there is no node in nodes the first one is created no matter what
            # create_first is
            if i > 0 or (len(nodes) == 0 or create_first):
                if create_rotation:
                    rotation = rotation_function(xi)
                else:
                    rotation = None
                tmp_node = Node(position_function(xi), rotation=rotation)
                
                # add to global node list
                nodes.append(tmp_node)
                
                # add to local node list
                self.nodes[local_index] = tmp_node
            else:
                self.nodes[self.node_create[0][1]] = nodes[-1]


class Beam3rHerm2Lin3(Beam):
    """
    Represents a BEAM3R HERM2LIN3 element.
    """
    
    def __init__(self, material, nodes=None):
        """
        Set the data for this beam element
        """
        
        node_create = [
            [-1, 0, True],
            [0, 2, True],
            [1, 1, True]
            ]

        Beam.__init__(self, material,
                      nodes=nodes,
                      element_name='BEAM3R HERM2LIN3',
                      node_create=node_create
                      )
    
    
    def get_dat_line(self):
        """
        return the line for the dat file for this element
        """
        
        string_nodes = ''

        for node in self.nodes:
            string_nodes += '{} '.format(node.n_global)
        
        string_triads = ''
        for node in self.nodes:
            string_triads += node.rotation.get_dat()
        
        return '{} {} {}MAT {} TRIADS{} FAD'.format(
            self.n_global,
            self.element_name,
            string_nodes,
            1,
            string_triads
            )
