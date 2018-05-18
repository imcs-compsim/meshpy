
# python packages
import numpy as np

# meshgen imports
from meshgen.rotation import Rotation
from meshgen.utility import get_section_string





# 
# class Coupling(object):
#     """
#     Represents a coupling between dof in BACI.
#     """
#     
#     def __init__(self, nodes, coupling_string):
#         
#         # flatten out nodes
#         self.nodes = []
#         self._add_nodes(nodes)
#         self.coupling_string = coupling_string
#         self.node_set = None
#     
#     def _add_nodes(self, nodes):
#         # check type
#         if type(nodes) == list:
#             for node in nodes:
#                 self._add_nodes(node)
#         elif type(nodes) == Node:
#             self.nodes.append(nodes)
#         elif type(nodes) == GeometrySet:
#             self.nodes.extend(nodes.nodes)
#         else:
#             print('Error! not node or list')
#     
#     def add_set_to_geometry(self, geometry):
#         # add set to global sets
#         self.node_set = GeometrySet('', self.nodes)
#         geometry.point_sets.append(self.node_set)
#     
#     def get_dat_line(self):
#         return 'E {} - {}'.format(self.node_set.n_global, self.coupling_string)
# 
# 
# 
# 
# class GeometrySet(object):
#     """
#     Represents a set of nodes, for points or lines
#     """
#     
#     def __init__(self, name, nodes=None):
#         """
#         Define the type of the set
#         """
#         
#         self.name = name
#         self.nodes = []
#         if nodes:
#             self.add_node(nodes)
#         self.n_global = None
#     
#     
#     def add_node(self, add):
#         """
#         Check if the object or list of objects given is a node and add it to self.
#         """
#         
#         if type(add) == Node:
#             if not add in self.nodes:
#                 self.nodes.append(add)
#         elif type(add) == list:
#             for item in add:
#                 self.add_node(item)
#         else:
#             print('ERROR only Nodes and list of Nodes can be added to this object.')



class BaseMeshItem(object):
    """
    A base class for nodes, elements, sets and so on that are given in dat files.
    """
    
    def __init__(self, dat_string):
        """ The defualt case is just set by a string from a dat file. """
        
        self.dat_string = dat_string
        self.is_dat = True
        self.n_global = None
    
    
    def get_dat_line(self):
        """
        By default return the dat string.
        Otherwise call a function that is defined in the sub classes.
        """
        
        return self.dat_string


class Node(object):
    """ A class that represents one node of the mesh in the input file. """
    
    
    def __init__(self, coordinates, rotation=None):
        """
        Each node has a position and an optional rotation object.
        The n_global value is set when the model is writen to a dat file.
        """
        
        self.coordinates = coordinates
        self.rotation = rotation
        self.n_global = None
        self.is_dat = False

    
    def rotate(self, rotation, only_rotate_triads=False):
        """
        Rotate the node.
        Default values is that the nodes is rotated around the origin.
        If only_rotate_triads is True, then only the triads are rotated,
        the position of the node stays the same.
        """
        
        # do not do anything if the node does not have a rotation
        if not self.rotation:
            # apply the roation to the triads
            self.rotation = rotation * self.rotation
            
            # rotate the positions
            if not only_rotate_triads:
                self.coordinates = rotation * self.coordinates


    def get_dat_line(self):
        """ Return the line for the dat file for this element. """
        
        return 'NODE {} COORD {} {} {}'.format(
            self.n_global,
            self.coordinates[0],
            self.coordinates[1],
            self.coordinates[2]
            )


class Beam(object):
    """ A base class for a beam element. """
    
    def __init__(self, material, element_name=None, node_create=None):
        """
        Set the base values for a beam element.
        node_create is described in self.create_beam
        """
        
        # name that will be displayed in the dat file
        self.element_name = element_name
        
        self.nodes = []
        self.material = material
        self.node_create = node_create
        
        self.n_global = None
        self.is_dat = False
        
    
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
    """ Represents a BEAM3R HERM2LIN3 element. """
    
    def __init__(self, material):
        """ Set the data for this beam element. """
        
        node_create = [
            [-1, 0, True],
            [0, 2, True],
            [1, 1, True]
            ]

        Beam.__init__(self, material,
                      element_name='BEAM3R HERM2LIN3',
                      node_create=node_create
                      )
    
    
    def get_dat_line(self):
        """ Return the line for the dat file for this element. """
        
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


class Mesh(object):
    """ Holds nodes, beams and couplings of beam_mesh geometry. """
    
    def __init__(self, name=None):
        """ Set empty parameters """
        
        self.name = name
        self.nodes = []
        self.elements = []
        self.couplings = []
        self.point_sets = []
        self.line_sets = []
        self.surf_sets = []
        self.vol_sets = []
    
    
    def add_mesh(self, mesh, add_sets=True):
        """ Add other mesh to this one. """
        
        self.nodes.extend(mesh.nodes)
        self.elements.extend(mesh.elements)
         
    # TODO
#         if add_sets:
#             self.point_sets.extend(beam_mesh.point_sets)
#             self.line_sets.extend(beam_mesh.line_sets)
#             self.mesh_sets.extend(beam_mesh.mesh_sets)
#             self.vol_sets.extend(beam_mesh.vol_sets)
    
    
    def translate(self, vector):
        """ Move all nodes of this mesh by the vector. """
        
        for node in self.nodes:
            if not node.is_dat:
                node.coordinates += vector
    
    
    def rotate(self, rotation, origin=None):
        """ Rotate the geometry about the origin. """
        
        # move structure to rotation origin
        if origin:
            self.translate(-np.array(origin))
        
        # rotate structure
        for node in self.nodes:
            if not node.is_dat:
                node.rotate(rotation, only_rotate_triads=False)
        
        # move origin back to initial place
        if origin:
            self.translate(np.array(origin))
    
    
    def wrap_around_cylinder(self):
        """
        Wrap the geometry around a cylinder.
        The y-z plane morphs into the roation axis.
        There should be NO points with negative x coordinates.
        """
        
        # check the y coordiantes
        for node in self.nodes:
            if not node.is_dat:
                if node.coordinates[0] < 0:
                    print('ERROR, there should be no points with negative x coordiantes')
        
        # transform the nodes
        for node in self.nodes:
            if not node.is_dat:
                # get the cylindercoordinates
                r = np.dot([1,0,0], node.coordinates)
                phi = node.coordinates[1] / r
    
                # first apply the rotations
                node.rotate(Rotation([0,0,1], phi), only_rotate_triads=True)
                
                # set the new coordinates
                node.coordinates = [
                    r * np.cos(phi),
                    r * np.sin(phi),
                    node.coordinates[2]
                    ]
    
    # TODO
#     def add_coupling(self, nodes, tmp):
#         """
#         Add a coupling to the mesh
#         """
#         
#         self.couplings.append(Coupling(nodes, tmp))
#     
#     
    def add_beam_mesh_line(self, beam_object, material, start_point, end_point, n, add_sets=True, add_first_node=True):
        """
        A straight line of beam elements.
            n: Number of elements along line
        """
        
        # direction vector of line
        direction = np.array(end_point) - np.array(start_point)
        
        # rotation for this line (is constant on the whole line)
        t1 = direction
        # check if the x or y axis are larger projected onto the direction
        if np.dot(t1,[1,0,0]) < np.dot(t1,[0,1,0]):
            t2 = [1,0,0]
        else:
            t2 = [0,1,0]
        rotation = Rotation.from_basis(t1, t2)
        
        # this function returns the position and the triads for each element
        def get_beam_function(point_a, point_b):
            
            def position_function(xi):
                return 1/2*(1-xi)*point_a + 1/2*(1+xi)*point_b
            
            def rotation_function(xi):
                return rotation
            
            return (
                position_function,
                rotation_function
                )
        
        # TODO
#         save the old number of nodes
#         node_start = len(self.nodes)
        
        # create the beams
        for i in range(n):
            
            functions = get_beam_function(
                start_point + i*direction/n,
                start_point + (i+1)*direction/n
                )
            
            tmp_beam = beam_object(material)
            if add_first_node and i == 0:
                tmp_beam.create_beam(self.nodes, functions[0], functions[1], create_first=True)
            else:
                tmp_beam.create_beam(self.nodes, functions[0], functions[1], create_first=False)
            self.elements.append(tmp_beam)
        
        # TODO
#         # add nodes to set
#         point_sets = [
#             GeometrySet('line_point_start', self.nodes[node_start]),
#             GeometrySet('line_point_end', self.nodes[-1])
#             ]
#         line_sets = [
#             GeometrySet('line_line', [self.nodes[i] for i in range(node_start,len(self.nodes))])
#             ]
#         if add_sets:
#             self.point_sets.extend(point_sets)
#             self.line_sets.extend(line_sets)
#         
#         return point_sets, line_sets, [], []            



class MeshInput(Mesh):
    """
    This is just a BeamMesh class that can additionally manage sections for the input file.
    """
    
    def _add_dat_section(self, section_name, section_data):
        """
        Check if the section has to be added to the mesh of if it is just a
        basic input section.
        """
        
        if section_name == 'MATERIALS':
            pass
        elif section_name == 'DESIGN LINE DIRICH CONDITIONS':
            pass
        elif section_name == 'DESIGN SURF DIRICH CONDITIONS':
            pass
        elif section_name == 'DNODE-NODE TOPOLOGY':
            pass
        elif section_name == 'DLINE-NODE TOPOLOGY':
            pass
        elif section_name == 'DSURF-NODE TOPOLOGY':
            pass
        elif section_name == 'DVOL-NODE TOPOLOGY':
            pass
        elif section_name == 'NODE COORDS':
            for line in section_data:
                self.nodes.append(BaseMeshItem(line))
        elif section_name == 'STRUCTURE ELEMENTS':
            for line in section_data:
                self.elements.append(BaseMeshItem(line))
        elif section_name == 'DESIGN DESCRIPTION':
            pass
        else:
            # descion is not in mesh
            return 1
        
        # section is in mesh
        return 0
        
        
    def get_dat_lines(self):
        """
        Get the lines for the input file that contain the information for
        the mesh.
        """
        
        # first all nodes, elements, sets and couplings are assigned a global value
        for i, node in enumerate(self.nodes):
            node.n_global = i + 1
        for i, element in enumerate(self.elements):
            element.n_global = i + 1
            
        
        lines = []
        
        # add the nodal data
        lines.append(get_section_string('NODE COORDS'))
        for node in self.nodes:
            lines.append(node.get_dat_line())

        # add the element data
        lines.append(get_section_string('STRUCTURE ELEMENTS'))
        for element in self.elements:
            lines.append(element.get_dat_line())
        
        return lines
        
        
        
        
        
        
        
        
        
        
        
        
        
        