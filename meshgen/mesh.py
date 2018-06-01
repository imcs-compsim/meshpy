
# python packages
import numpy as np

# meshgen imports
from meshgen.rotation import Rotation
from meshgen.utility import get_section_string, flatten
from _collections import OrderedDict
from numpy import dtype



# constans for sets and BCs
__POINT__ = 0
__LINE__ = 1
__SURF__ = 2
__VOL__ = 3
__DIRICH__ = 10
__NEUMANN__ = 20


def get_type_geometry(item_description, return_type):
    """
    Return the string for different cases of return_type.
    """
    
    string_array = [
        [__POINT__, 'DNODE-NODE TOPOLOGY', 'DNODE', 'DESIGN POINT DIRICH CONDITIONS', 'DESIGN POINT NEUMANN CONDITIONS', 'DPOINT'],
        [__LINE__, 'DLINE-NODE TOPOLOGY', 'DLINE', 'DESIGN LINE DIRICH CONDITIONS', 'DESIGN LINE NEUMANN CONDITIONS', 'DLINE'],
        [__SURF__, 'DSURF-NODE TOPOLOGY', 'DSURFACE', 'DESIGN SURF DIRICH CONDITIONS', 'DESIGN SURF NEUMANN CONDITIONS', 'DSURF'],
        [__VOL__, 'DVOL-NODE TOPOLOGY', 'DVOLUME', 'DESIGN VOL DIRICH CONDITIONS', 'DESIGN VOL NEUMANN CONDITIONS', 'DVOL']
        ]
    
    for i, line in enumerate(string_array):
        if item_description in line:
            item_index = i
    
    if return_type == 'enum':
        return_index = 0
    elif return_type == 'setsection':
        return_index = 1
    elif return_type == 'settopology':
        return_index = 2
    elif return_type == 'dirich':
        return_index = 3
    elif return_type == 'neumann':
        return_index = 4
    elif return_type == 'bccounter':
        return_index = 5
    
    return string_array[item_index][return_index]


def get_type_bc(item_description, return_type):
    """
    Return the string for different cases of return_type.
    """
    
    string_array = [
        [__DIRICH__, 'DESIGN POINT DIRICH CONDITIONS', 'DESIGN LINE DIRICH CONDITIONS', 'DESIGN SURF DIRICH CONDITIONS', 'DESIGN VOL DIRICH CONDITIONS'],
        [__NEUMANN__, 'DESIGN POINT DIRICH NEUMANN', 'DESIGN LINE DIRICH NEUMANN', 'DESIGN SURF DIRICH NEUMANN', 'DESIGN VOL DIRICH NEUMANN']
        ]
    
    for i, line in enumerate(string_array):
        if item_description in line:
            item_index = i
    
    if return_type == 'enum':
        return_index = 0
    elif return_type == __POINT__:
        return_index = 1
    elif return_type == __LINE__:
        return_index = 2
    elif return_type == __SURF__:
        return_index = 3
    elif return_type == __VOL__:
        return_index = 4
    
    return string_array[item_index][return_index]
    
    
  
 
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



class Container(OrderedDict):
    """
    A base class for a container that will store node sets and 
    boundary conditions.
    """
    
    def __init__(self, aliases):
        """ Create a dictionary with the keys. """
        
        self.aliases = aliases
        
        # set empty lists
        for line in self.aliases:
            self[line[0]] = self.empty_item()
    
    
    def empty_item(self):
        """ What will be in the default items. """
        return []

    def _get_key(self, key):
        """ Return the key for the dictionary. Look in self.aliases. """
        for line in self.aliases:
            if key == line[0]:
                return line[0]
            elif key in line[1]:
                return line[0]
        print('Error, key {} not found!'.format(key))
    
    
    def merge_containers(self, other_container):
        """ Merge the contents of this set with a other SetContainer. """
        if type(other_container) == type(self):
            for key in other_container.keys():
                self[key].extend(other_container[key])
        else:
            print('Error, expected type {}, got {}!'.format(type(self), type(other_container)))
    
    def set_global(self):
        """ Set the global values in each set element. """
        for key in self.keys():
            for i, item in enumerate(self[key]):
                item.n_global = i + 1
            
    def __setitem__(self, key, value):
        """ Set items of the dictionary. """
        dict_key = self._get_key(key)
        OrderedDict.__setitem__(self, dict_key, value)
        
    
    def __getitem__(self, key):
        """ Gets items of the dictionary. """
        dict_key = self._get_key(key)
        return OrderedDict.__getitem__(self, dict_key)
    
    
    def append_item(self, key, data):
        """ Add set(s) to object. Set the type of set in the item. """
        data_type = self._get_key(key)
        if type(data) == list:
            self[key].extend(data)
            for item in data:
                item.item_type = data_type
        else:
            self[key].append(data)
            data.item_type = data_type


class ContainerGeom(Container):
    """
    This object contains sets for a mesh or as the return value of a
    mesh creation function.
    """
    
    def __init__(self):
        """
        Create empty list for different types of sets
        """
        
        aliases = [
            # [key, [# list of aliases], 'string in dat-file' ]
            [__POINT__, ['p', 'node', 'point', 'DNODE-NODE TOPOLOGY']],
            [__LINE__, ['l', 'line', 'DLINE-NODE TOPOLOGY']],
            [__SURF__, ['s', 'surf', 'surface', 'DSURF-NODE TOPOLOGY']],
            [__VOL__, ['v', 'vol', 'volume', 'DVOL-NODE TOPOLOGY']]
            ]
        Container.__init__(self, aliases)
        
    def set_global(self, all_sets=False):
        """ Set the global values in each set element. With the flaf all_sets """
        for key in self.keys():
            for i, item in enumerate(self.get_sets(key, all_sets)):
                item.n_global = i + 1
    
    def get_sets(self, key, all_sets=False):
        """ If all_sets = False only the referenced sets are returned. """
        if all_sets:
            return self[key]
        else:
            return [item for item in self[key] if item.is_referenced]
    
    @property
    def point(self):
        return self[__POINT__]
    @property
    def line(self):
        return self[__LINE__]
    @property
    def surf(self):
        return self[__SURF__]
    @property
    def vol(self):
        return self[__VOL__]

class ContainerBC(Container):
    """ This object contains bc. """
    
    def __init__(self):
        """
        Create empty list for different types of bc
        """
        
        aliases = [
            # [key, [# list of aliases], 'string in dat-file' ]
            [__DIRICH__, ['dbc', 'dirich']],
            [__NEUMANN__, ['nbc', 'neumann']]
            ]
        Container.__init__(self, aliases)
        
    def empty_item(self):
        """ What will be in the default items. """
        return ContainerGeom()
    
    def merge_containers(self, other_container):
        """ Merge the contents of this set with a other ContainerBC. """
        if type(other_container) == type(self):
            for key1 in other_container.keys():
                for key2 in other_container[key1].keys():
                    self[key1,key2].extend(other_container[key1,key2])
        else:
            print('Error, expected type {}, got {}!'.format(type(self), type(other_container)))
            
    def set_global(self):
        """ Check if . """
        for key1 in self.keys():
            for key2 in self[key1].keys():
                for i, item in enumerate(self[key1,key2]):
                    item.n_global = i + 1
    
    def __setitem__(self, key, value):
        """
        Set items of the dictionary.
        This is done with a tuple containing the bc mode, and geom type
        """
        
        if type(key) == tuple:
            self[key[0]][key[1]] = value
        else:
            dict_key = self._get_key(key)
            OrderedDict.__setitem__(self, dict_key, value)
    
    def __getitem__(self, key):
        """
        Gets items of the dictionary.
        This is done with a tuple containing the bc mode, and geom type
        """
        
        if type(key) == tuple:
            return self[key[0]][key[1]]
        else:
            dict_key = self._get_key(key)
            return OrderedDict.__getitem__(self, dict_key)


class BC(object):
    """ This object is one BC. """
    
    def __init__(self, set_item, bc_string, format_replacement=None):
        """
        Set the default values. Format_replacement will be called on string.
        """
        
        self.bc_string = bc_string
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


class GeometrySet(object):
    """
    Represents a set of nodes, for points, lines, surfaces or volumes.
    """
     
    def __init__(self, name, nodes=None):
        """
        Define the type of the set
        """
        
        # the name will be a list of names, as the parent meshes will also
        # be in the name
        if type(name) == list:
            self.name = name
        else:
            self.name = [name]
        
        self.nodes = []
        if nodes:
            self.add_node(nodes)
        
        self.n_global = None
        self.is_dat = False
        self.item_type = None
        self.is_referenced = False
     
     
    def add_node(self, add):
        """
        Check if the object or list of objects given is a node and add it to self.
        """
         
        if type(add) == Node:
            if not add in self.nodes:
                self.nodes.append(add)
        elif type(add) == list:
            for item in add:
                self.add_node(item)
        else:
            print('ERROR only Nodes and list of Nodes can be added to this object.')
    
    
    def get_dat_lines(self):
        """ Return the dat lines for this object. """
        return ['NODE {} {} {}'.format(node.n_global, get_type_geometry(self.item_type, 'settopology'), self.n_global) for node in self.nodes]

    
    def get_dat_name(self):
        """ Return a comment with the name of this set. """
        return '// {} {} name in beamgen: {}'.format(
            get_type_geometry(self.item_type, 'settopology'),
            self.n_global,
            '_'.join([str(item) for item in flatten(self.name)])
            )
        
    def output_to_dat(self):
        """
        Check if the item is linked to any boundary conditions. If not,
        the set will not appear in the dat file.
        """
        return True
        

class Material(object):
    """ Holds material definition for beams and solids. """
    
    def __init__(self, material_string, youngs_modulus, nu, density, diameter, shear_correction=0.75):
        
        self.material_string = material_string
        self.youngs_modulus = youngs_modulus
        self.nu = nu
        self.density = density
        self.diameter = diameter
        self.area = diameter**2 * np.pi * 0.25
        self.mom2 = (diameter*0.5)**4 * np.pi * 0.25
        self.mom3 = self.mom2
        self.polar = self.mom2 + self.mom3
        self.shear_correction = shear_correction
        
        self.n_global = None
        
    
    def get_dat_line(self):
        """ Return the line for the dat file. """
        return 'MAT {} {} YOUNG {} POISSONRATIO {} DENS {} CROSSAREA {} SHEARCORR {} MOMINPOL {} MOMIN2 {} MOMIN3 {}'.format(
            self.n_global,
            self.material_string,
            self.youngs_modulus,
            self.nu,
            self.density,
            self.area,
            self.shear_correction,
            self.polar,
            self.mom2,
            self.mom3
            )


class Function(object):
    """ Holds information for a function. """
    
    def __init__(self, data):
        if type(data) == list:
            self.data = data
        else:
            self.data = [data]

        self.n_global = None
        self.is_dat = False
    
    def get_dat_lines(self):
        """ Return the lines for the dat file. """
        return self.data
    
    def __str__(self):
        """ Check if the function has a global index. """
        if self.n_global:
            return str(self.n_global)
        else:
            print('Error function does not have a global index! It is probably not added to the mesh')


class BaseMeshItem(object):
    """
    A base class for nodes, elements, sets and so on that are given in dat files.
    """
    
    def __init__(self, *args, dat_string=None, dat_list=None):
        """ The defualt case is just set by a string from a dat file. """
        
        # if one argument is given check the type
        if len(args) == 1:
            if type(args[0]) == str:
                self.dat_string = args[0]
            elif type(args[0]) == list:
                self.dat_list = args[0]
            else:
                print('ERROR, type of arg not expected!')
        elif len(args) == 0:
            self.dat_string = dat_string
            self.dat_list = dat_list
        else:
            print('ERROR, does not support arg with len > 1')
        self.is_dat = True
        self.n_global = None
        self.is_referenced = True


    def output_to_dat(self):
        """ If the object will be shown in the dat file. """
        return True
    
    
    def get_dat_line(self):
        """ By default return the dat string. """
        return self.dat_string
    
    
    def get_dat_lines(self):
        """ By default return the dat list. """
        return self.dat_list


class Node(object):
    """ A class that represents one node of the mesh in the input file. """
    
    
    def __init__(self, coordinates, rotation=None):
        """
        Each node has a position and an optional rotation object.
        The n_global value is set when the model is writen to a dat file.
        """
        
        self.coordinates = np.array(coordinates)
        self.rotation = rotation
        self.n_global = None
        self.is_dat = False
        self.connected_elements = []
        self.connected_couplings = []
        # for the end nodes of a line
        self.is_end_node = False

    
    def rotate(self, rotation, only_rotate_triads=False):
        """
        Rotate the node.
        Default values is that the nodes is rotated around the origin.
        If only_rotate_triads is True, then only the triads are rotated,
        the position of the node stays the same.
        """
        
        # do not do anything if the node does not have a rotation
        if self.rotation:
            # apply the rotation to the triads
            self.rotation = rotation * self.rotation
        
        # rotate the positions (around origin)
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
    
    def __init__(self, material, element_name=None, node_create=None, mesh=None):
        """
        Set the base values for a beam element.
        node_create is described in self.create_beam
        """
        
        # name that will be displayed in the dat file
        self.element_name = element_name
        
        self.nodes = []
        self.node_create = node_create
        
        self.material = material
        if mesh:
            mesh.add_material(material)
        
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
                self._add_node(local_index, tmp_node)
            else:
                self._add_node(self.node_create[0][1], nodes[-1])
    
    
    def _add_node(self, local_node_index, node):
        """
        Add the node to the element and place a link to the element in
        the node object.
        """
        
        if self.nodes[local_node_index] == None:
            self.nodes[local_node_index] = node
            node.connected_elements.append(self)
        else:
            print('ERROR, node list should not me filled')


class Beam3rHerm2Lin3(Beam):
    """ Represents a BEAM3R HERM2LIN3 element. """
    
    def __init__(self, material, mesh=None):
        """ Set the data for this beam element. """
        
        node_create = [
            [-1, 0, True],
            [0, 2, True],
            [1, 1, True]
            ]

        Beam.__init__(self, material,
                      element_name='BEAM3R HERM2LIN3',
                      node_create=node_create,
                      mesh=mesh
                      )
    
    
    def get_dat_line(self):
        """ Return the line for the dat file for this element. """
        
        string_nodes = ''

        for node in self.nodes:
            string_nodes += '{} '.format(node.n_global)
        
        string_triads = ''
        for node in self.nodes:
            string_triads += node.rotation.get_dat()
        
        return '{} {} {}MAT {} TRIADS{}'.format(
            self.n_global,
            self.element_name,
            string_nodes,
            self.material.n_global,
            string_triads
            )


class Mesh(object):
    """ Holds nodes, beams and couplings of beam_mesh geometry. """
    
    def __init__(self, name='mesh'):
        """ Set empty variables """
        
        self.name = name
        self.nodes = []
        self.elements = []
        self.materials = []
        self.functions = []
        self.sets = ContainerGeom()
        self.bc = ContainerBC()
        self.couplings = []
        
        # count the number of items created for numbering in the comments
        self.mesh_item_counter = {}
        
    
    def add_mesh(self, mesh, add_sets=True):
        """ Add other mesh to this one. """
        
        self.nodes.extend(mesh.nodes)
        self.elements.extend(mesh.elements)
        for material in mesh.materials:
            self.add_material(material)
        for function in mesh.functions:
            self.add_function(function)
        if add_sets:
            self.sets.merge_containers(mesh.sets)
        self.bc.merge_containers(mesh.bc)
        self.couplings.extend(mesh.couplings)
    
    
    def add_coupling(self, coupling):
        """ Add a coupling to the mesh object. """
        
        # first perform checks
        # check that all nodes have the same position and are in mesh
        pos_set = False
        for node in coupling.nodes:
            if not node in self.nodes:
                print('Error, node not in mesh!')
            if not pos_set:
                pos_set = True
                pos = node.coordinates
            if np.linalg.norm(pos - node.coordinates) > 1e-8:
                print('Error, nodes of coupling do not have the same positions!')
        
        self.couplings.append(coupling)
        
        # add set with coupling conditions
        node_set = GeometrySet([coupling.name, 'coupling'], coupling.nodes)
        self.sets.append_item(__POINT__, node_set)
        coupling.node_set = node_set
        for node in coupling.nodes:
            node.connected_couplings = coupling
        
    
    def add_bc(self, bc_type, bc):
        """ Add a boundary condition to this mesh. """
        
        for key in self.sets.keys():
            if bc.set in self.sets[key]:
                break
        else:
            print('Error, the set is not yet added to this mesh object. BC can not be added until the set is added!')
            return
        
        if not bc in self.bc[bc_type, bc.set.item_type]:
            self.bc[bc_type, bc.set.item_type].append(bc)
        else:
            print('Error, each BC can only be added once!')
            
    def add_function(self, function):
        """ Add a function to this mesh item. """
        if not function in self.functions:
            self.functions.append(function)
    
    def add_material(self, material):
        """Add a material to this mesh. Every material can only be once in a mesh. """       
        if not material in self.materials:
            self.materials.append(material)
    
    
    def translate(self, vector):
        """ Move all nodes of this mesh by the vector. """
        
        for node in self.nodes:
            if not node.is_dat:
                node.coordinates += vector
    
    
    def rotate(self, rotation, origin=None):
        """ Rotate the geometry about the origin. """
        
        # get numpy array with all quaternions for the nodes
        quaternions = np.zeros([len(self.nodes),4])
        rot_new = np.zeros([len(self.nodes),4])
        for i, node in enumerate(self.nodes):
            if not node.is_dat:
                tmp = node.rotation.get_quaternion()
                quaternions[i,0] = tmp[0]
                quaternions[i,1:] = tmp[1]
        
        # get quaternion of rotation
        rot_quaternion = np.zeros(4)
        tmp = rotation.get_quaternion()
        rot_quaternion[0] = tmp[0]
        rot_quaternion[1:] = tmp[1]
        
        # get the new quaternions of the nodes
        rot_new[:,0] = quaternions[:,0] * rot_quaternion[0] - np.dot(quaternions[:,1:], rot_quaternion[1:])
        rot_new[:,1:] = (
            quaternions[:,1:] * rot_quaternion[0] + 
            np.dot(np.transpose([quaternions[:,0]]),[rot_quaternion[1:]]) -
            np.cross(quaternions[:,1:], rot_quaternion[1:])
            )
        # transform to rotation vector
        cos = rot_new[:,0]
        sin = np.linalg.norm(rot_new[:,1:],axis=1)
        
        phi = 2*np.arctan2(sin, cos)
        n = rot_new[:,1:]
        for i in range(len(n)):
            if np.abs(sin[i] < 1e-10):
                n[i,:] = [1,0,0]
                phi[i] = 0
            else:
                n[i,:] = n[i,:] / sin[i]
        
        # rotate the position of the nodes
        R = rotation.get_rotation_matrix()
        
        # set the new rotations and positions
        for i, node in enumerate(self.nodes):
            if not node.is_dat:
                node.rotation.phi = phi[i]
                node.rotation.n = n[i,:]
        
                # move coordinates to origin
                if not origin is None:
                    node.coordinates -= origin

                node.coordinates = np.dot(R, node.coordinates)
                
                # move coordinates back from origin
                if not origin is None:
                    node.coordinates += origin
    
    
    def add_connections(self, input_list, connection_type='fix'):
        """
        Search through nodes and connect all nodes with the same coordinates.
        """
        
        # make a copy of the input list, only consider nodes that are linked to one element
        node_list = [node for node in input_list if node.is_end_node]
        
        close_node_list = []

        # get array of nodal coordinates
        coordinates = np.zeros([len(node_list),3])
        
        # get max and min coorinates of nodes
        x_max = y_max = z_max = -1000
        x_min = y_min = z_min = 1000
        for i, node in enumerate(node_list):
            if node.coordinates[0] > x_max:
                x_max = node.coordinates[0]
            if node.coordinates[1] > y_max:
                y_max = node.coordinates[1]
            if node.coordinates[2] > z_max:
                z_max = node.coordinates[2]
            if node.coordinates[0] < x_min:
                x_min = node.coordinates[0]
            if node.coordinates[1] < y_min:
                y_min = node.coordinates[1]
            if node.coordinates[2] < z_min:
                z_min = node.coordinates[2]
            
            coordinates[i, :] = node.coordinates

        # split up domain into 20x20x20 cubes
        n_seg = 20
        x_seg = (x_max - x_min) / (n_seg-1)
        y_seg = (y_max - y_min) / (n_seg-1)
        z_seg = (z_max - z_min) / (n_seg-1)
        segments = [[[[] for k in range(n_seg)] for j in range(n_seg)] for i in range(n_seg)]
        #print(segments)
        for i, coord in enumerate(coordinates):
            ix = int((coord[0]-x_min+1e-5) // x_seg)
            iy = int((coord[1]-y_min+1e-5) // y_seg)
            iz = int((coord[2]-z_min+1e-5) // z_seg)
            #print([ix,iy,iz])
            #print(i)
            #print(len(node_list))
            segments[ix][iy][iz].append(node_list[i])  
        
        # loop over all segments
        for ix in range(n_seg):
            for iy in range(n_seg):
                for iz in range(n_seg):
                    node_list = segments[ix][iy][iz]
                    # loop over all (remaining) nodes in list
                    while len(node_list) > 1:
                        # check the first node with all others
                        current_node = node_list[0]
                        close_to_this_node = [current_node]
                        del node_list[0]
                        for i, node in enumerate(node_list):
                            if np.linalg.norm(current_node.coordinates-node.coordinates) < 1e-8:
                                close_to_this_node.append(node)
                                node_list[i] = None
                        node_list = [item for item in node_list if item is not None]
                        # if more than one node is close add to close_node_list
                        if len(close_to_this_node) > 1:
                            close_node_list.append(close_to_this_node)
        
        # add connections to mesh
        for nodes in close_node_list:
            # add sets to mesh
            self.add_coupling(Coupling(nodes, connection_type))
    
    
    def get_nodes_by_function(self, function, overlapping=True):
        """
        Return all nodes for which the function evaluates to true.
        overlaping means that nodes with the same coordinates are also added.
        """
        
        node_list = []
        for node in self.nodes:
            if function(node):
                if overlapping:
                    node_list.append(node)
                else:
                    for true_node in node_list:
                        if np.linalg.norm(true_node.coordinates - node.coordinates) < 1e-8:
                            break
                    else:
                        node_list.append(node)
        return node_list
    
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
                node.coordinates = np.array([
                    r * np.cos(phi),
                    r * np.sin(phi),
                    node.coordinates[2]
                    ])
                
    def _get_mesh_name(self, name, mesh_type):
        """
        Return the name for the mesh item. This name will be the prefix for
        all set names created by the mesh create function.
        """
        
        if not name:
            # name was not given, is None
            if not mesh_type in self.mesh_item_counter.keys():
                # mesh_type does not exist in counter -> this is first mesh_type
                self.mesh_item_counter[mesh_type] = 0
            # add one to the counter
            self.mesh_item_counter[mesh_type] += 1
            return [mesh_type, self.mesh_item_counter[mesh_type]]
        else:
            return name
    
    
    def add_beam_mesh_line(self,
                           beam_object,
                           material,
                           start_point,
                           end_point,
                           n,
                           name=None,
                           add_sets=True,
                           add_first_node=True
                           ):
        """
        A straight line of beam elements.
            n: Number of elements along line
        """
        
        # get name for the mesh added
        name = self._get_mesh_name(name, 'line')
        
        # direction vector of line
        direction = np.array(end_point) - np.array(start_point)
        
        # rotation for this line (is constant on the whole line)
        t1 = direction / np.linalg.norm(direction)
        # check if the z or y axis are larger projected onto the direction
        if abs(np.dot(t1,[0,0,1])) < abs(np.dot(t1,[0,1,0])):
            t2 = [0,0,1]
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
        
        # saave the index of the first node
        if add_first_node:
            node_start = len(self.nodes)
        else:
            node_start = len(self.nodes) - 1
        
        # create the beams
        for i in range(n):
            
            functions = get_beam_function(
                start_point + i*direction/n,
                start_point + (i+1)*direction/n
                )
            
            tmp_beam = beam_object(material, mesh=self)
            if add_first_node and i == 0:
                tmp_beam.create_beam(self.nodes, functions[0], functions[1], create_first=True)
            else:
                tmp_beam.create_beam(self.nodes, functions[0], functions[1], create_first=False)
            self.elements.append(tmp_beam)
        
        
        # add sets to mesh
        node_set_line = ContainerGeom()
        node_set_line.append_item(__POINT__, GeometrySet([name, 'start'], self.nodes[node_start]))
        node_set_line.append_item(__POINT__, GeometrySet([name, 'end'], self.nodes[-1]))
        self.nodes[node_start].is_end_node = True
        self.nodes[-1].is_end_node = True
        node_set_line.append_item(__LINE__, GeometrySet(name, self.nodes[node_start:]))
        if add_sets:
            self.sets.merge_containers(node_set_line)
        
        # return set container
        return node_set_line
    
    
    def add_beam_mesh_honeycomb_flat(self,
                                     beam_object,
                                     material,
                                     width,
                                     n_width,
                                     n_height,
                                     elements_per_line,
                                     closed_width=True,
                                     closed_height=True,
                                     connection_type='fix',
                                     name=None,
                                     add_sets=True,
                                     create_couplings=True
                                     ):
        """
        Add a flat honeycomb structure
        """
        
        def add_line(pointa, pointb):
            """ Shortcut to add line. """
            geom_set = self.add_beam_mesh_line(
                beam_object,
                material,
                pointa,
                pointb,
                elements_per_line,
                add_sets=False
                )
            return geom_set
        
        # get name for the mesh added
        name = self._get_mesh_name(name, 'honeycomb_flat')
        
        # list for nodes -> used for connections
        honeycomb_nodes = []
        
        # shortcuts
        sin30 = np.sin(np.pi/6)
        cos30 = np.sin(2*np.pi/6)
        a = width * 0.5 / cos30
        
        nx = np.array([1.,0.,0.])
        ny = np.array([0.,1.,0.])
        
        zig_zag_y = ny * a * sin30 * 0.5
        
        # create zig zag lines, first node is at [0,0,0]
        origin = np.array([0,a*0.5*sin30,0])
        
        # loop to create elements
        for i_height in range(n_height + 1):
            
            base_row = origin + ny * a * (1 + sin30 ) * i_height
        
            # if the first node is up or down of base_row
            if i_height % 2 == 0:
                direction = 1
            else:
                direction = -1    

            for i_width in range(n_width + 1):
            
                base_zig_zag = base_row + direction * zig_zag_y + width * i_width * nx
                
                # do not add on last run
                if i_width < n_width:
                    tmp = add_line(
                        base_zig_zag,
                        base_zig_zag + nx * width * 0.5 - 2 * direction * zig_zag_y
                        )
                    # add the nodes to the node list for connections and sets
                    honeycomb_nodes.extend([item.nodes[0] for item in tmp.point])
                    tmp = add_line(
                        base_zig_zag + nx * width * 0.5 - 2 * direction * zig_zag_y,
                        base_zig_zag + nx * width
                        )
                    honeycomb_nodes.extend([item.nodes[0] for item in tmp.point])
                    
                
                if i_height % 2 == 0:
                    base_vert = base_zig_zag
                else:
                    base_vert = base_zig_zag + nx * width * 0.5 - 2 * direction * zig_zag_y
                
                # check if width is closed
                if (i_width < n_width) or (direction==1 and closed_width):
                    # check if height is closed
                    if not (i_height == n_height) or (not closed_height): 
                        tmp = add_line(
                            base_vert,
                            base_vert + ny * a
                            )
                        honeycomb_nodes.extend([item.nodes[0] for item in tmp.point])

        # function to get nodes for boundaries
        def node_in_box(x_range,y_range,z_range):
            def funct(node):
                coord = node.coordinates
                eps = 1e-8
                if -eps + x_range[0] < coord[0] < x_range[1] + eps:
                    if -eps + y_range[0] < coord[1] < y_range[1] + eps:
                        if -eps + z_range[0] < coord[2] < z_range[1] + eps:
                            # also check if the node is in honeycomb_nodes -> we
                            # only want nodes that are on the crossing points of the mesh
                            if node in honeycomb_nodes:
                                return True
                return False
            return funct
        
        x_max = y_max = 0
        for node in honeycomb_nodes:
            if node.coordinates[0] > x_max:
                x_max = node.coordinates[0]
            if node.coordinates[1] > y_max:
                y_max = node.coordinates[1]
                
        node_set = ContainerGeom()
        node_set.append_item(__POINT__, GeometrySet([name, 'north'],
            self.get_nodes_by_function(node_in_box([0,x_max], [y_max,y_max], [-1,1]))
            ))
        node_set.append_item(__POINT__, GeometrySet([name, 'east'],
            self.get_nodes_by_function(node_in_box([x_max,x_max], [0,y_max], [-1,1]))
            ))
        node_set.append_item(__POINT__, GeometrySet([name, 'south'],
            self.get_nodes_by_function(node_in_box([0,x_max], [0,0], [-1,1]))
            ))
        node_set.append_item(__POINT__, GeometrySet([name, 'west'],
            self.get_nodes_by_function(node_in_box([0,0], [0,y_max], [-1,1]))
            ))
        if add_sets:
            self.sets.merge_containers(node_set)
        
        # add connection for nodes with same positions
        if create_couplings:
            self.add_connections(honeycomb_nodes)
        
        return node_set
    
    def add_beam_mesh_honeycomb(self,
                           beam_object,
                           material,
                           diameter,
                           n_circumference,
                           n_height,
                           n_element,
                           name=None,
                           add_sets=True
                           ):
        """
        TODO
        """
        
        # calculate stuff
        width = diameter * np.pi / n_circumference
        
        # first create a mesh with the flat mesh
        mesh_temp = Mesh(name='honeycomb_' + str(1))
        mesh_temp.add_beam_mesh_honeycomb_flat(Beam3rHerm2Lin3, material, width, n_circumference, n_height, n_element,
                                                                    closed_width=False,
                                                                    closed_height=False,
                                                                    create_couplings=False,
                                                                    add_sets=False
                                                                    )
        
        print('add flat honeycomb complete')
        
        # move the mesh to the correct position
        mesh_temp.rotate(Rotation([1,0,0],np.pi/2))
        mesh_temp.rotate(Rotation([0,0,1],np.pi/2))
        
        
        
        mesh_temp.translate([diameter/2, 0, 0])
        mesh_temp.wrap_around_cylinder()
        
        print('wraping complete')
        
        mesh_temp.add_connections(mesh_temp.nodes)
        
        print('connections complete')
        
        
        # function to get nodes for boundaries
        def node_in_box(x_range,y_range,z_range):
            def funct(node):
                coord = node.coordinates
                eps = 1e-8
                if -eps + x_range[0] < coord[0] < x_range[1] + eps:
                    if -eps + y_range[0] < coord[1] < y_range[1] + eps:
                        if -eps + z_range[0] < coord[2] < z_range[1] + eps:
                            # also check if the node is in honeycomb_nodes -> we
                            # only want nodes that are on the crossing points of the mesh
                            if node in mesh_temp.nodes:
                                return True
                return False
            return funct
        
        x_max = y_max = z_max = 0
        for node in mesh_temp.nodes:
            if node.coordinates[0] > x_max:
                x_max = node.coordinates[0]
            if node.coordinates[1] > y_max:
                y_max = node.coordinates[1]
            if node.coordinates[2] > z_max:
                z_max = node.coordinates[2]
                
        node_set = ContainerGeom()
        node_set.append_item(__POINT__, GeometrySet([name, 'bottom'],
            mesh_temp.get_nodes_by_function(node_in_box([-2*x_max,2*x_max], [-2*y_max,2*y_max], [0,0]))
            ))
        node_set.append_item(__POINT__, GeometrySet([name, 'top'],
            mesh_temp.get_nodes_by_function(node_in_box([-2*x_max,2*x_max], [-2*y_max,2*y_max], [z_max,z_max]))
            ))
        
        self.add_mesh(mesh_temp)
        
        if add_sets:
            self.sets.merge_containers(node_set)
        
        
        
        return node_set


class MeshInput(Mesh):
    """
    This is just a BeamMesh class that can additionally manage sections for the input file.
    """
    
    def _add_dat_section(self, section_name, section_data):
        """
        Check if the section has to be added to the mesh of if it is just a
        basic input section.
        """
        
        def add_bc(section_header):
            """ Add boundary conditions to the object. """
            
            for i, item in enumerate(section_data):
                # first line is number of BCs skip this one
                if i > 0:
                    self.bc[get_type_bc(section_header, 'enum'), get_type_geometry(section_header, 'enum')].append(BaseMeshItem(item))
        
        def add_set(section_header):
            """ Add sets of points, lines, surfs or volumes to item. """
            
            if len(section_data) > 0:
                # look for the individual sets 
                last_index = 1
                dat_list = []
                for line in section_data:
                    if last_index == int(line.split()[3]):
                        dat_list.append(line)
                    else:
                        last_index = int(line.split()[3])
                        self.sets[section_header].append(BaseMeshItem(dat_list))
                        dat_list = [line]
                self.sets[section_header].append(BaseMeshItem(dat_list))
        
        if section_name == 'MATERIALS':
            for line in section_data:
                self.materials.append(BaseMeshItem(line))
        elif section_name.endswith('CONDITIONS'):
            add_bc(section_name)
        elif section_name.endswith('TOPOLOGY'):
            add_set(section_name)
        elif section_name == 'NODE COORDS':
            for line in section_data:
                self.nodes.append(BaseMeshItem(line))
        elif section_name == 'STRUCTURE ELEMENTS':
            for line in section_data:
                self.elements.append(BaseMeshItem(line))
        elif section_name == 'DESIGN DESCRIPTION':
            pass
        elif section_name.startswith('FUNCT'):
            self.functions.append(BaseMeshItem(section_data))
        else:
            # descion is not in mesh
            return 1
        
        # section is in mesh
        return 0
        
        
    def get_dat_lines(self, print_set_names=False, print_all_sets=False):
        """
        Get the lines for the input file that contain the information for
        the mesh.
        """
        
        def set_n_global(data_list):
            """ Set n_global in every item of list. """
            for i, item in enumerate(data_list):
                item.n_global = i + 1
        
        def get_section_dat(section_name, data_list, header_lines=None):
            """
            Output a section name and apply the get_dat_line for each list item.
            """
            
            lines.append(get_section_string(section_name))
            if header_lines:
                if type(header_lines) == list:
                    lines.extend(header_lines)
                elif type(header_lines) == str:
                    lines.append(header_lines)
                else:
                    print('ERROR, you can either add a list or a string')
            for item in data_list:
                lines.append(item.get_dat_line())
        
        
        # first all nodes, elements, sets and couplings are assigned a global value
        set_n_global(self.nodes)
        set_n_global(self.elements)
        set_n_global(self.materials)
        set_n_global(self.functions)
        
        # first set referenc counter of sets to False, then add bc and then renumber sets
        for i, coupling in enumerate(self.couplings):
            coupling.n_global = i + 1
            if coupling.node_set.is_referenced:
                print('Error this set can not be referenced by something else')
                print(coupling.node_set.nodes[0].coordinates)
                print(type(coupling.node_set))
            else:
                coupling.node_set.is_referenced = True
        self.bc.set_global()
        self.sets.set_global(all_sets=print_all_sets)
        
        lines = []
        
        # add the material data
        get_section_dat('MATERIALS', self.materials)
        
        # add the functions
        for i, funct in enumerate(self.functions):
            lines.append(get_section_string('FUNCT{}'.format(str(i+1))))
            lines.extend(funct.get_dat_lines())
        
        # add the design descriptions
        lines.append(get_section_string('DESIGN DESCRIPTION'))
        lines.append('NDPOINT {}'.format(len(self.sets.get_sets('point', all_sets=print_all_sets))))
        lines.append('NDLINE {}'.format(len(self.sets.get_sets('line', all_sets=print_all_sets))))
        lines.append('NDSURF {}'.format(len(self.sets.get_sets('surf', all_sets=print_all_sets))))
        lines.append('NDVOL {}'.format(len(self.sets.get_sets('vol', all_sets=print_all_sets))))
        
        # add boundary conditions
        for key1 in self.bc.keys():
            for key2 in self.bc[key1].keys():
                if len(self.bc[key1, key2]) > 0:
                    lines.append(get_section_string(get_type_bc(key1, key2)))
                    lines.append('{} {}'.format(get_type_geometry(key2, 'bccounter'), len(self.bc[key1, key2])))
                    for bc in self.bc[key1, key2]:
                        lines.append(bc.get_dat_line())

        # add the couplings
        lines.append(get_section_string('DESIGN POINT COUPLING CONDITIONS'))
        lines.append('DPOINT {}'.format(len(self.couplings)))
        for coupling in self.couplings:
            lines.append(coupling.get_dat_line())
        
        # add the node sets
        for key in self.sets.keys():
            if len(self.sets[key]) > 0:
                lines.append(get_section_string(get_type_geometry(key, 'setsection')))
                # print the description for the sets
                for mesh_set in self.sets.get_sets(key, all_sets=print_all_sets):
                    if (not mesh_set.is_dat) and print_set_names:
                        lines.append(mesh_set.get_dat_name())
                for mesh_set in self.sets.get_sets(key, all_sets=print_all_sets):
                    lines.extend(mesh_set.get_dat_lines())
                    
        # add the nodal data
        get_section_dat('NODE COORDS', self.nodes)

        # add the element data
        get_section_dat('STRUCTURE ELEMENTS', self.elements)
        
        return lines
        
        
        
        
        
        
        
        
        
        
        
        
        
        