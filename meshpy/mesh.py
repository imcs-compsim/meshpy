
# python packages
import numpy as np
from _collections import OrderedDict

# meshpy imports
from . import Rotation, get_section_string, flatten, Beam, Beam3rHerm2Lin3, Node, BaseMeshItem, Function, Material


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
        
        
    def add(self, *args, **kwargs):
        """
        Add an item depending on what it is
        """
        
        if len(args) == 0:
            raise ValueError('At least one argument should be given!')
        elif len(args) == 1:
            add_item = args[0]
            if isinstance(add_item, Mesh):
                self.add_mesh(add_item, **kwargs)
            elif isinstance(add_item, Function):
                self.add_function(add_item, **kwargs)
            elif isinstance(add_item, BC):
                self.add_bc(add_item, **kwargs)
            elif isinstance(add_item, list):
                for item in add_item:
                    self.add(item, **kwargs)
            else:
                raise TypeError('Did not expect {}!'.format(type(item)))
        else:
            for item in args:
                self.add(item, **kwargs)
        
    
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
        
    
    def add_bc(self, bc):
        """ Add a boundary condition to this mesh. """
        
        bc_type = bc.type
        
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
    
    
    def get_nodes_by_function(self, function, overlapping=True, middle_nodes=False):
        """
        Return all nodes for which the function evaluates to true.
        overlaping means that nodes with the same coordinates are also added.
        """
        
        node_list = []
        for node in self.nodes:
            if node.is_middle_node:
                continue
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
    
    
    def add_beam_mesh_line(
            self,
            beam_object,
            material,
            start_point,
            end_point,
            n_el=1,
            start_node=None
                           #name=None,
                           #add_sets=True
            ):
        
        """Generate a straight line in this mesh.
        
        Args
        ----
        beam_object: Beam
            Class of beam that will be used for this line
        material: Material
            Material for this line
        start_point, end_point: np.array, list
            3D-coordinates for the start and end point of the line
        n_el: int
            Number of equally spaces beam elements along the line
        start_node: Node
            Node to use as the first node for this line. Use this if the line
            is connected to other lines (angles have to be the same, otherwise
            connections should be used)
        """
        
        self.add_material(material)
        
        ## get name for the mesh added
        #name = self._get_mesh_name(name, 'line')
        
        # Direction vector of line
        direction = np.array(end_point) - np.array(start_point)
        
        # Rotation for this line (is constant on the whole line)
        t1 = direction / np.linalg.norm(direction)
        # Check if the z or y axis are larger projected onto the direction
        if abs(np.dot(t1,[0,0,1])) < abs(np.dot(t1,[0,1,0])):
            t2 = [0,0,1]
        else:
            t2 = [0,1,0]
        rotation = Rotation.from_basis(t1, t2)
        
        # This function returns the function for the position and triads along
        # the beam element
        def get_beam_function(point_a, point_b):
            
            def position_function(xi):
                return 1/2*(1-xi)*point_a + 1/2*(1+xi)*point_b
            
            def rotation_function(xi):
                return rotation
            
            return (
                position_function,
                rotation_function
                )
        
        # nodes in this line
        if start_node is None:
            nodes = []
        else:
            nodes = [start_node]
        
        # create the beams
        for i in range(n_el):
            
            functions = get_beam_function(
                start_point + i*direction/n_el,
                start_point + (i+1)*direction/n_el
                )
            
            tmp_beam = beam_object(material=material)
            if (start_node is None) and i == 0:
                tmp_start_node = None
            else:
                tmp_start_node = nodes[-1]
            nodes.extend(tmp_beam.create_beam(functions[0], functions[1],
                                 start_node=tmp_start_node))
            self.elements.append(tmp_beam)
        
        # add nodes to mesh
        self.nodes.extend(nodes)
        
        # add sets to mesh
        node_set_line = ContainerGeom()
        node_set_line.append_item(__POINT__, GeometrySet(['start'], nodes[0]))
        node_set_line.append_item(__POINT__, GeometrySet(['end'], nodes[-1]))
        nodes[0].is_end_node = True
        nodes[-1].is_end_node = True
        node_set_line.append_item(__LINE__, GeometrySet('line', nodes))
        #if add_sets:
        self.sets.merge_containers(node_set_line)
        
        
        # return set container
        return node_set_line
    
    
    def add_beam_mesh_honeycomb_flat(
            self,
            beam_object,
            material,
            width,
            n_width,
            n_height,
            n_el=1,
            closed_width=True,
            closed_height=True,
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
                n_el=n_el#,
                #add_sets=False
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
                           n_el=1,
                           name=None,
                           add_sets=True,
                           closed_top=True,
                           vertical=True
                           ):
        """
        TODO
        """
        
        if vertical:
            width = diameter * np.pi / n_circumference 
            closed_width = False
            closed_height = closed_top
            rotation = Rotation([0,0,1],np.pi/2) * Rotation([1,0,0],np.pi/2)
            n_h = n_height
            n_w = n_circumference
        else:
            if not n_circumference % 2 == 0:
                raise ValueError('There has to be an even number of elements along the diameter in horizontal mode. Given {}!'.format(n_circumference))
            H = diameter * np.pi / n_circumference
            r = H / (1+np.sin(np.pi/6))
            width = 2*r*np.cos(np.pi/6)
            closed_width = closed_top
            closed_height = False
            rotation = Rotation([0,1,0],-np.pi/2)
            n_h = n_circumference
            n_w = n_height
        
        # first create a mesh with the flat mesh
        mesh_temp = Mesh(name='honeycomb_' + str(1))
        mesh_temp.add_beam_mesh_honeycomb_flat(Beam3rHerm2Lin3, material, width, n_w, n_h, n_el,
                                                                    closed_width=closed_width,
                                                                    closed_height=closed_height,
                                                                    create_couplings=False,
                                                                    add_sets=False
                                                                    )
        
        print('add flat honeycomb complete')
        
        # move the mesh to the correct position
        mesh_temp.rotate(rotation)
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

        
        
        
        
        
        
        
        
        