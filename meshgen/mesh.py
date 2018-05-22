
# python packages
import numpy as np

# meshgen imports
from meshgen.rotation import Rotation
from meshgen.utility import get_section_string
from _collections import OrderedDict





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


class SetContainer(OrderedDict):
    """
    This object contains sets for a mesh or as the return value of a
    mesh creation function.
    """
    
    def __init__(self):
        """
        Create empty list for different types of sets
        """
        
        self.aliases = [
            # [key, [# list of aliases], 'string in dat-file' ]
            ['DNODE-NODE TOPOLOGY', ['p', 'node', 'point'], 'DNODE'],
            ['DLINE-NODE TOPOLOGY', ['l', 'line'], 'DLINE'],
            ['DSURF-NODE TOPOLOGY', ['s', 'surf', 'surface'], 'DSURFACE'],
            ['DVOL-NODE TOPOLOGY', ['v', 'vol', 'volume'], 'DVOLUME']
            ]
        
        # set empty lists
        for line in self.aliases:
            self[line[0]] = []
    
    
    def _get_key(self, key, return_dat_string=False):
        """ Return the key for the dictionary. Look in self.aliases. """
        for line in self.aliases:
            if key == line[0]:
                if return_dat_string:
                    return line[2]
                else:
                    return line[0]
            elif key in line[1]:
                if return_dat_string:
                    return line[2]
                else:
                    return line[0]
        print('Error, key {} not found!'.format(key))
    
    
    def merge_sets(self, other_set):
        """ Merge the contents of this set with a other SetContainer. """
        if type(other_set) == SetContainer:
            for key in other_set.keys():
                self[key].extend(other_set[key])
        else:
            print('Error, expected type SetContainer, got {}!'.format(type(other_set)))
    
    def set_global(self):
        """ Set the global values in each set element. """
        for key in self.keys():
            dat_string = self._get_key(key, return_dat_string=True)
            for i, node_set in enumerate(self[key]):
                node_set.n_global = i + 1
                node_set.set_type = dat_string
            
    def __setitem__(self, key, value):
        """ Set items of the dictionary. """
        dict_key = self._get_key(key)
        OrderedDict.__setitem__(self, dict_key, value)
        
    
    def __getitem__(self, key):
        """ Gets items of the dictionary. """
        dict_key = self._get_key(key)
        return OrderedDict.__getitem__(self, dict_key)


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
        self.set_type = None
     
     
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
        return ['NODE {} {} {}'.format(node.n_global, self.set_type, self.n_global) for node in self.nodes]

    def get_dat_name(self):
        """ Return a comment with the name of this set. """
        
        # flatten name list
        def flatten(data):
            flatten_list = []
            if type(data) == list:
                for item in data:
                    flatten_list.extend(flatten(item))
                return flatten_list 
            else:
                return [str(data)]
        return '// {} {} name in beamgen: {}'.format(
            self.set_type,
            self.n_global,
            '_'.join(flatten(self.name))
            )
        

class Material(object):
    """ Holds material definition for beams and solids. """
    
    def __init__(self, material_string):
        self.material_string = material_string
        self.n_global = None
    
    def get_dat_line(self):
        """ Return the line for the dat file. """
        return 'MAT {} {}'.format(self.n_global, self.material_string)


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
        
        self.coordinates = coordinates
        self.rotation = rotation
        self.n_global = None
        self.is_dat = False
        self.connected_elements = []

    
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
        
        return '{} {} {}MAT {} TRIADS{} FAD'.format(
            self.n_global,
            self.element_name,
            string_nodes,
            self.material.n_global,
            string_triads
            )


class Mesh(object):
    """ Holds nodes, beams and couplings of beam_mesh geometry. """
    
    def __init__(self, name=None):
        """ Set empty variables """
        
        self.name = name
        self.nodes = []
        self.elements = []
        self.materials = []
        self.functions = []
        self.sets = SetContainer()
        
        # count the number of items created for numbering in the comments
        self.mesh_item_counter = {}
        
        #self.couplings = []
        
    
    def add_mesh(self, mesh, add_sets=True):
        """ Add other mesh to this one. """
        
        self.nodes.extend(mesh.nodes)
        self.elements.extend(mesh.elements)
        for material in mesh.materials:
            self.add_material(material)
        if add_sets:
            self.sets.merge_sets(mesh.sets)
    
    
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
    
    
    def add_beam_mesh_line(self, beam_object, material, start_point, end_point, n, name=None, add_sets=True, add_first_node=True):
        """
        A straight line of beam elements.
            n: Number of elements along line
        """
        
        # get name for the mesh added
        name = self._get_mesh_name(name, 'line')
        
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
        
        # create the beams
        for i in range(n):
            
            functions = get_beam_function(
                start_point + i*direction/n,
                start_point + (i+1)*direction/n
                )
            
            tmp_beam = beam_object(material, mesh=self)
            if add_first_node and i == 0:
                node_start = len(self.nodes)
                tmp_beam.create_beam(self.nodes, functions[0], functions[1], create_first=True)
            else:
                node_start = len(self.nodes) - 1
                tmp_beam.create_beam(self.nodes, functions[0], functions[1], create_first=False)
            self.elements.append(tmp_beam)
        
        
        # add sets to mesh
        node_set_line = SetContainer()
        node_set_line['point'].append(GeometrySet([name, 'start'], self.nodes[node_start]))
        node_set_line['point'].append(GeometrySet([name, 'end'], self.nodes[-1]))
        node_set_line['line'].append(GeometrySet(name, self.nodes[node_start:]))
        self.sets.merge_sets(node_set_line)
        
        # return set container
        return node_set_line


class MeshInput(Mesh):
    """
    This is just a BeamMesh class that can additionally manage sections for the input file.
    """
    
    def _add_dat_section(self, section_name, section_data):
        """
        Check if the section has to be added to the mesh of if it is just a
        basic input section.
        """
        
        def add_set(section_header):
            """ Add sets of points, lines, surfs or volumes to item. """
            
            if len(section_data) > 0:
                # look for the individual sets 
                last_index = 1
                set_dat_list = []
                for line in section_data:
                    if last_index == int(line.split()[3]):
                        set_dat_list.append(line)
                    else:
                        last_index = int(line.split()[3])
                        self.sets[section_header].append(BaseMeshItem(set_dat_list))
                        set_dat_list = [line]
                self.sets[section_header].append(BaseMeshItem(set_dat_list))
        
        if section_name == 'MATERIALS':
            for line in section_data:
                self.materials.append(BaseMeshItem(line))
        elif section_name == 'DESIGN LINE DIRICH CONDITIONS':
            pass
        elif section_name == 'DESIGN SURF DIRICH CONDITIONS':
            pass
        elif section_name == 'DNODE-NODE TOPOLOGY':
            add_set('point')
        elif section_name == 'DLINE-NODE TOPOLOGY':
            add_set('line')
        elif section_name == 'DSURF-NODE TOPOLOGY':
            add_set('surf')
        elif section_name == 'DVOL-NODE TOPOLOGY':
            add_set('vol')
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
        
        
    def get_dat_lines(self, print_set_names=False):
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
        set_n_global(self.functions)
        set_n_global(self.materials)
        self.sets.set_global()
        
        lines = []
        
        # add the material data
        get_section_dat('MATERIALS', self.materials)
        
        # add the functions
        for i, funct in enumerate(self.functions):
            lines.append(get_section_string('FUNCT{}'.format(str(i+1))))
            lines.extend(funct.get_dat_lines())
        
        # add the design descriptions
        lines.append(get_section_string('DESIGN DESCRIPTION'))
        lines.append('NDPOINT {}'.format(len(self.sets['point'])))
        lines.append('NDLINE {}'.format(len(self.sets['line'])))
        lines.append('NDSURF {}'.format(len(self.sets['surf'])))
        lines.append('NDVOL {}'.format(len(self.sets['vol'])))
        
        # add boundary conditions
        # TODO
        
        # add the coupings
        # TODO
        
        # add the node sets
        for key in self.sets.keys():
            if len(self.sets[key]) > 0:
                lines.append(get_section_string(key))
                # print the description for the sets
                for mesh_set in self.sets[key]:
                    if (not mesh_set.is_dat) and print_set_names:
                        lines.append(mesh_set.get_dat_name()) 
                for mesh_set in self.sets[key]:
                    lines.extend(mesh_set.get_dat_lines())
                    
        
        # add the nodal data
        get_section_dat('NODE COORDS', self.nodes)

        # add the element data
        get_section_dat('STRUCTURE ELEMENTS', self.elements)
        
        return lines
        
        
        
        
        
        
        
        
        
        
        
        
        
        