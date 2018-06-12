
# python packages
import numpy as np

# meshpy imports
from . import Rotation, Node, Function, Material, Element, mpy, NodeSetContainer, NodeSet, \
    BC, Coupling


   
  





class Mesh(object):
    """ Holds nodes, beams and couplings of beam_mesh geometry. """
    
    def __init__(self, name='mesh'):
        """ Set empty variables """

        self.nodes = []
        self.elements = []
        self.materials = []
        self.functions = []
        self.couplings = []
        
        self.sets = {}
        for key in mpy.geometry:
            self.sets[key] = []
        self.bc = {}
        for key1 in mpy.boundary_condition:
            self.bc[key1] = {}
            for key2 in mpy.geometry:
                self.bc[key1][key2] = []
        
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
            elif isinstance(add_item, Material):
                self.add_material(add_item, **kwargs)
            elif isinstance(add_item, Node):
                self.add_node(add_item, **kwargs)
            elif isinstance(add_item, Element):
                self.add_element(add_item, **kwargs)
            elif isinstance(add_item, NodeSet):
                self.add_set(add_item)
            elif isinstance(add_item, list):
                for item in add_item:
                    self.add(item, **kwargs)
            else:
                raise TypeError('Did not expect {}!'.format(type(add_item)))
        else:
            for item in args:
                self.add(item, **kwargs)
        
    
    def add_mesh(self, mesh):
        """ Add other mesh to this one. """
        
        for node in mesh.nodes:
            self.add_node(node)
        for element in mesh.elements:
            self.add_element(element)
        for material in mesh.materials:
            self.add_material(material)
        for function in mesh.functions:
            self.add_function(function)
        for key1 in self.bc.keys():
            for key2 in self.bc[key1].keys():
                self.bc[key1][key2].extend(mesh.bc[key1][key2])
        for key in self.sets.keys():
                self.sets[key].extend(mesh.sets[key])
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
        node_set = NodeSet(mpy.point, coupling.nodes)
        self.add(node_set)
        coupling.node_set = node_set
        for node in coupling.nodes:
            node.connected_couplings = coupling
        
    
    def add_bc(self, bc):
        """ Add a boundary condition to this mesh. """
        bc_key = bc.type
        geom_key = bc.set.geo_type
        
        self.bc[bc_key][geom_key].append(bc)
        self.sets[geom_key].append(bc.set)
            
    def add_function(self, function):
        """ Add a function to this mesh item. """
        if not function in self.functions:
            self.functions.append(function)
    
    def add_material(self, material):
        """Add a material to this mesh. Every material can only be once in a mesh. """       
        if not material in self.materials:
            self.materials.append(material)
    
    def add_node(self, node):
        """ Add a node to this mesh."""
        self.nodes.append(node)
        
    def add_element(self, element):
        """ Add a element to this mesh."""
        self.elements.append(element)
    
    def add_set(self, node_set):
        """ Add a node set to the mesh. """
        self.sets[node_set.geo_type].append(node_set)
    
    
    def translate(self, vector):
        """ Move all nodes of this mesh by the vector. """
        
        for node in self.nodes:
            if not node.is_dat:
                node.coordinates += vector
    
    
    def rotate(self, rotation, origin=None):
        """ Rotate the geometry about the origin. """
        
        # get numpy array with all quaternions and positions for the nodes
        rot1 = np.zeros([len(self.nodes),4],dtype=mpy.dtype)
        pos = np.zeros([len(self.nodes),3],dtype=mpy.dtype)
        for i, node in enumerate(self.nodes):
            if not node.is_dat:
                rot1[i,:] = node.rotation.get_quaternion()
                pos[i,:] = node.coordinates
        
        # check if origin has to be added
        if not origin is None:
            pos -= origin
        
        # new arrays
        rotnew = np.zeros_like(rot1)
        posnew = np.zeros_like(pos)
        
        # additional rotation
        rot2 = rotation.get_quaternion()
        
        # temp array for AceGen variables
        temp_val=[None for i in range(11)]
        
        # code generated with AceGen (rotation.nb)
        temp_val[0]=2*10**0*rot2[1]*rot2[2]
        temp_val[1]=2*10**0*rot2[0]*rot2[3]
        temp_val[2]=2*10**0*rot2[0]*rot2[2]
        temp_val[3]=2*10**0*rot2[1]*rot2[3]
        temp_val[4]=rot2[0]**2
        temp_val[5]=rot2[1]**2
        temp_val[6]=rot2[2]**2
        temp_val[10]=temp_val[4]-temp_val[6]
        temp_val[7]=rot2[3]**2
        temp_val[8]=2*10**0*rot2[0]*rot2[1]
        temp_val[9]=2*10**0*rot2[2]*rot2[3]
        posnew[:,0]=pos[:,1]*(temp_val[0]-temp_val[1])+pos[:,2]*(temp_val[2]+temp_val[3])+pos[:,0]*(temp_val[5]-temp_val[7]+temp_val[10])
        posnew[:,1]=pos[:,0]*(temp_val[0]+temp_val[1])+pos[:,1]*(temp_val[4]-temp_val[5]+temp_val[6]-temp_val[7])+pos[:,2]*(-temp_val[8] +temp_val[9])
        posnew[:,2]=pos[:,0]*(-temp_val[2]+temp_val[3])+pos[:,1]*(temp_val[8]+temp_val[9])+pos[:,2]*(-temp_val[5]+temp_val[7]+temp_val[10])
        rotnew[:,0]=rot1[:,0]*rot2[0]-rot1[:,1]*rot2[1]-rot1[:,2]*rot2[2]-rot1[:,3]*rot2[3]
        rotnew[:,1]=rot1[:,1]*rot2[0]+rot1[:,0]*rot2[1]+rot1[:,3]*rot2[2]-rot1[:,2]*rot2[3]
        rotnew[:,2]=rot1[:,2]*rot2[0]-rot1[:,3]*rot2[1]+rot1[:,0]*rot2[2]+rot1[:,1]*rot2[3]
        rotnew[:,3]=rot1[:,3]*rot2[0]+rot1[:,2]*rot2[1]-rot1[:,1]*rot2[2]+rot1[:,0]*rot2[3]
                
        if not origin is None:
            posnew += origin
            
        for i, node in enumerate(self.nodes):
            if not node.is_dat:
                node.rotation.q = rotnew[i,:]
                node.coordinates = posnew[i,:]
    
    
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
                node.coordinates[0] = r * np.cos(phi)
                node.coordinates[1] = r * np.sin(phi)
    
    
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
        direction = np.array(end_point,dtype=mpy.dtype) - \
            np.array(start_point,dtype=mpy.dtype)
        
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

        # set the nodes that are at the beginning and end of line (for search of
        # overlapping points)
        nodes[0].is_end_node = True
        nodes[-1].is_end_node = True
        
        # add nodes to mesh
        self.nodes.extend(nodes)
        
        # add sets to mesh
        return_set = NodeSetContainer()
        return_set['start'] = NodeSet(mpy.point, nodes=nodes[0])
        return_set['end'] = NodeSet(mpy.point, nodes=nodes[-1])
        return_set['line'] = NodeSet(mpy.point, nodes=nodes)
        return return_set
    
    
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
        
        # shortcuts
        sin30 = np.sin(np.pi/6)
        cos30 = np.sin(2*np.pi/6)
        a = width * 0.5 / cos30
        
        nx = np.array([1.,0.,0.])
        ny = np.array([0.,1.,0.])
        
        zig_zag_y = ny * a * sin30 * 0.5
        
        # create zig zag lines, first node is at [0,0,0]
        origin = np.array([0,a*0.5*sin30,0])
        
        # node index from the already existing nodes
        i_node_start = len(self.nodes)
        
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
                    add_line(
                        base_zig_zag,
                        base_zig_zag + nx * width * 0.5 - 2 * direction * zig_zag_y
                        )
                    add_line(
                        base_zig_zag + nx * width * 0.5 - 2 * direction * zig_zag_y,
                        base_zig_zag + nx * width
                        )
                
                if i_height % 2 == 0:
                    base_vert = base_zig_zag
                else:
                    base_vert = base_zig_zag + nx * width * 0.5 - 2 * direction * zig_zag_y
                
                # check if width is closed
                if (i_width < n_width) or (direction==1 and closed_width):
                    # check if height is closed
                    if not (i_height == n_height) or (not closed_height): 
                        add_line(
                            base_vert,
                            base_vert + ny * a
                            )
        
        # list of nodes from the honeycomb that are candidates for connections
        honeycomb_nodes = [self.nodes(i) for i in range(i_node_start) if self.nodes(i).is_end_node]
        
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
        
        # add connection for nodes with same positions
        if create_couplings:
            self.add_connections(honeycomb_nodes)
            
        return_set = NodeSetContainer()
        return_set['north'] = NodeSet(mpy.point, nodes=self.get_nodes_by_function(node_in_box([0,x_max], [y_max,y_max], [-1,1])))
        return_set['east'] = NodeSet(mpy.point, nodes=self.get_nodes_by_function(node_in_box([x_max,x_max], [0,y_max], [-1,1])))
        return_set['south'] = NodeSet(mpy.point, nodes=self.get_nodes_by_function(node_in_box([0,x_max], [0,0], [-1,1])))
        return_set['west'] = NodeSet(mpy.point, nodes=self.get_nodes_by_function(node_in_box([0,0], [0,y_max], [-1,1])))
        return return_set
    
    
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
        mesh_temp.add_beam_mesh_honeycomb_flat(beam_object, material, width, n_w, n_h, n_el,
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
                
        return_set = NodeSetContainer()
        return_set['bottom'] = NodeSet(mpy.point, nodes=mesh_temp.get_nodes_by_function(node_in_box([-2*x_max,2*x_max], [-2*y_max,2*y_max], [0,0])))
        return_set['top'] = NodeSet(mpy.point, nodes=mesh_temp.get_nodes_by_function(node_in_box([-2*x_max,2*x_max], [-2*y_max,2*y_max], [z_max,z_max])))

        self.add_mesh(mesh_temp)
        
        return return_set

        
        
        
        
        
        
        
        
        