# -*- coding: utf-8 -*-
"""
This module defines the Mesh class, which holds the content (nodes, elements,
sets, ...) for a meshed geometry.
"""

# python packages
import numpy as np

# meshpy imports
from . import mpy, Rotation, Function, Material, Node, Element, GeometryName, \
    GeometrySet, GeometrySetContainer, BoundaryCondition, Coupling, \
    BoundaryConditionContainer


class Mesh(object):
    """
    A class that contains a full mesh, i.e. Nodes, Elements, Boundary
    Conditions, Sets, Couplings, Materials and Functions.
    """
    
    def __init__(self):
        """ Initialize all empty containers."""
        
        self.nodes = []
        self.elements = []
        self.materials = []
        self.functions = []
        self.couplings = []
        self.geometry_sets = GeometrySetContainer()
        self.boundary_conditions = BoundaryConditionContainer()


    def add(self, *args, **kwargs):
        """
        Add an item to this mesh, depending on its type. If an list is given
        each list element is added with this function. If multiple arguments
        are given, each one is individually added with this funciton.Keyword
        arguments are passed through to the adding function.
        """
        
        if len(args) == 0:
            raise ValueError('At least one argument is required!')
        elif len(args) == 1:
            add_item = args[0]
            if isinstance(add_item, Mesh):
                self.add_mesh(add_item, **kwargs)
            elif isinstance(add_item, Function):
                self.add_function(add_item, **kwargs)
            elif isinstance(add_item, BoundaryCondition):
                self.add_bc(add_item, **kwargs)
            elif isinstance(add_item, Material):
                self.add_material(add_item, **kwargs)
            elif isinstance(add_item, Node):
                self.add_node(add_item, **kwargs)
            elif isinstance(add_item, Element):
                self.add_element(add_item, **kwargs)
            elif isinstance(add_item, GeometrySet):
                self.add_geometry_set(add_item, **kwargs)
            elif isinstance(add_item, Coupling):
                self.add_coupling(add_item, **kwargs)
            elif isinstance(add_item, list):
                for item in add_item:
                    self.add(item, **kwargs)
            else:
                raise(TypeError(
                    'No Mesh.add case implemented for type: ' + \
                    '{}!'.format(type(add_item))
                    ))
        else:
            for item in args:
                self.add(item, **kwargs)


    def add_mesh(self, mesh):
        """Add the content of another mesh to this mesh."""
        
        # Add each item from mesh to self. 
        self.add(mesh.nodes)
        self.add(mesh.elements)
        self.add(mesh.materials)
        self.add(mesh.functions)
        self.add(mesh.couplings)
        for key in self.geometry_sets.keys():
            self.add(mesh.geometry_sets[key])
        for key in self.boundary_conditions.keys():
            self.add(mesh.boundary_conditions[key])
    
    def add_coupling(self, coupling):
        """Add a coupling to the mesh object."""
        self.couplings.append(coupling)

    def add_bc(self, bc):
        """Add a boundary condition to this mesh."""
        bc_key = bc.bc_type
        geom_key = bc.geometry_set.geometry_type
        self.boundary_conditions[bc_key,geom_key].append(bc)

    def add_function(self, function):
        """
        Add a function to this mesh item. Check that the function is only added
        once.
        """
        if not function in self.functions:
            self.functions.append(function)
    
    def add_material(self, material):
        """
        Add a material to this mesh item. Check that the material is only added
        once.
        """
        if not material in self.materials:
            self.materials.append(material)
    
    def add_node(self, node):
        """Add a node to this mesh."""
        self.nodes.append(node)
        
    def add_element(self, element):
        """Add an element to this mesh."""
        self.elements.append(element)
    
    def add_geometry_set(self, geometry_set):
        """Add a geometry set to this mesh."""
        if not geometry_set in self.sets[geometry_set.geometry_type]:
            self.sets[geometry_set.geometry_type].append(geometry_set)

    def translate(self, vector):
        """
        Translate all nodes of this mesh.
        
        Args
        ----
        vector: np.array, list
            3D vector that will be added to all nodes.
        """
        for node in self.nodes:
            if not node.is_dat:
                node.coordinates += vector
    
    
    def rotate(self, rotation, origin=None, only_rotate_triads=False):
        """
        Rotate all nodes of the mesh with rotation.
        
        Args
        ----
        rotation: Rotation, list(quaternions) (nx4)
            The rotation that will be applies to the nodes. Can also be an array
            with a quaternion for each node.
        origin: 3D vector
            If this is given, the mesh is rotated about this point. Default is
            (0,0,0)
        only_rotate_triads: bool
            If true the nodal positions are not changed.
        """
        
        # get array with all quaternions and positions for the nodes
        rot1 = np.zeros([len(self.nodes),4])
        pos = np.zeros([len(self.nodes),3])
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
        if isinstance(rotation, Rotation):
            rot2 = rotation.get_quaternion()
        else:
            rot2 = rotation.transpose()
        
        # Temporary AceGen variables.
        tmp = [None for i in range(11)]
        
        # code generated with AceGen (rotation.nb)
        tmp[0]=2*10**0*rot2[1]*rot2[2]
        tmp[1]=2*10**0*rot2[0]*rot2[3]
        tmp[2]=2*10**0*rot2[0]*rot2[2]
        tmp[3]=2*10**0*rot2[1]*rot2[3]
        tmp[4]=rot2[0]**2
        tmp[5]=rot2[1]**2
        tmp[6]=rot2[2]**2
        tmp[10]=tmp[4]-tmp[6]
        tmp[7]=rot2[3]**2
        tmp[8]=2*10**0*rot2[0]*rot2[1]
        tmp[9]=2*10**0*rot2[2]*rot2[3]
        posnew[:,0]=(tmp[5]-tmp[7]+tmp[10])*pos[:,0]+(tmp[0]-tmp[1])*pos[:,1]+(tmp[2]+tmp[3])*pos[:,2]
        posnew[:,1]=(tmp[0]+tmp[1])*pos[:,0]+(tmp[4]-tmp[5]+tmp[6]-tmp[7])*pos[:,1]+(-tmp[8]+tmp[9])*pos[:,2]
        posnew[:,2]=(-tmp[2]+tmp[3])*pos[:,0]+(tmp[8]+tmp[9])*pos[:,1]+(-tmp[5]+tmp[7]+tmp[10])*pos[:,2]
        rotnew[:,0]=rot1[:,0]*rot2[0]-rot1[:,1]*rot2[1]-rot1[:,2]*rot2[2]-rot1[:,3]*rot2[3]
        rotnew[:,1]=rot1[:,1]*rot2[0]+rot1[:,0]*rot2[1]+rot1[:,3]*rot2[2]-rot1[:,2]*rot2[3]
        rotnew[:,2]=rot1[:,2]*rot2[0]-rot1[:,3]*rot2[1]+rot1[:,0]*rot2[2]+rot1[:,1]*rot2[3]
        rotnew[:,3]=rot1[:,3]*rot2[0]+rot1[:,2]*rot2[1]-rot1[:,1]*rot2[2]+rot1[:,0]*rot2[3]
        
        if not origin is None:
            posnew += origin
            
        for i, node in enumerate(self.nodes):
            if not node.is_dat:
                node.rotation.q = rotnew[i,:]
                if not only_rotate_triads:
                    node.coordinates = posnew[i,:]


    def wrap_around_cylinder(self):
        """
        Wrap the geometry around a cylinder. The y-z plane gets morphed into the
        axis of symmetry.
        """
        
        quaternions = np.zeros([len(self.nodes),4])
        pos = np.zeros([len(self.nodes),3])
        for i, node in enumerate(self.nodes):
            if not node.is_dat:
                pos[i,:] = node.coordinates
        
        # The x coordinate is the radius, the y coordinate the arc length.
        radius = pos[:,0].copy()
        phi = pos[:,1] / radius
        
        # The rotation is about the z-axis.
        quaternions[:,0] = np.cos(0.5*phi)
        quaternions[:,3] = np.sin(0.5*phi)
        
        # Set the new positions in the global array.
        pos[:,0] = radius * np.cos(phi)
        pos[:,1] = radius * np.sin(phi)
        
        # Rotate the mesh
        self.rotate(quaternions, only_rotate_triads=True)
        
        # Set the new position for the nodes.
        for i, node in enumerate(self.nodes):
            if not node.is_dat:
                node.coordinates = pos[i,:]


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
            self.add(Coupling(nodes, connection_type))
    
    
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
    

    def create_beam_mesh_line(self, beam_object, material, start_point,
            end_point, n_el=1, start_node=None ):
        """
        Generate a straight line of beam elements.
        
        Args
        ----
        beam_object: Beam
            Class of beam that will be used for this line.
        material: Material
            Material for this line.
        start_point, end_point: np.array, list
            3D-coordinates for the start and end point of the line.
        n_el: int
            Number of equally spaces beam elements along the line.
        start_node: Node
            Node to use as the first node for this line. Use this if the line
            is connected to other lines (angles have to be the same, otherwise
            connections should be used).
        """
        
        self.add_material(material)
        
        # Get geometrical values for this line.
        direction = np.array(end_point) - np.array(start_point)
        t1 = direction / np.linalg.norm(direction)
        
        # Check if the z or y axis are larger projected onto the direction.
        if abs(np.dot(t1,[0,0,1])) < abs(np.dot(t1,[0,1,0])):
            t2 = [0,0,1]
        else:
            t2 = [0,1,0]
        rotation = Rotation.from_basis(t1, t2)
        
        def get_beam_function(point_a, point_b):
            """
            Return a function for the position and rotation along the beam axis.
            """
            def position_function(xi):
                return 1/2*(1-xi)*point_a + 1/2*(1+xi)*point_b
            def rotation_function(xi):
                return rotation
            return (position_function, rotation_function)
        
        # List with nodes and elements of this line.
        elements = []
        nodes = []
        if not start_node is None:
            nodes = [start_node]
        
        # Create the beams.
        for i in range(n_el):
            
            functions = get_beam_function(
                start_point + i*direction/n_el,
                start_point + (i+1)*direction/n_el
                )
            
            if (start_node is None) and i == 0:
                first_node = None
            else:
                first_node = nodes[-1]
            elements.append(beam_object(material=material))
            nodes.extend(elements[-1].create_beam(
                functions[0], functions[1], start_node=first_node))
        
        # Set the nodes that are at the beginning and end of line (for search of
        # overlapping points)
        nodes[0].is_end_node = True
        nodes[-1].is_end_node = True
        
        # Add items to the mesh
        self.elements.extend(elements)
        self.nodes.extend(nodes)
        
        # Create geometry sets that will be returned.
        return_set = GeometryName()
        return_set['start'] = GeometrySet(mpy.point, nodes=nodes[0])
        return_set['end'] = GeometrySet(mpy.point, nodes=nodes[-1])
        return_set['line'] = GeometrySet(mpy.point, nodes=nodes)
        return return_set
    
    
    def create_beam_mesh_honeycomb_flat(
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
            geom_set = self.create_beam_mesh_line(
                beam_object,
                material,
                pointa,
                pointb,
                n_el=n_el#,
                #add_sets=False
                )
            return geom_set
        
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
            
        return_set = GeometryName()
        return_set['north'] = GeometrySet(mpy.point, nodes=self.get_nodes_by_function(node_in_box([0,x_max], [y_max,y_max], [-1,1])))
        return_set['east'] = GeometrySet(mpy.point, nodes=self.get_nodes_by_function(node_in_box([x_max,x_max], [0,y_max], [-1,1])))
        return_set['south'] = GeometrySet(mpy.point, nodes=self.get_nodes_by_function(node_in_box([0,x_max], [0,0], [-1,1])))
        return_set['west'] = GeometrySet(mpy.point, nodes=self.get_nodes_by_function(node_in_box([0,0], [0,y_max], [-1,1])))
        return return_set
    
    
    def create_beam_mesh_honeycomb(self,
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
        mesh_temp = Mesh()
        mesh_temp.create_beam_mesh_honeycomb_flat(beam_object, material, width, n_w, n_h, n_el,
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
                
        return_set = GeometryName()
        return_set['bottom'] = GeometrySet(mpy.point, nodes=mesh_temp.get_nodes_by_function(node_in_box([-2*x_max,2*x_max], [-2*y_max,2*y_max], [0,0])))
        return_set['top'] = GeometrySet(mpy.point, nodes=mesh_temp.get_nodes_by_function(node_in_box([-2*x_max,2*x_max], [-2*y_max,2*y_max], [z_max,z_max])))

        self.add_mesh(mesh_temp)
        
        return return_set

        
        
        
        
        
        
        
        
        