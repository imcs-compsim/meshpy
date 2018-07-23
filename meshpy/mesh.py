# -*- coding: utf-8 -*-
"""
This module defines the Mesh class, which holds the content (nodes, elements,
sets, ...) for a meshed geometry.
"""

# Python modules.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Meshpy modules.
from . import mpy, Rotation, Function, Material, Node, Element, \
    GeometryName, GeometrySet, GeometrySetContainer, BoundaryCondition, \
    Coupling, BoundaryConditionContainer, get_close_nodes, VTKWriter


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
                    'No Mesh.add case implemented for type: '
                    + '{}!'.format(type(add_item))
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
        self.boundary_conditions[bc_key, geom_key].append(bc)

    def add_function(self, function):
        """
        Add a function to this mesh item. Check that the function is only added
        once.
        """
        if function not in self.functions:
            self.functions.append(function)

    def add_material(self, material):
        """
        Add a material to this mesh item. Check that the material is only added
        once.
        """
        if material not in self.materials:
            self.materials.append(material)

    def add_node(self, node):
        """Add a node to this mesh."""
        self.nodes.append(node)

    def add_element(self, element):
        """Add an element to this mesh."""
        self.elements.append(element)

    def add_geometry_set(self, geometry_set):
        """Add a geometry set to this mesh."""
        if geometry_set not in self.sets[geometry_set.geometry_type]:
            self.sets[geometry_set.geometry_type].append(geometry_set)

    def get_global_coordinates(self, nodes=None):
        """
        Return an array with the coordinates of all nodes.

        Args
        ----
        nodes: list(Nodes)
            If this one is given return an array with the coordinates of the
            nodes in list, otherwise of all nodes in the mesh.
        """
        if nodes is None:
            node_list = self.nodes
        else:
            node_list = nodes
        pos = np.zeros([len(node_list), 3])
        for i, node in enumerate(node_list):
            if not node.is_dat:
                pos[i, :] = node.coordinates
            else:
                pos[i, :] = [1, 0, 0]
        return pos

    def get_global_quaternions(self):
        """Return an array with the quaternions of all nodes."""
        rot = np.zeros([len(self.nodes), 4])
        for i, node in enumerate(self.nodes):
            if (not node.is_dat) and (node.rotation is not None):
                rot[i, :] = node.rotation.get_quaternion()
            else:
                rot[i, 0] = 1
        return rot

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
            The rotation that will be applies to the nodes. Can also be an
            array with a quaternion for each node.
        origin: 3D vector
            If this is given, the mesh is rotated about this point. Default is
            (0,0,0)
        only_rotate_triads: bool
            If true the nodal positions are not changed.
        """

        # get array with all quaternions and positions for the nodes
        pos = self.get_global_coordinates()
        rot1 = self.get_global_quaternions()

        # check if origin has to be added
        if origin is not None:
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
        tmp[0]= 2*10**0*rot2[1]*rot2[2]
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
        posnew[:,0]=(tmp[5]-tmp[7]+tmp[10])*pos[:,0]+(tmp[0]-tmp[1])*pos[:,1]+(
            tmp[2]+tmp[3])*pos[:,2]
        posnew[:,1]=(tmp[0]+tmp[1])*pos[:,0]+(tmp[4]-tmp[5]+tmp[6]-tmp[7])* \
            pos[:,1]+(-tmp[8]+tmp[9])*pos[:,2]
        posnew[:,2]=(-tmp[2]+tmp[3])*pos[:,0]+(tmp[8]+tmp[9])*pos[:,1]+(
            -tmp[5]+tmp[7]+tmp[10])*pos[:,2]
        rotnew[:,0]=rot1[:,0]*rot2[0]-rot1[:,1]*rot2[1]-rot1[:,2]* \
            rot2[2]-rot1[:,3]*rot2[3]
        rotnew[:,1]=rot1[:,1]*rot2[0]+rot1[:,0]*rot2[1]+rot1[:,3]* \
            rot2[2]-rot1[:,2]*rot2[3]
        rotnew[:,2]=rot1[:,2]*rot2[0]-rot1[:,3]*rot2[1]+rot1[:,0]* \
            rot2[2]+rot1[:,1]*rot2[3]
        rotnew[:,3]=rot1[:,3]*rot2[0]+rot1[:,2]*rot2[1]-rot1[:,1]* \
            rot2[2]+rot1[:,0]*rot2[3]

        if origin is not None:
            posnew += origin

        for i, node in enumerate(self.nodes):
            if not node.is_dat:
                if node.rotation is not None:
                    node.rotation.q = rotnew[i, :]
                if not only_rotate_triads:
                    node.coordinates = posnew[i, :]

    def wrap_around_cylinder(self):
        """
        Wrap the geometry around a cylinder. The y-z plane gets morphed into
        the axis of symmetry.
        """

        pos = self.get_global_coordinates()
        quaternions = np.zeros([len(self.nodes), 4])

        # The x coordinate is the radius, the y coordinate the arc length.
        radius = pos[:, 0].copy()
        phi = pos[:, 1] / radius

        # The rotation is about the z-axis.
        quaternions[:, 0] = np.cos(0.5 * phi)
        quaternions[:, 3] = np.sin(0.5 * phi)

        # Set the new positions in the global array.
        pos[:, 0] = radius * np.cos(phi)
        pos[:, 1] = radius * np.sin(phi)

        # Rotate the mesh
        self.rotate(quaternions, only_rotate_triads=True)

        # Set the new position for the nodes.
        for i, node in enumerate(self.nodes):
            if not node.is_dat:
                node.coordinates = pos[i, :]

    def couple_nodes(self, nodes=None, coupling_type=mpy.coupling_fix):
        """
        Search through nodes and connect all nodes with the same coordinates.
        """

        # Get list of partner nodes
        partner_nodes = self.get_close_nodes(nodes=nodes)

        # Connect close nodes with a coupling.
        for node_list in partner_nodes:
            self.add(Coupling(node_list, coupling_type))

    def get_nodes_by_function(self, function, *args, middle_nodes=False):
        """
        Return all nodes for which the function evaluates to true.

        Args
        ----
        function: function(node)
            Nodes for which this function is true are returned.
        middle_nodes: bool
            If this is true, middle nodes of a beam are also returned.
        """

        node_list = []
        for node in self.nodes:
            if middle_nodes or (not node.is_middle_node):
                if function(node, *args):
                    node_list.append(node)
        return node_list

    def get_min_max_nodes(self, nodes=None):
        """
        Return a geometry set with the max and min nodes in all directions.

        Args
        ----
        nodes: list(Nodes)
            If this one is given return an array with the coordinates of the
            nodes in list, otherwise of all nodes in the mesh.
        """

        geometry = GeometryName()

        if nodes is None:
            node_list = self.nodes
        else:
            node_list = nodes
        pos = self.get_global_coordinates(nodes=node_list)
        for i, direction in enumerate(['x', 'y', 'z']):
            # Check if there is more than one value in dimension.
            min_max = [np.min(pos[:, i]), np.max(pos[:, i])]
            if np.abs(min_max[1] - min_max[0]) >= mpy.eps_pos:
                for j, text in enumerate(['min', 'max']):
                    # get all nodes with the min / max coordinate
                    min_max_nodes = []
                    for index, value in enumerate(
                            np.abs(pos[:, i] - min_max[j]) < mpy.eps_pos
                            ):
                        if value:
                            min_max_nodes.append(node_list[index])
                    geometry['{}_{}'.format(direction, text)] = GeometrySet(
                        mpy.point,
                        min_max_nodes
                        )
        return geometry

    def get_close_nodes(self, nodes=None, eps=mpy.eps_pos):
        """
        Find nodes that are close to each other.

        Args
        ----
        nodes: list(Node)
            If this argument is given, the closest nodes within this list are
            returned, otherwise all nodes in the mesh are checked.
        eps: double
            Spherical value that the nodes have to be within, to be identified
            as overlapping.

        Return
        ----
        partner_nodes: list(list(Node))
            A list of lists with partner nodes.
        """

        # Check if input argument was given
        if nodes is None:
            node_list = [node for node in self.nodes if not node.is_dat]
        else:
            node_list = nodes

        return get_close_nodes(node_list)

    def preview_python(self):
        """Display the elements in this mesh in matplotlib."""

        # Create figure.
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Loop over elements.
        for element in self.elements:
            if not element.is_dat:
                element.preview_python(ax)

        # Finish plot.
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def write_vtk(self, filepath, **kwargs):
        """Write the contents of this mesh to a VTK file."""

        # Object to store VKT data and write it to file.
        vtkwriter = VTKWriter()

        # Get representation of elements.
        for element in self.elements:
            element.get_vtk(vtkwriter)

        # Write to file
        vtkwriter.write_vtk(filepath, **kwargs)

    def create_beam_mesh_curve(self, beam_object, material, function, interval,
            n_el=1, function_rotation=None):
        """
        Generate a beam from a parametric curve. Integration along the beam is
        performed with scipy, and the gradient is calculated with autograd.

        Args
        ----
        beam_object: Beam
            Class of beam that will be used for this line.
        material: Material
            Material for this line.
        function: function
            3D-parametric curve that represents the beam axis. If only a 2D
            point is returned, the triad creation is simplified. If
            mathematical functions are used, they have to come from the wrapper
            autograd.numpy.
        interval: [start end]
            Start and end values for the parameter of the curve.
        n_el: int
            Number of equally spaces beam elements along the line.
        function_rotation: function
            If this argument is given, the triads are computed with this
            function, on the same interval as the position function. Must
            return a Rotation object.
        """

        # Packages for AD and numerical integration.
        from autograd import jacobian
        import autograd.numpy as npAD
        import scipy.integrate as integrate
        import scipy.optimize as optimize

        # Check size of position function
        if len(function(interval[0])) == 2:
            is_3d_curve = False
        elif len(function(interval[0])) == 3:
            is_3d_curve = True
        else:
            raise ValueError('Function must return either 2d or 3d curve!')

        # Check rotation function.
        if function_rotation is None:
            is_rot_funct = False
        else:
            is_rot_funct = True

        # Check that the position is an np.array
        if not isinstance(function(float(interval[0])), np.ndarray):
            raise TypeError(
                'Function must be of type np.ndarray, got {}!'.format(
                    type(function(float(interval[0])))
                    ))

        # Get the derivative of the position function and the increment along
        # the curve.
        rp = jacobian(function)

        def ds(t):
            """Increment along the curve."""
            return npAD.linalg.norm(rp(t))

        def S(t, start_t=None, start_S=None):
            """
            Function that integrates the length until a parameter value.
            A speedup can be achieved by giving start_t and start_S, the
            parameter and Length at a known point.
            """
            if start_t is None and start_S is None:
                st = interval[0]
                sS = 0
            elif start_t is not None and start_S is not None:
                st = start_t
                sS = start_S
            else:
                raise ValueError('Input parameters are wrong!')
            return integrate.quad(ds, st, t)[0] + sS

        def get_t_along_curve(arc_length, t0, **kwargs):
            """
            Calculate the parameter t where the length along the curve is
            arc_length. t0 is the start point for the newton iteration.
            """
            t_root = optimize.newton(lambda t: S(t, **kwargs) - arc_length, t0,
                fprime=ds)
            return t_root

        def get_beam_functions(length_a, length_b):
            """
            Return a function for the position and rotation along the beam
            axis.
            """

            # Length of the beam element in physical space.
            L = length_b - length_a

            def beam_function(xi):
                """
                Return position and rotation along the beam in the parameter
                coordinate xi.
                """

                # Global values for the start of the element.
                global t_temp, t_start_element, t2_temp

                # Parameter for xi.
                t = get_t_along_curve(
                    length_a + 0.5 * (xi + 1) * L,
                    t_start_element, start_t=t_start_element, start_S=length_a)
                t_temp = t

                # Position at xi.
                if is_3d_curve:
                    pos = function(t)
                else:
                    pos = np.zeros(3)
                    pos[:2] = function(t)

                # Rotation at xi.
                if is_rot_funct:
                    rot = function_rotation(t)
                else:
                    if is_3d_curve:
                        rot = Rotation.from_basis(rp(t), t2_temp)
                    else:
                        # The rotation simplifies in the 2d case.
                        rprime = rp(t)
                        rot = Rotation(
                            [0, 0, 1],
                            np.arctan2(rprime[1], rprime[0])
                            )

                # Return the needed values for beam creation.
                return (pos, rot)

            return beam_function

        # Now create the beam.
        # Get the length of the whole segment.
        length = S(interval[1])

        # Add the beam elements
        self.add_material(material)

        # List with nodes and elements of this line.
        elements = []
        nodes = []

        # Create the beams.
        global t_temp, t_start_element, t2_temp
        t_temp = interval[0]
        t_start_element = interval[0]

        # The first t2 basis is the one with the larger projection on rp.
        if is_3d_curve:
            rprime = rp(float(interval[0]))
            if abs(np.dot(rprime, [0, 0, 1])) < \
                    abs(np.dot(rprime, [0, 1, 0])):
                t2_temp = [0, 0, 1]
            else:
                t2_temp = [0, 1, 0]

        for i in range(n_el):
            function_pos_rot = get_beam_functions(
                length * i / n_el,
                length * (i + 1) / n_el
                )
            if i == 0:
                first_node = None
            else:
                first_node = nodes[-1]
            elements.append(beam_object(material=material))
            nodes.extend(elements[-1].create_beam(function_pos_rot,
                start_node=first_node))

            # Reset the start value of t and t2.
            t_start_element = t_temp
            t2_temp = nodes[-1].rotation.get_rotation_matrix()[:, 1]

        # Set the nodes that are at the beginning and end of line (for search
        # of overlapping points)
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

    def create_beam_mesh_line(self, beam_object, material, start_point,
            end_point, n_el=1, start_node=None):
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
        start_node: Node, GeometrySet
            Node to use as the first node for this line. Use this if the line
            is connected to other lines (angles have to be the same, otherwise
            connections should be used). If a geometry set is given, it can
            contain one, and one node only.
        """

        self.add_material(material)

        # Get geometrical values for this line.
        direction = np.array(end_point) - np.array(start_point)
        t1 = direction / np.linalg.norm(direction)

        # Check if the z or y axis are larger projected onto the direction.
        if abs(np.dot(t1, [0, 0, 1])) < abs(np.dot(t1, [0, 1, 0])):
            t2 = [0, 0, 1]
        else:
            t2 = [0, 1, 0]
        rotation = Rotation.from_basis(t1, t2)

        def get_beam_geometry(point_a, point_b):
            """
            Return a function for the position and rotation along the beam
            axis.
            """
            def beam_function(xi):
                return (
                    0.5 * (1 - xi) * point_a + 0.5 * (1 + xi) * point_b,
                    rotation
                    )
            return beam_function

        # List with nodes and elements of this line.
        elements = []
        nodes = []
        if start_node is not None:
            # Check type of start node.
            if isinstance(start_node, Node):
                nodes = [start_node]
            elif isinstance(start_node, GeometrySet):
                # Check if there is only one node in the set
                if len(start_node.nodes) == 1:
                    nodes = [start_node.nodes[0]]
                else:
                    raise ValueError('GeometrySet does not have one node!')
            else:
                raise TypeError('start_node can be node or GeometrySet '
                    + 'got "{}"!'.format(type(start_node)))

        # Create the beams.
        for i in range(n_el):

            function = get_beam_geometry(
                start_point + i * direction / n_el,
                start_point + (i + 1) * direction / n_el
                )

            if (start_node is None) and i == 0:
                first_node = None
            else:
                first_node = nodes[-1]
            elements.append(beam_object(material=material))
            nodes.extend(elements[-1].create_beam(function,
                start_node=first_node))

        # Set the nodes that are at the beginning and end of line (for search
        # of overlapping points)
        nodes[0].is_end_node = True
        nodes[-1].is_end_node = True

        # Add items to the mesh
        self.elements.extend(elements)
        if start_node is None:
            self.nodes.extend(nodes)
        else:
            self.nodes.extend(nodes[1:])

        # Create geometry sets that will be returned.
        return_set = GeometryName()
        return_set['start'] = GeometrySet(mpy.point, nodes=nodes[0])
        return_set['end'] = GeometrySet(mpy.point, nodes=nodes[-1])
        return_set['line'] = GeometrySet(mpy.point, nodes=nodes)
        return return_set

    def create_beam_mesh_honeycomb_flat(self, beam_object, material, width,
            n_width, n_height, n_el=1, closed_width=True, closed_height=True,
            create_couplings=True):
        """
        Add a flat honeycomb structure. The structure will be created in the
        x-y plane.

        Args
        ----
        beam_object: Beam
            Object that will be used to create the beam elements.
        material: Material
            Material for the beam.
        width: float
            Width of one honeycomb.
        n_width: int
            Number of honeycombs in x-direction.
        n_height: int
            Number of honeycombs in y-direction.
        n_el: int
            Number of elements per beam line.
        closed_width: bool
            If the last honeycombs in x-direction will be closed.
        closed_height: bool
            If the last vertical lines in y-direction will be created.
        create_couplings: bool
            If the nodes will be connected in this function.

        Return
        ----
        return_set: GeometryName
            Set with nodes on the north, south, east and west boundaries. This
            set only contains end nodes of lines.
        """

        def add_line(pointa, pointb):
            """Shortcut to add line."""
            return self.create_beam_mesh_line(
                beam_object,
                material,
                pointa,
                pointb,
                n_el=n_el
                )

        # Geometrical shortcuts.
        sin30 = np.sin(np.pi / 6)
        cos30 = np.sin(2 * np.pi / 6)
        a = width * 0.5 / cos30
        nx = np.array([1., 0., 0.])
        ny = np.array([0., 1., 0.])
        zig_zag_x = nx * width * 0.5
        zig_zag_y = ny * a * sin30 * 0.5

        # Create the honeycomb structure
        origin = np.array([0, a * 0.5 * sin30, 0])
        i_node_start = len(self.nodes)
        for i_height in range(n_height + 1):

            # Start point for this zig-zag line.
            base_row = origin + (2 * zig_zag_y + a * ny) * i_height

            # If the first node is up or down of base_row.
            if i_height % 2 == 0:
                direction = 1
            else:
                direction = -1

            for i_width in range(n_width + 1):
                base_zig_zag = (base_row + direction * zig_zag_y
                    + width * i_width * nx)

                # Do not add a zig-zag line on the last run (that one is only
                # for the remaining vertical lines).
                if i_width < n_width:
                    add_line(
                        base_zig_zag,
                        base_zig_zag + zig_zag_x - 2 * direction * zig_zag_y
                        )
                    add_line(
                        base_zig_zag + zig_zag_x - 2 * direction * zig_zag_y,
                        base_zig_zag + nx * width
                        )

                # Check where the vertical lines start.
                if i_height % 2 == 0:
                    base_vert = base_zig_zag
                else:
                    base_vert = base_zig_zag + zig_zag_x - \
                        2 * direction * zig_zag_y

                # Only add vertical lines at the end if closed_width.
                if (i_width < n_width) or (direction == 1 and closed_width):
                    # Check if the vertical lines at the top should be added.
                    if not (i_height == n_height) or (not closed_height):
                        add_line(
                            base_vert,
                            base_vert + ny * a
                            )

        # List of nodes from the honeycomb that are candidates for connections.
        honeycomb_nodes = [
            self.nodes[i] for i in range(i_node_start, len(self.nodes))
            if self.nodes[i].is_end_node
            ]

        # Add connections for the nodes with same positions.
        if create_couplings:
            self.couple_nodes(nodes=honeycomb_nodes)

        # Get min and max nodes of the honeycomb.
        min_max_nodes = self.get_min_max_nodes(nodes=honeycomb_nodes)

        # Return the geometry set.
        return_set = GeometryName()
        return_set['north'] = min_max_nodes['y_max']
        return_set['east'] = min_max_nodes['x_max']
        return_set['south'] = min_max_nodes['y_min']
        return_set['west'] = min_max_nodes['x_min']
        return return_set

    def create_beam_mesh_honeycomb(self, beam_object, material, diameter,
            n_circumference, n_axis, n_el=1, closed_top=True, vertical=True):
        """
        Add a honeycomb structure around a cylinder. The cylinder axis will be
        the z-axis.

        Args
        ----
        beam_object: Beam
            Object that will be used to create the beam elements.
        material: Material
            Material for the beam.
        diameter: float
            Diameter of the cylinder.
        n_circumference: int
            Number of honeycombs around the diameter. If vertical is False this
            has to be an odd number.
        n_axis: int
            Number of honeycombs in axial-direction.
        n_el: int
            Number of elements per beam line.
        closed_top: bool
            If the last honeycombs in axial-direction will be closed.
        vertical: bool
            If there are vertical lines in the honeycomb or horizontal.

        Return
        ----
        return_set: GeometryName
            Set with nodes top and bottom nodes. This will only contains end
            nodes of lines.
        """

        # Calculate the input values for the flat honeycomb mesh.
        if vertical:
            width = diameter * np.pi / n_circumference
            closed_width = False
            closed_height = closed_top
            rotation = Rotation([0, 0, 1],
                np.pi / 2) * Rotation([1, 0, 0], np.pi / 2)
            n_height = n_axis
            n_width = n_circumference
        else:
            if not n_circumference % 2 == 0:
                raise ValueError('There has to be an even number of '
                    + 'elements along the diameter in horizontal mode.'
                    + ' Given {}!'.format(n_circumference))
            H = diameter * np.pi / n_circumference
            r = H / (1 + np.sin(np.pi / 6))
            width = 2 * r * np.cos(np.pi / 6)
            closed_width = closed_top
            closed_height = False
            rotation = Rotation([0, 1, 0], -0.5 * np.pi)
            n_height = n_circumference - 1
            n_width = n_axis

        # Create the flat mesh, do not create couplings, as they will be added
        # later in this function, where also the diameter nodes will be
        # connected.
        mesh_temp = Mesh()
        honeycomb_sets = mesh_temp.create_beam_mesh_honeycomb_flat(beam_object,
            material, width, n_width, n_height, n_el=n_el,
            closed_width=closed_width, closed_height=closed_height,
            create_couplings=False)

        # Move the mesh to the correct position.
        mesh_temp.rotate(rotation)
        mesh_temp.translate([diameter / 2, 0, 0])
        mesh_temp.wrap_around_cylinder()

        # Add connections to the mesh.
        mesh_temp.couple_nodes()

        # Return the geometry set'
        return_set = GeometryName()
        if vertical:
            return_set['top'] = honeycomb_sets['north']
            return_set['bottom'] = honeycomb_sets['south']
        else:
            return_set['top'] = honeycomb_sets['east']
            return_set['bottom'] = honeycomb_sets['west']

        # Add to this mesh
        self.add_mesh(mesh_temp)

        return return_set
