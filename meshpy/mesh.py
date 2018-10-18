# -*- coding: utf-8 -*-
"""
This module defines the Mesh class, which holds the content (nodes, elements,
sets, ...) for a meshed geometry.
"""

# Python modules.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from _collections import OrderedDict

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
            elif isinstance(add_item, GeometryName):
                self.add_geometry_name(add_item, **kwargs)
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
        if geometry_set not in self.geometry_sets[geometry_set.geometry_type]:
            self.geometry_sets[geometry_set.geometry_type].append(geometry_set)

    def add_geometry_name(self, geometry_name):
        """Add a set of geometry sets to this mesh."""
        for _key, value in geometry_name.items():
            self.add(value)

    def get_unique_geometry_sets(self, link_nodes=False):
        """
        Return a geometry set container that contains all geometry sets
        explicitly added to the mesh, as well as all sets from boundary
        conditions and couplings.
        After all the sets are gathered, each sets tells its nodes that they
        are part of the set (mainly for vtk output).
        """

        if link_nodes:
            # First clear all links in existing nodes.
            for node in self.nodes:
                node.node_sets_link = []

        # Make a copy of the sets in this mesh.
        mesh_sets = self.geometry_sets.copy()

        # Add sets from boundary conditions and couplings.
        for (_bc_key, geom_key), bc_list in self.boundary_conditions.items():
            for bc in bc_list:
                if not bc.is_dat:
                    mesh_sets[geom_key].append(bc.geometry_set)
        for coupling in self.couplings:
            mesh_sets[coupling.node_set.geometry_type].append(
                coupling.node_set)

        for key in mesh_sets.keys():
            # Remove double node sets in the container.
            mesh_sets[key] = list(OrderedDict.fromkeys(mesh_sets[key]))

            for i, geometry_set in enumerate(mesh_sets[key]):
                # Add global indices to the geometry set.
                geometry_set.n_global = i + 1

                if link_nodes and not geometry_set.is_dat:
                    for node in geometry_set:
                        node.node_sets_link.append(geometry_set)

        # Set the global value for digits in the VTK output.
        if link_nodes:

            # Get highest number of node_sets.
            max_sets = max([
                len(geometry_list) for geometry_list in mesh_sets.values()
                ])

            # Set the mpy value.
            digits = len(str(max_sets))
            mpy.vtk_node_set_format = '{:0' + str(digits) + '}'

        return mesh_sets

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

        # Get array with all quaternions and positions for the nodes.
        pos = self.get_global_coordinates()
        rot1 = self.get_global_quaternions()

        # Check if origin has to be added.
        if origin is not None:
            pos -= origin

        # New arrays.
        rotnew = np.zeros_like(rot1)
        posnew = np.zeros_like(pos)

        # Additional rotation.
        if isinstance(rotation, Rotation):
            rot2 = rotation.get_quaternion()
        else:
            rot2 = rotation.transpose()

        # Temporary AceGen variables.
        tmp = [None for i in range(11)]

        # Code generated with AceGen (rotation.nb).
        tmp[0] = 2 * 10**0 * rot2[1] * rot2[2]
        tmp[1] = 2 * 10**0 * rot2[0] * rot2[3]
        tmp[2] = 2 * 10**0 * rot2[0] * rot2[2]
        tmp[3] = 2 * 10**0 * rot2[1] * rot2[3]
        tmp[4] = rot2[0]**2
        tmp[5] = rot2[1]**2
        tmp[6] = rot2[2]**2
        tmp[10] = tmp[4] - tmp[6]
        tmp[7] = rot2[3]**2
        tmp[8] = 2 * 10**0 * rot2[0] * rot2[1]
        tmp[9] = 2 * 10**0 * rot2[2] * rot2[3]
        posnew[:, 0] = (tmp[5] - tmp[7] + tmp[10]) * pos[:, 0] + (tmp[0]
            - tmp[1]) * pos[:, 1] + (tmp[2] + tmp[3]) * pos[:, 2]
        posnew[:, 1] = (tmp[0] + tmp[1]) * pos[:, 0] + (tmp[4] - tmp[5]
            + tmp[6] - tmp[7]) * pos[:, 1] + (-tmp[8] + tmp[9]) * pos[:, 2]
        posnew[:, 2] = (-tmp[2] + tmp[3]) * pos[:, 0] + (tmp[8] + tmp[9]) * \
            pos[:, 1] + (-tmp[5] + tmp[7] + tmp[10]) * pos[:, 2]
        rotnew[:, 0] = rot1[:, 0] * rot2[0] - rot1[:, 1] * rot2[1] - \
            rot1[:, 2] * rot2[2] - rot1[:, 3] * rot2[3]
        rotnew[:, 1] = rot1[:, 1] * rot2[0] + rot1[:, 0] * rot2[1] + \
            rot1[:, 3] * rot2[2] - rot1[:, 2] * rot2[3]
        rotnew[:, 2] = rot1[:, 2] * rot2[0] - rot1[:, 3] * rot2[1] + \
            rot1[:, 0] * rot2[2] + rot1[:, 1] * rot2[3]
        rotnew[:, 3] = rot1[:, 3] * rot2[0] + rot1[:, 2] * rot2[1] - \
            rot1[:, 1] * rot2[2] + rot1[:, 0] * rot2[3]

        if origin is not None:
            posnew += origin

        for i, node in enumerate(self.nodes):
            if not node.is_dat:
                if node.rotation is not None:
                    node.rotation.q = rotnew[i, :]
                if not only_rotate_triads:
                    node.coordinates = posnew[i, :]

    def reflect(self, normal_vector, origin=None):
        """
        Reflect all nodes of the mesh with respect to a plane defined by its
        normal_vector. Per default the plane goes through the origin, if not
        a point on the plane can be given with the parameter origin.

        For the reflection we assume that e1' and e2' are mirrored with respect
        to the original frame and e3' is in the opposite direction than the
        mirrored e3.

        With the defined mirroring strategy, the quaternion to be applied on
        the existing rotations can be calculated the following way:
            q[0] = e3 * n
            q[1,2,3] = e3 x n
        This constructs a rotation with the rotation axis on the plane, and
        normal to the vector e3. The rotation angle is twice the angle of e3
        to n.

        Args
        ----
        normal_3D vector
            The normal vector of the reflection plane.
        origin: 3D vector
            Per default the reflection plane goes through the origin. If this
            parameter is given, the point is on the plane.
        """

        # Normalize the normal vector.
        normal_vector = np.array(normal_vector / np.linalg.norm(normal_vector))

        # Get array with all quaternions and positions for the nodes.
        pos = self.get_global_coordinates()
        rot1 = self.get_global_quaternions()

        # Check if origin has to be added.
        if origin is not None:
            pos -= origin

        # Get the reflection matrix A.
        A = np.eye(3) - 2. * np.dot(
            np.transpose(np.asmatrix(normal_vector)),
            np.asmatrix(normal_vector)
            )

        # Calculate the new positions.
        pos_new = np.array(pos * A)

        # Move back from the origin.
        if origin is not None:
            pos_new += origin

        # Set the new positions.
        for i, node in enumerate(self.nodes):
            if not node.is_dat:
                node.coordinates = pos_new[i, :]

        # First get all e3 vectors of the nodes.
        e3 = np.zeros_like(pos)
        e3[:, 0] = 2 * (rot1[:, 0] * rot1[:, 2] + rot1[:, 1] * rot1[:, 3])
        e3[:, 1] = 2 * (-1 * rot1[:, 0] * rot1[:, 1] + rot1[:, 2] * rot1[:, 3])
        e3[:, 2] = rot1[:, 0]**2 - rot1[:, 1]**2 - rot1[:, 2]**2 + \
            rot1[:, 3]**2

        # Get the dot and cross product of e3 and the normal vector.
        rot_new = np.zeros_like(rot1)
        rot_new[:, 0] = np.dot(e3, normal_vector)
        rot_new[:, 1:] = np.cross(e3, normal_vector)

        # Rotate the triads.
        self.rotate(rot_new, only_rotate_triads=True)

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

    def get_close_nodes(self, nodes=None, **kwargs):
        """
        Find nodes that are close to each other.

        Args
        ----
        nodes: list(Node)
            If this argument is given, the closest nodes within this list are
            returned, otherwise all nodes in the mesh are checked.
        kwargs:
            Keyword arguments for get_close_nodes.

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

        return get_close_nodes(node_list, **kwargs)

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

    def write_vtk(self, output_name='meshpy', output_directory='', **kwargs):
        """Write the contents of this mesh to a VTK file."""

        # Object to store VKT data and write it to file.
        vtkwriter_beam = VTKWriter()
        vtkwriter_solid = VTKWriter()

        # Get the set numbers of the mesh
        self.get_unique_geometry_sets(link_nodes=True)

        # Get representation of elements.
        for element in self.elements:
            element.get_vtk(vtkwriter_beam, vtkwriter_solid)

        # Write to file, only if there is at least one point in the writer.
        if (vtkwriter_beam.points.GetNumberOfPoints() > 0):
            filepath = os.path.join(
                output_directory,
                output_name + '_beam.vtu'
                )
            vtkwriter_beam.write_vtk(filepath, **kwargs)
        if (vtkwriter_solid.points.GetNumberOfPoints() > 0):
            filepath = os.path.join(
                output_directory,
                output_name + '_solid.vtu'
                )
            vtkwriter_solid.write_vtk(filepath, **kwargs)

    def create_beam_mesh_function(self, *, beam_object=None, material=None,
            function_generator=None, interval=[0, 1], n_el=1, add_sets=False,
            start_node=None, end_node=None):
        """
        Generic beam creation function.

        Args
        ----
        beam_object: Beam
            Class of beam that will be used for this line.
        material: Material
            Material for this line.
        function_generator: function that returns function
            The function_generator has to take two variables, point_a and
            point_b (both within the interval) and return a function(xi) that
            calculates the position and rotation along the beam, where
            point_a -> xi = -1 and point_b -> xi = 1.
        interval: [start end]
            Start and end values for interval that will be used to create the
            beam.
        n_el: int
            Number of equally spaces beam elements along the line.
        add_sets: bool
            If this is true the sets are added to the mesh and then displayed
            in eventual VTK output, even if they are not used for a boundary
            condition or coupling.
        start_node: Node, GeometrySet
            Node to use as the first node for this line. Use this if the line
            is connected to other lines (angles have to be the same, otherwise
            connections should be used). If a geometry set is given, it can
            contain one, and one node only. If the rotation does not match, but
            the tangent vector is the same, the created beams triads are
            rotated so the physical problem stays the same (for axi-symmetric
            beam cross-sections) but the same nodes can be used.
        end_node: Node, GeometrySet, bool
            If this is a Node or GeometrySet, the last node of the created beam
            is set to that node.
            If it is True the created beam is closed within itself.

        Return
        ----
        return_set: GeometryName
            Set with the 'start' and 'end' node of the curve. Also a 'line' set
            with all nodes of the curve.
        """

        # Make sure the material is in the mesh.
        self.add_material(material)

        # List with nodes and elements that will be added in the creation of
        # this beam.
        elements = []
        nodes = []

        def get_node(item, name):
            """
            Function to get a node from the input variable. This function
            accepts a Node object as well as a GeometrySet object.
            """
            if isinstance(item, Node):
                return item
            elif isinstance(item, GeometrySet):
                # Check if there is only one node in the set
                if len(item.nodes) == 1:
                    return item.nodes[0]
                else:
                    raise ValueError('GeometrySet does not have one node!')
            else:
                raise TypeError(('{} can be node or GeometrySet '
                    + 'got "{}"!').format(name, type(item)))

        # If a start node is given, set this as the first node for this beam.
        if start_node is not None:
            nodes = [get_node(start_node, 'start_node')]

        # If an end node is given, check what behavior is wanted.
        close_beam = False
        if end_node is True:
            close_beam = True
        elif end_node is not None:
            end_node = get_node(end_node, 'end_node')

        # Create the beams.
        for i in range(n_el):

            # If the beam is closed with itself, set the end node to be the
            # first node of the beam. This is done when the second element is
            # created, as the first node already exists here.
            if i == 1 and close_beam:
                end_node = nodes[0]

            # Get the function to create this beam element.
            function = function_generator(
                interval[0] + i * (interval[1] - interval[0]) / n_el,
                interval[0] + (i + 1) * (interval[1] - interval[0]) / n_el
                )

            # Set the start node for the created beam.
            if start_node is not None or i > 0:
                first_node = nodes[-1]
            else:
                first_node = None

            # If an end node is given, set this one for the last element.
            if end_node is not None and i == n_el - 1:
                last_node = end_node
            else:
                last_node = None

            element = beam_object(material=material)
            elements.append(element)
            nodes.extend(element.create_beam(function, start_node=first_node,
                end_node=last_node))

        # Add items to the mesh
        self.elements.extend(elements)
        if start_node is None:
            self.nodes.extend(nodes)
        else:
            self.nodes.extend(nodes[1:])

        # Set the last node of the beam.
        if end_node is None:
            end_node = nodes[-1]

        # Set the nodes that are at the beginning and end of line (for search
        # of overlapping points)
        nodes[0].is_end_node = True
        end_node.is_end_node = True

        # Create geometry sets that will be returned.
        return_set = GeometryName()
        return_set['start'] = GeometrySet(mpy.point, nodes=nodes[0])
        return_set['end'] = GeometrySet(mpy.point, nodes=end_node)
        return_set['line'] = GeometrySet(mpy.line, nodes=nodes)
        if add_sets:
            self.add(return_set)
        return return_set
