# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# MeshPy: A beam finite element input generator
#
# MIT License
#
# Copyright (c) 2021 Ivo Steinbrecher
#                    Institute for Mathematics and Computer-Based Simulation
#                    Universitaet der Bundeswehr Muenchen
#                    https://www.unibw.de/imcs-en
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------
"""
This module defines the Mesh class, which holds the content (nodes, elements,
sets, ...) for a meshed geometry.
"""

# Python modules.
import numpy as np
import os
import warnings
import copy

# Meshpy modules.
from .conf import mpy
from .rotation import Rotation, add_rotations
from .function import Function
from .material import Material
from .node import Node
from .element_beam import Beam
from .geometry_set import GeometrySet
from .container import (GeometryName, GeometrySetContainer,
    BoundaryConditionContainer)
from .boundary_condition import BoundaryConditionBase
from .coupling import Coupling
from .vtk_writer import VTKWriter
from .utility import (find_close_points, find_close_nodes,
    partner_indices_to_point_partners)


class Mesh(object):
    """
    A class that contains a full mesh, i.e. Nodes, Elements, Boundary
    Conditions, Sets, Couplings, Materials and Functions.
    """

    def __init__(self):
        """ Initialize all empty containers."""

        self.nodes = []
        self.elements = []
        self.elements_fluid = []
        self.materials = []
        self.functions = []
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
            elif isinstance(add_item, BoundaryConditionBase):
                self.add_bc(add_item, **kwargs)
            elif isinstance(add_item, Material):
                self.add_material(add_item, **kwargs)
            elif isinstance(add_item, Node):
                self.add_node(add_item, **kwargs)
            elif isinstance(add_item, Beam):
                self.add_element(add_item, **kwargs)
            elif isinstance(add_item, GeometrySet):
                self.add_geometry_set(add_item, **kwargs)
            elif isinstance(add_item, GeometryName):
                self.add_geometry_name(add_item, **kwargs)
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
        for key in self.geometry_sets.keys():
            self.add(mesh.geometry_sets[key])
        for key in self.boundary_conditions.keys():
            self.add(mesh.boundary_conditions[key])

    def add_bc(self, bc):
        """Add a boundary condition to this mesh."""
        bc_key = bc.bc_type
        geom_key = bc.geometry_set.geometry_type
        bc.geometry_set.check_replaced_nodes()
        if bc in self.boundary_conditions[bc_key, geom_key]:
            raise ValueError('The boundary condition is already in this mesh!')
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
        if node in self.nodes:
            raise ValueError('The node is already in this mesh!')
        self.nodes.append(node)

    def add_element(self, element):
        """Add an element to this mesh."""
        if element in self.elements:
            raise ValueError('The element is already in this mesh!')
        self.elements.append(element)

    def add_geometry_set(self, geometry_set):
        """Add a geometry set to this mesh."""
        if geometry_set in self.geometry_sets[geometry_set.geometry_type]:
            raise ValueError('The geometry set is already in this mesh')
        geometry_set.check_replaced_nodes()
        self.geometry_sets[geometry_set.geometry_type].append(geometry_set)

    def add_geometry_name(self, geometry_name):
        """Add a set of geometry sets to this mesh."""
        for _key, value in geometry_name.items():
            self.add(value)

    def replace_node(self, old_node, new_node):
        """Replace the first node with the second node."""

        # Check that the new node is in mesh.
        if new_node not in self.nodes:
            raise ValueError('The new node is not in the mesh!')

        for i, node in enumerate(self.nodes):
            if node == old_node:
                del self.nodes[i]
                break
        else:
            raise ValueError('The node that should be replaced is not in the '
                + 'mesh')

    def get_unique_geometry_sets(self, coupling_sets=True, link_nodes=False):
        """
        Return a geometry set container that contains geometry sets explicitly
        added to the mesh, as well as sets for boundary conditions.

        Args
        ----
        coupling_sets: bool
            If this is true, also sets for couplings will be added.
        link_nodes: bool
            If a link to the geometry sets should be added to each connected
            node (this option is mainly for vtk output).
        """

        if link_nodes:
            # First clear all links in existing nodes.
            for node in self.nodes:
                node.node_sets_link = []

        # Make a copy of the sets in this mesh.
        mesh_sets = self.geometry_sets.copy()

        # Add sets from boundary conditions.
        for (bc_key, geom_key), bc_list in self.boundary_conditions.items():
            for bc in bc_list:
                if not bc.is_dat:
                    # Check if sets from couplings should be added.
                    is_coupling = (bc_key == mpy.bc.point_coupling or
                        bc_key == mpy.bc.point_coupling_penalty)
                    if (is_coupling and coupling_sets) or (not is_coupling):
                        # Only add set if it is not already in the container.
                        # For example if multiple Neumann boundary conditions
                        # are applied on the same node set.
                        if bc.geometry_set not in mesh_sets[geom_key]:
                            mesh_sets[geom_key].append(bc.geometry_set)

        for key in mesh_sets.keys():
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

    def set_node_links(self):
        """
        Create a link of all elements to the nodes connected to them. Also add
        a link to this mesh.
        """
        for element in self.elements:
            if not element.is_dat:
                for node in element.nodes:
                    node.element_link.append(element)
        for node in self.nodes:
            if not node.is_dat:
                node.mesh = self

        # Add a link to the couplings. For now the implementation only allows
        # one coupling per node.
        for coupling_type in [mpy.bc.point_coupling,
                mpy.bc.point_coupling_penalty]:
            for coupling in self.boundary_conditions[
                    coupling_type, mpy.geo.point]:
                if not coupling.is_dat:
                    for node in coupling.geometry_set.nodes:
                        if node.coupling_link is None:
                            node.coupling_link = coupling
                        else:
                            raise ValueError('It is currently not possible to '
                                + 'add more than one coupling to a node.')

    def get_global_nodes(self, *, nodes=None, include_solid_nodes=False,
            middle_nodes=True):
        """
        Return a list with the global beam nodes. If in the future we also want
        to perform translate / rotate / cylinder wrapping / ... on solid
        elements this can be implemented here.

        Args
        ----
        nodes: list(Nodes)
            If this list is given it will be returned as is.
        include_solid_nodes: bool
            If solid nodes should be included. This only works if the solid
            mesh is imported as a full mesh and not just the lines in the input
            file.
        middle_nodes: bool
            If middle nodes should be returned or not.
        """

        if nodes is None:
            node_list = self.nodes
        else:
            node_list = nodes
        return [node for node in node_list if
            (include_solid_nodes or not node.is_dat) and
            (middle_nodes or not node.is_middle_node)
            ]

    def get_global_coordinates(self, **kwargs):
        """
        Return an array with the coordinates of some nodes. As well as a list
        with the corresponding nodes.

        Args
        ----
        kwargs:
            Will be passed to self.get_global_nodes.

        Return
        ----
        pos: np.array
            Numpy array with all the positions of the nodes.
        beam_nodes: [Node]
            List of the nodes corresponding to the rows in the pos array. If
            the input argument nodes was given, it will be returned as is.
        """
        beam_nodes = self.get_global_nodes(**kwargs)
        pos = np.zeros([len(beam_nodes), 3])
        for i, node in enumerate(beam_nodes):
            pos[i, :] = node.coordinates
        return pos, beam_nodes

    def get_global_quaternions(self, **kwargs):
        """
        Return an array with the quaternions of some beam nodes. As well as a
        list with the corresponding nodes.

        Args
        ----
        kwargs:
            Will be passed to self.get_global_nodes.

        Return
        ----
        rot: np.array
            Numpy array with all the quaternions of the nodes.
        beam_nodes: [Node]
            List of the nodes corresponding to the rows in the rot array. If
            the input argument nodes was given, it will be returned as is.
        """
        beam_nodes = self.get_global_nodes(**kwargs)
        rot = np.zeros([len(beam_nodes), 4])
        for i, node in enumerate(beam_nodes):
            if node.rotation is not None:
                rot[i, :] = node.rotation.get_quaternion()
            else:
                raise ValueError('Got a node without rotation, this should not'
                    + ' happen here!')
        return rot, beam_nodes

    def translate(self, vector):
        """
        Translate all beam nodes of this mesh.

        Args
        ----
        vector: np.array, list
            3D vector that will be added to all nodes.
        """
        beam_nodes = self.get_global_nodes()
        for node in beam_nodes:
            node.coordinates += vector

    def rotate(self, rotation, origin=None, only_rotate_triads=False):
        """
        Rotate all beam nodes of the mesh with rotation.

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

        # Get array with all quaternions for the nodes.
        rot1, beam_nodes = self.get_global_quaternions()

        # Additional rotation.
        rotnew = add_rotations(rotation, rot1)

        if not only_rotate_triads:
            if isinstance(rotation, Rotation):
                rot2 = rotation.get_quaternion().transpose()
            else:
                rot2 = rotation

            # Get array with all positions for the nodes.
            pos, _beam_nodes = self.get_global_coordinates(nodes=beam_nodes)

            # Check if origin has to be added.
            if origin is not None:
                pos -= origin

            # New position array.
            posnew = np.zeros_like(pos)

            # Evaluate the new positions using the numpy data structure.
            # (code is taken from /utility/rotation.nb)
            rot2 = rot2.transpose()

            q0_q0 = np.square(rot2[0])
            q0_q1_2 = 2.0 * rot2[0] * rot2[1]
            q0_q2_2 = 2.0 * rot2[0] * rot2[2]
            q0_q3_2 = 2.0 * rot2[0] * rot2[3]

            q1_q1 = np.square(rot2[1])
            q1_q2_2 = 2.0 * rot2[1] * rot2[2]
            q1_q3_2 = 2.0 * rot2[1] * rot2[3]

            q2_q2 = np.square(rot2[2])
            q2_q3_2 = 2.0 * rot2[2] * rot2[3]

            q3_q3 = np.square(rot2[3])

            posnew[:, 0] = (
                (q0_q0 + q1_q1 - q2_q2 - q3_q3) * pos[:, 0] +
                (q1_q2_2 - q0_q3_2) * pos[:, 1] +
                (q0_q2_2 + q1_q3_2) * pos[:, 2]
                )
            posnew[:, 1] = (
                (q1_q2_2 + q0_q3_2) * pos[:, 0] +
                (q0_q0 - q1_q1 + q2_q2 - q3_q3) * pos[:, 1] +
                (-q0_q1_2 + q2_q3_2) * pos[:, 2]
                )
            posnew[:, 2] = (
                (-q0_q2_2 + q1_q3_2) * pos[:, 0] +
                (q0_q1_2 + q2_q3_2) * pos[:, 1] +
                (q0_q0 - q1_q1 - q2_q2 + q3_q3) * pos[:, 2]
                )

            if origin is not None:
                posnew += origin

        for i, node in enumerate(beam_nodes):
            if node.rotation is not None:
                node.rotation.q = rotnew[i, :]
            if not only_rotate_triads:
                node.coordinates = posnew[i, :]

    def reflect(self, normal_vector, origin=None, flip=False):
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
        flip: bool
            When True, the beams are flipped, so that the direction along the
            beam is reversed.
        """

        # Normalize the normal vector.
        normal_vector = np.array(normal_vector / np.linalg.norm(normal_vector))

        # Get array with all quaternions and positions for the nodes.
        pos, beam_nodes = self.get_global_coordinates()
        rot1, _beam_nodes = self.get_global_quaternions(nodes=beam_nodes)

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

        # First get all e3 vectors of the nodes.
        e3 = np.zeros_like(pos)
        e3[:, 0] = 2 * (rot1[:, 0] * rot1[:, 2] + rot1[:, 1] * rot1[:, 3])
        e3[:, 1] = 2 * (-1 * rot1[:, 0] * rot1[:, 1] + rot1[:, 2] * rot1[:, 3])
        e3[:, 2] = rot1[:, 0]**2 - rot1[:, 1]**2 - rot1[:, 2]**2 + \
            rot1[:, 3]**2

        # Get the dot and cross product of e3 and the normal vector.
        rot2 = np.zeros_like(rot1)
        rot2[:, 0] = np.dot(e3, normal_vector)
        rot2[:, 1:] = np.cross(e3, normal_vector)

        # Add to the existing rotations.
        rot_new = add_rotations(rot2, rot1)

        if flip:
            # To achieve the flip, the triads are rotated with the angle pi
            # around the e2 axis.
            rot_flip = Rotation([0, 1, 0], np.pi)
            rot_new = add_rotations(rot_new, rot_flip)

            # Each element has to switch its nodes internally.
            for element in self.elements:
                element.flip()

        # Set the new positions and rotations.
        for i, node in enumerate(beam_nodes):
            node.coordinates = pos_new[i, :]
            node.rotation.q = rot_new[i, :]

    def wrap_around_cylinder(self, radius=None, advanced_warning=True):
        """
        Wrap the geometry around a cylinder. The y-z plane gets morphed into
        the z-axis of symmetry. If all nodes are on the same y-z plane, the
        radius of the created cylinder is the x coordinate of that plane. If
        the nodes are not on the same y-z plane, the radius has to be given
        explicitly.

        Args
        ----
        radius: double
            If this value is given AND not all nodes are on the same y-z plane,
            then use this radius for the calculation of phi for all nodes.
            This might still lead to distorted elements!.
        advanced_warning: bool
            If each element should be checked if it is either parallel to the
            y-z or x-z plane. This is computationally expensive, but in most
            cases (up to 100,000 elements) this check can be left activated.
        """

        pos, beam_nodes = self.get_global_coordinates()
        quaternions = np.zeros([len(beam_nodes), 4])

        # The x coordinate is the radius, the y coordinate the arc length.
        points_x = pos[:, 0].copy()

        # Check if all points are on the same y-z plane.
        if np.abs(np.min(points_x) - np.max(points_x)) > mpy.eps_pos:
            # The points are not all on the y-z plane, get the reference
            # radius.
            if radius is not None:
                if advanced_warning:
                    # Here we check, if each element lays on a plane parallel
                    # to the y-z plane, or parallel to the x-z plane.
                    #
                    # To be exactly sure, we could check the rotations here,
                    # i.e. if they are also in plane.
                    element_warning = []
                    for i_element, element in enumerate(
                            [e for e in self.elements if not e.is_dat]):
                        element_coordinates = np.zeros([len(element.nodes), 3])
                        for i_node, node in enumerate(element.nodes):
                            element_coordinates[i_node, :] = node.coordinates
                        is_yz = np.max(np.abs(element_coordinates[:, 0]
                            - element_coordinates[0, 0])) < mpy.eps_pos
                        is_xz = np.max(np.abs(element_coordinates[:, 1]
                            - element_coordinates[0, 1])) < mpy.eps_pos
                        if not(is_yz or is_xz):
                            element_warning.append(i_element)
                    if len(element_warning) != 0:
                        warnings.warn('There are elements which are not '
                            'parallel to the y-z or x-y plane. This will lead '
                            'to distorted elements!')
                else:
                    warnings.warn('The nodes are not on the same y-z plane. '
                        'This may lead to distorted elements!')
            else:
                raise ValueError('The nodes that should be wrapped around a '
                    + 'cylinder are not on the same y-z plane. This will give '
                    + 'unexpected results. Give a reference radius!')
            radius_phi = radius
            radius_points = points_x
        elif radius is None or np.abs(points_x[0] - radius) < mpy.eps_pos:
            radius_points = radius_phi = points_x[0]
        else:
            raise ValueError(('The points are all on the same y-z plane with '
                + 'the x-coordinate {} but the given radius {} is different. '
                + 'This does not make sense.').format(points_x[0], radius))

        # Get the angle for all nodes.
        phi = pos[:, 1] / radius_phi

        # The rotation is about the z-axis.
        quaternions[:, 0] = np.cos(0.5 * phi)
        quaternions[:, 3] = np.sin(0.5 * phi)

        # Set the new positions in the global array.
        pos[:, 0] = radius_points * np.cos(phi)
        pos[:, 1] = radius_points * np.sin(phi)

        # Rotate the mesh
        self.rotate(quaternions, only_rotate_triads=True)

        # Set the new position for the nodes.
        for i, node in enumerate(beam_nodes):
            node.coordinates = pos[i, :]

    def couple_nodes(self, *, nodes=None, reuse_matching_nodes=False,
            coupling_type=mpy.bc.point_coupling,
            coupling_dof_type=mpy.coupling_dof.fix):
        """
        Search through nodes and connect all nodes with the same coordinates.

        Args:
        ----
        nodes: [Node]
            List of nodes to couple. If None is given, all nodes of the mesh
            are coupled (except middle and dat nodes).
        reuse_matching_nodes: bool
            If two nodes have the same position and rotation, the nodes are
            reduced to one node in the mesh. Be aware, that this might lead to
            issues if not all DOFs of the nodes should be coupled.
        coupling_type: mpy.bc
            Type of point coupling.
        coupling_dof_type: str, mpy.coupling_dof
            str: The string that will be used in the input file.
            mpy.coupling_dof.fix: Fix all positional and rotational DOFs of the
                nodes together.
            mpy.coupling_dof.joint: Fix all positional DOFs of the nodes
                together.
        """

        # Check that a coupling BC is given.
        if not (coupling_type == mpy.bc.point_coupling
                or coupling_type == mpy.bc.point_coupling_penalty):
            raise ValueError('Only coupling conditions can be applied in '
                '"couple_nodes"!')

        # Get the nodes that should be checked for coupling. Middle nodes are
        # not checked, as coupling can only be applied to the boundary nodes.
        node_list = self.get_global_nodes(nodes=nodes, middle_nodes=False)
        partner_nodes = find_close_nodes(node_list)
        if len(partner_nodes) == 0:
            # If no partner nodes were found, end this function.
            return

        if reuse_matching_nodes:
            # Check if there are nodes with the same rotation. If there are the
            # nodes are reused, and no coupling is inserted.

            # Set the links to all nodes in the mesh.
            self.unlink_nodes()
            self.get_unique_geometry_sets(link_nodes=True)
            self.set_node_links()

            # Go through partner nodes.
            for node_list in partner_nodes:
                # Get array with rotation vectors.
                rotation_vectors = np.zeros([len(node_list), 3])
                for i, node in enumerate(node_list):
                    rotation_vectors[i, :] = \
                        node.rotation.get_rotation_vector()

                # Abuse the find close nodes function to find nodes with the
                # same rotation.
                rot_partner_indices = find_close_points(rotation_vectors,
                    binning=False, eps=mpy.eps_quaternion)
                has_partner, n_partners = partner_indices_to_point_partners(
                    rot_partner_indices, len(node_list))

                # Check if nodes with the same rotations were found.
                if len(rot_partner_indices) == 0:
                    self.add(Coupling(node_list, coupling_type,
                        coupling_dof_type))
                else:
                    # There are nodes that need to be combined.
                    combining_nodes = []
                    coupling_nodes = []
                    found_partner_id = [None for _i in range(n_partners)]

                    # Add the nodes that need to be combined and add the nodes
                    # that will be coupled.
                    for i, partner in enumerate(has_partner):

                        if partner == -1:
                            # This node does not have a partner with the same
                            # rotation.
                            coupling_nodes.append(node_list[i])

                        elif found_partner_id[partner] is not None:
                            # This node has already a processed partner, add
                            # this one to the combining nodes.
                            combining_nodes[found_partner_id[partner]].append(
                                node_list[i])

                        else:
                            # This is the first node of a partner set that was
                            # found. This one will remain, the other ones will
                            # be replaced with this one.
                            new_index = len(combining_nodes)
                            found_partner_id[partner] = new_index
                            combining_nodes.append([node_list[i]])
                            coupling_nodes.append(node_list[i])

                    # Add the coupling nodes.
                    if len(coupling_nodes) > 1:
                        self.add(Coupling(coupling_nodes, coupling_type,
                            coupling_dof_type))

                    # Replace the identical nodes.
                    for combine_list in combining_nodes:
                        master_node = combine_list[0]
                        for node in combine_list[1:]:
                            node.replace_with(master_node)

        else:
            # Connect close nodes with a coupling.
            for node_list in partner_nodes:
                self.add(Coupling(node_list, coupling_type, coupling_dof_type))

    def unlink_nodes(self):
        """Delete the linked arrays and global indices in all nodes."""
        for node in self.nodes:
            if not node.is_dat:
                node.unlink()

    def get_nodes_by_function(self, function, *args, middle_nodes=False,
            **kwargs):
        """
        Return all nodes for which the function evaluates to true.

        Args
        ----
        function: function(node, *args)
            Nodes for which this function is true are returned.
        middle_nodes: bool
            If this is true, middle nodes of a beam are also returned.
        """
        node_list = self.get_global_nodes(middle_nodes=middle_nodes)
        return [node for node in node_list if function(node, *args, **kwargs)]

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

        pos, beam_nodes = self.get_global_coordinates(nodes=nodes)
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
                            min_max_nodes.append(beam_nodes[index])
                    geometry['{}_{}'.format(direction, text)] = GeometrySet(
                        mpy.geo.point,
                        min_max_nodes
                        )
        return geometry

    def check_overlapping_elements(self, raise_error=True):
        """
        Check if there are overlapping elements in the mesh. This is done by
        checking if all middle nodes of beam elements have unique coordinates
        in the mesh.
        """

        # Number of middle nodes.
        middle_nodes = [node for node in self.nodes if
            (not node.is_dat) and node.is_middle_node]

        # Only check if there are middle nodes.
        if len(middle_nodes) == 0:
            return

        # Get array with middle nodes.
        coordinates = np.zeros([len(middle_nodes), 3])
        for i, node in enumerate(middle_nodes):
            coordinates[i, :] = node.coordinates

        # Check if there are double entries in the coordinates.
        partner_indices = find_close_points(coordinates)
        if len(partner_indices) > 0:
            if raise_error:
                raise ValueError('There are multiple middle nodes with the '
                    + 'same coordinates. Per default this raises an error! '
                    + 'This check can be turned of with '
                    + 'mpy.check_overlapping_elements=False')
            else:
                warnings.warn('There are multiple middle nodes with the '
                        + 'same coordinates!')

            # Add the partner index to the middle nodes.
            for i_partner, partners in enumerate(partner_indices):
                for i_node in partners:
                    middle_nodes[i_node].element_partner_index = i_partner

    def preview_python(self):
        """Display the elements in this mesh in matplotlib."""

        # Import the relevant matplotlib modules.
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # Create figure.
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Loop over elements.
        for element in self.elements:
            if not element.is_dat:
                element.preview_python(ax)

        # Finish plot.
        ax.set_aspect('auto')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def write_vtk(self, output_name='meshpy', output_directory='',
            overlapping_elements=True, coupling_sets=False, **kwargs):
        """
        Write the contents of this mesh to VTK files.

        Args
        ----
        output_name: str
            Base name of the output file. There will be a {}_beam.vtu and
            {}_solid.vtu file.
        output_directory: path
            Directory where the output files will be written.
        overlapping_elements: bool
            I elements should be checked for overlapping. If they overlap, the
            output will mark them.
        coupling_sets: bool
            If coupling sets should also be displayed.
        """

        # Object to store VKT data and write it to file.
        vtkwriter_beam = VTKWriter()
        vtkwriter_solid = VTKWriter()

        # Get the set numbers of the mesh
        self.get_unique_geometry_sets(coupling_sets=coupling_sets,
            link_nodes=True)

        if overlapping_elements:
            # Check for overlapping elements.
            self.check_overlapping_elements(raise_error=False)

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
            function_generator=None, interval=None, n_el=1, add_sets=False,
            start_node=None, end_node=None, vtk_cell_data=None):
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
        vtk_cell_data: {cell_data_name (str): cell_data_value (float)}
            With this argument, a vtk cell data can be set for the elements
            created within this function. This can be used to check which
            elements are created by which function.

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

        # Set vtk cell data on created elements.
        if vtk_cell_data is not None:
            for data_name, data_value in vtk_cell_data.items():
                for element in elements:
                    if data_name in element.vtk_cell_data.keys():
                        raise KeyError(('The cell data "{}" '
                            + 'already exists!').format(data_name))
                    element.vtk_cell_data[data_name] = data_value

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
        return_set['start'] = GeometrySet(mpy.geo.point, nodes=nodes[0])
        return_set['end'] = GeometrySet(mpy.geo.point, nodes=end_node)
        return_set['line'] = GeometrySet(mpy.geo.line, nodes=nodes)
        if add_sets:
            self.add(return_set)
        return return_set

    def copy(self):
        """
        Return a deep copy of this mesh. The functions and materials will not
        be deep copied.
        """
        return copy.deepcopy(self)
