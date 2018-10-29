# -*- coding: utf-8 -*-
"""
This module implements beam elements for the mesh.
"""

# Python modules.
import numpy as np
import vtk
import warnings

# Meshpy modules.
from . import mpy, Element, Node, add_point_data_node_sets


class Beam(Element):
    """A base class for a beam element."""

    # A array that defines the creation of the nodes for this beam:
    #    self.nodes_create = [
    #        [ xi, create_rotation, is_middle_node ], # fist node
    #        ...
    #        ]
    nodes_create = []

    def __init__(self, material=None, nodes=None):
        Element.__init__(self, nodes=nodes, material=material)

    def create_beam(self, beam_function, start_node=None, end_node=None):
        """
        Create the nodes for this beam element. The function returns a list
        with the created nodes.

        Args
        ----
        beam_function: function(xi)
            Returns the position and rotation of the beam along the local
            coordinate xi.
        start_node: Node
            If this argument is given, this is the node of the beam at xi=-1.
        end_node: Node
            If this argument is given, this is the node of the beam at xi=1.
        In the case of start_node and end_node, it is checked, that the
        function and the node have the same coordinates / rotations. If the
        start node has a different rotation, but the same tangent vector, the
        beam is rotated to match the start node rotation.
        """

        if len(self.nodes) > 0:
            raise ValueError('The beam should not have any local nodes yet!')

        def check_node(node, xi, name, relative_rotation=None):
            """
            Check if the given node matches with the function at value xi.
            The return value is the relative rotation.
            """

            # Get position and rotation of beam at xi.
            pos, rot = beam_function(xi)

            # Check position.
            if np.linalg.norm(pos - node.coordinates) > mpy.eps_pos:
                raise ValueError('{} does not match with function!'.format(
                    name))

            # Check rotation.
            if not node.rotation == rot:

                if np.abs(xi - 1) < mpy.eps_pos:
                    # In the case of end node check if the beam is rotated.
                    if relative_rotation is not None:
                        if node.rotation == rot * relative_rotation:
                            return pos, rot, relative_rotation
                    # Otherwise, throw error, as the rotation has to match
                    # exactly with the given node.
                    raise ValueError('End rotation does not match with '
                        + 'given function!')

                elif not mpy.allow_beam_rotation:
                    # The settings do not allow for a rotation of the beam.
                    raise ValueError('Start rotation does not match with '
                        + 'given function!')

                # Now check if the first basis vector is in the same direction.
                relative_basis_1 = (node.rotation.inv() * rot * [1, 0, 0])
                if (np.linalg.norm(relative_basis_1 - [1, 0, 0])
                        < mpy.eps_quaternion):
                    # Calculate the relative rotation.
                    relative_rotation = rot.inv() * node.rotation
                else:
                    raise ValueError('The tangent of the start node does not '
                        + 'match with the given function!')

            # The default value for the relative rotation is None.
            return pos, rot, relative_rotation

        # Check start and end nodes.
        has_start_node = start_node is not None
        has_end_node = end_node is not None
        relative_rotation = None

        # Loop over local nodes.
        for i, [xi, create_rot, middle_node] in enumerate(self.nodes_create):

            # Get the position and rotation at xi.
            if i == 0 and has_start_node:
                pos, rot, relative_rotation = check_node(start_node, xi,
                    'start_node')
                self.nodes = [start_node]
            elif (i == len(self.nodes_create) - 1) and has_end_node:
                pos, rot, _relative_rotation = check_node(end_node, xi,
                    'end_node', relative_rotation=relative_rotation)
            else:
                pos, rot = beam_function(xi)

            # Create the node.
            if (
                    (i > 0 or not has_start_node)
                    and
                    (i < len(self.nodes_create) - 1 or not has_end_node)
                    ):

                if create_rot:
                    self.nodes.append(Node(
                        pos,
                        rotation=rot * relative_rotation,
                        is_middle_node=middle_node
                        ))
                else:
                    self.nodes.append(Node(
                        pos,
                        rotation=None,
                        is_middle_node=middle_node
                        ))

        # Get a list with the created nodes.
        if has_start_node:
            created_nodes = self.nodes[1:]
        else:
            created_nodes = self.nodes.copy()

        # Add the end node to the beam.
        if has_end_node:
            self.nodes.append(end_node)

        # Return the created nodes.
        return created_nodes

    def flip(self):
        """
        Reverse the nodes of this element. This is usually used when reflected.
        """
        self.nodes = [self.nodes[-1 - i] for i in range(len(self.nodes))]

    def preview_python(self, ax):
        """Plot the beam in matplotlib, by connecting the nodes."""

        coordinates = np.array([node.coordinates for node in self.nodes])
        ax.plot(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], '-x')

    def get_vtk(self, vtkwriter_beam, vtkwriter_solid):
        """
        Add the representation of this element to the VTK writer as a poly
        line.
        """

        # Dictionary with cell data.
        cell_data = {}
        cell_data['cross_section_radius'] = self.material.radius

        # Dictionary with point data.
        point_data = {}
        point_data['node_value'] = []
        point_data['base_vector_1'] = []
        point_data['base_vector_2'] = []
        point_data['base_vector_3'] = []

        # Array with nodal coordinates.
        coordinates = np.zeros([len(self.nodes), 3])
        for i, node in enumerate(self.nodes):
            coordinates[i, :] = node.coordinates
            if node.is_middle_node:
                point_data['node_value'].append(0.5)
            else:
                point_data['node_value'].append(1.)

            R = node.rotation.get_rotation_matrix()

            # Set small values to 0.
            R[abs(R.real) < mpy.eps_quaternion] = 0.0

            point_data['base_vector_1'].append(R[:, 0])
            point_data['base_vector_2'].append(R[:, 1])
            point_data['base_vector_3'].append(R[:, 2])

        # Add the node sets connected to this element.
        add_point_data_node_sets(point_data, self.nodes)

        # Add poly line to writer.
        vtkwriter_beam.add_cell(vtk.vtkPolyLine, coordinates,
            cell_data=cell_data, point_data=point_data)


class Beam3rHerm2Lin3(Beam):
    """Represents a BEAM3R HERM2LIN3 element."""

    nodes_create = [
        [-1, True, False],
        [0, True, True],
        [1, True, False]
        ]

    beam_type = mpy.beam_type_reissner

    def _get_dat(self):
        """ Return the line for the input file. """

        string_nodes = ''
        string_triads = ''
        for i in [0, 2, 1]:
            node = self.nodes[i]
            string_nodes += '{} '.format(node.n_global)
            string_triads += ' ' + node.rotation.get_dat()

        return '{} BEAM3R HERM2LIN3 {}MAT {} TRIADS{}'.format(
            self.n_global,
            string_nodes,
            self.material.n_global,
            string_triads
            )


class Beam3kClass(Beam):
    """Represents a Kirchhoff beam element."""

    nodes_create = [
        [-1, True, False],
        [0, True, True],
        [1, True, False]
        ]

    beam_type = mpy.beam_type_kirchhoff

    def __init__(self, *, weak=True, rotvec=True, FAD=False, **kwargs):
        Beam.__init__(self, **kwargs)

        # Set the parameters for this beam.
        self.weak = weak
        self.rotvec = rotvec
        self.FAD = FAD

        # Show warning when not using rotvec.
        if not rotvec:
            warnings.warn('Use rotvec=False with caution, especially when '
                + 'applying the boundary conditions and couplings.')

    def _get_dat(self):
        """ Return the line for the input file. """

        string_nodes = ''
        string_triads = ''
        for i in [0, 2, 1]:
            node = self.nodes[i]
            string_nodes += '{} '.format(node.n_global)
            string_triads += ' ' + node.rotation.get_dat()

        string_dat = ('{} BEAM3K LIN3 {} WK {} ROTVEC {} MAT {} ' +
            'TRIADS{}{}').format(
                self.n_global,
                string_nodes,
                '1' if self.weak else '0',
                '1' if self.rotvec else '0',
                self.material.n_global,
                string_triads,
                ' FAD' if self.FAD else ''
                )

        return string_dat


def Beam3k(**kwargs_class):
    """
    This factory returns a function that creates a new Beam3kClass object with
    certain attributes defined. The returned function behaves like a call to
    the object.
    """

    def create_class(**kwargs):
        """
        The function that will be returned. This function should behave like
        the call to the __init__ function of the class. 
        """
        return Beam3kClass(**kwargs_class, **kwargs)

    return create_class
