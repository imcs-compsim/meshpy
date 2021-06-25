# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# MeshPy: A beam finite element input generator.
#
# Copyright (c) 2021 Ivo Steinbrecher
#                    Institute for Mathematics and Computer-Based Simulation
#                    Universitaet der Bundeswehr Muenchen
#                    https://www.unibw.de/imcs-en
#
# TODO: Add license.
# -----------------------------------------------------------------------------
"""
This module implements a class to couple geometry together.
"""

# Python modules.
import numpy as np

# Meshpy modules.
from .conf import mpy
from .geometry_set import GeometrySet
from .boundary_condition import BoundaryConditionBase


class Coupling(BoundaryConditionBase):
    """Represents a coupling between geometry in BACI."""

    def __init__(self, geometry_set, coupling_type,
            check_overlapping_nodes=True, **kwargs):
        """
        Initialize this object.

        Args
        ----
        geometry_set: GeometrySet, [Nodes]
            Geometry that this boundary condition acts on.
        coupling_type: mpy.coupling, str
            If this is a string it is the string that will be used in the input
            file, otherwise it has to be of type mpy.coupling.
        check_overlapping_nodes: bool
            If all nodes of this coupling condition have to be at the same
            physical position.
        """

        if isinstance(geometry_set, list):
            geometry_set = GeometrySet(mpy.geo.point, geometry_set)
        super().__init__(geometry_set,
            bc_type=mpy.bc.point_coupling, **kwargs)
        self.coupling_type = coupling_type
        self.check_overlapping_nodes = check_overlapping_nodes

        # Perform the checks on this boundary condition.
        self.check()

    def check(self):
        """
        Check that all nodes that are coupled have the same position (depending
        on the check_overlapping_nodes parameter and if the object does not
        come from a dat file).
        """

        if self.is_dat or (not self.check_overlapping_nodes):
            return
        else:
            pos = np.zeros([len(self.geometry_set.nodes), 3])
            for i, node in enumerate(self.geometry_set.nodes):
                # Get the difference to the first node.
                pos[i, :] = (node.coordinates
                    - self.geometry_set.nodes[0].coordinates)
            if np.linalg.norm(pos) > mpy.eps_pos:
                raise ValueError('The nodes given to Coupling do not have the '
                    'same position.')

    def _get_dat(self):
        """
        Return the dat line for this object. If no explicit string was given,
        it depends on the coupling type as well as the beam type.
        """

        if isinstance(self.coupling_type, str):
            string = self.coupling_type
        else:
            # In this case we have to check which beams are connected to the
            # node.
            # TODO: Coupling also makes sense for different beam types, this
            # can be implemented at some point.
            beam_type = self.geometry_set.nodes[0].element_link[0].beam_type
            for node in self.geometry_set.nodes:
                for element in node.element_link:
                    if beam_type is not element.beam_type:
                        raise ValueError(('The first element in this coupling '
                            + 'is of the type "{}" another one is of type '
                            + '"{}"! They have to be of the same kind.'.format(
                                beam_type, element.beam_type)))
                    elif (beam_type is mpy.beam.kirchhoff
                            and element.rotvec is False):
                        raise ValueError('Couplings for Kirchhoff beams and '
                            + 'rotvec==False not yet implemented.')

            # In BACI it is not possible to couple beams of the same type, but
            # with different centerline discretizations, e.g. Beam3rHerm2Line3
            # and Beam3rLine2Line2, therefore, we check that all beams are
            # exactly the same type and discretization.
            # TODO: Remove this check once it is possible to couple beams, but
            # then also the syntax in the next few lines has to be adapted.
            beam_baci_type = type(self.geometry_set.nodes[0].element_link[0])
            for node in self.geometry_set.nodes:
                for element in node.element_link:
                    if not beam_baci_type == type(element):
                        raise ValueError('Coupling beams of different types '
                            'is not yet possible!')

            string = beam_baci_type.get_coupling_string(self.coupling_type)

        return 'E {} - {}'.format(self.geometry_set.n_global, string)
