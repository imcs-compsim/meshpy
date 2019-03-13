# -*- coding: utf-8 -*-
"""
This module implements a class to couple geometry together.
"""

# Python modules.
import numpy as np

# Meshpy modules.
from .conf import mpy
from .base_mesh_item import BaseMeshItem
from .geometry_set import GeometrySet


class Coupling(BaseMeshItem):
    """Represents a coupling between geometry in BACI."""

    def __init__(self, nodes, coupling_type):
        """Initialize this object."""
        BaseMeshItem.__init__(self, is_dat=False)
        self.node_set = GeometrySet(mpy.geo.point, nodes=nodes)
        self.coupling_type = coupling_type

        # Check that all nodes that are coupled have the same position (for now
        # this is the only way we consider couplings, maybe this has to be
        # enhanced later).
        pos = np.zeros([len(nodes), 3])
        for i, node in enumerate(nodes):
            if node.is_dat:
                raise TypeError('Couplings can only be applied to beam nodes!')
            else:
                # Get the difference to the first node.
                pos[i, :] = node.coordinates - nodes[0].coordinates
        if np.linalg.norm(pos) > mpy.eps_pos:
            raise ValueError('The nodes given to Coupling do not have the same'
                ' position. For now this case is not yet implemented.')

    def _get_dat(self):
        """
        Return the dat line for this object. It depends on the coupling type as
        well as the beam type.
        """

        # Check the beam type.
        beam_type = self.node_set.nodes[0].element_link[0].beam_type
        for node in self.node_set.nodes:
            for element in node.element_link:
                if beam_type is not element.beam_type:
                    raise ValueError(('The first element in this coupling is '
                        + 'of the type "{}" another one is of type "{}"! '
                        + 'They have to be of the same kind.'.format(beam_type,
                            element.beam_type)))
                elif (beam_type is mpy.beam.kirchhoff
                        and element.rotvec is False):
                    raise ValueError('Couplings for Kirchhoff beams and '
                        + 'rotvec==False not yet implemented.')

        # Get coupling string.
        if self.coupling_type is mpy.coupling.joint:
            if beam_type is mpy.beam.reissner:
                string = 'NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0'
            else:
                string = 'NUMDOF 7 ONOFF 1 1 1 0 0 0 0'
        elif self.coupling_type is mpy.coupling.fix:
            if beam_type is mpy.beam.reissner:
                string = 'NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0'
            else:
                string = 'NUMDOF 7 ONOFF 1 1 1 1 1 1 0'
        elif isinstance(self.coupling_type, str):
            string = self.coupling_type
        else:
            raise ValueError('coupling_type "{}" is not implemented!'.format(
                self.coupling_type
                ))
        return 'E {} - {}'.format(self.node_set.n_global, string)
