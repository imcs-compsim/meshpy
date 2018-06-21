# -*- coding: utf-8 -*-
"""
This module implements a class to couple geometry together.
"""

# Meshpy modules.
from . import mpy, GeometrySet, BaseMeshItem


class Coupling(BaseMeshItem):
    """Represents a coupling between geometry in BACI."""

    def __init__(self, nodes, coupling_type):
        BaseMeshItem.__init__(self, is_dat=False)
        self.node_set = GeometrySet(mpy.point, nodes=nodes) 
        self.coupling_type = coupling_type

    def _get_dat(self):
        if self.coupling_type == mpy.coupling_joint:
            string = 'NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0'
        elif self.coupling_type == mpy.coupling_fix:
            string = 'NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0'
        elif isinstance(self.coupling_type, str):
            string = self.coupling_type
        else:
            raise ValueError('coupling_type "{}" is not implemented!'.format(
                self.coupling_type
                )) 
        return 'E {} - {}'.format(self.node_set.n_global, string)