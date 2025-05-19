# The MIT License (MIT)
#
# Copyright (c) 2018-2025 MeshPy Authors
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""This file implements beam elements for 4C."""

import warnings as _warnings

import numpy as _np

from meshpy.core.conf import mpy as _mpy
from meshpy.core.element_beam import Beam as _Beam
from meshpy.four_c.material import MaterialEulerBernoulli as _MaterialEulerBernoulli
from meshpy.four_c.material import MaterialKirchhoff as _MaterialKirchhoff
from meshpy.four_c.material import MaterialReissner as _MaterialReissner
from meshpy.four_c.material import (
    MaterialReissnerElastoplastic as _MaterialReissnerElastoplastic,
)


class Beam3rHerm2Line3(_Beam):
    """Represents a BEAM3R HERM2LINE3 element."""

    nodes_create = [-1, 0, 1]
    beam_type = _mpy.beam.reissner
    valid_material = [_MaterialReissner, _MaterialReissnerElastoplastic]

    coupling_fix_dict = {"NUMDOF": 9, "ONOFF": [1, 1, 1, 1, 1, 1, 0, 0, 0]}
    coupling_joint_dict = {"NUMDOF": 9, "ONOFF": [1, 1, 1, 0, 0, 0, 0, 0, 0]}

    def dump_to_list(self):
        """Return a list with the (single) item representing this element."""

        # Check the material.
        self._check_material()

        # TODO here a numpy data type is converted to a standard Python
        # data type. Once FourCIPP can handle non standard data types,
        # this should be removed.
        return {
            "id": self.i_global,
            "cell": {
                "type": "HERM2LINE3",
                "connectivity": [int(self.nodes[i].i_global) for i in [0, 2, 1]],
            },
            "data": {
                "type": "BEAM3R",
                "MAT": self.material.i_global,
                "TRIADS": [
                    float(item)
                    for i in [0, 2, 1]
                    for item in self.nodes[i].rotation.get_rotation_vector()
                ],
            },
        }


class Beam3rLine2Line2(_Beam):
    """Represents a Reissner beam with linear shapefunctions in the rotations
    as well as the displacements."""

    nodes_create = [-1, 1]
    beam_type = _mpy.beam.reissner
    valid_material = [_MaterialReissner]

    coupling_fix_dict = {"NUMDOF": 6, "ONOFF": [1, 1, 1, 1, 1, 1]}
    coupling_joint_dict = {"NUMDOF": 6, "ONOFF": [1, 1, 1, 0, 0, 0]}

    def dump_to_list(self):
        """Return a list with the (single) item representing this element."""

        # Check the material.
        self._check_material()

        # TODO here a numpy data type is converted to a standard Python
        # data type. Once FourCIPP can handle non standard data types,
        # this should be removed.
        return {
            "id": self.i_global,
            "cell": {
                "type": "LINE2",
                "connectivity": [int(self.nodes[i].i_global) for i in [0, 1]],
            },
            "data": {
                "type": "BEAM3R",
                "MAT": self.material.i_global,
                "TRIADS": [
                    float(item)
                    for i in [0, 1]
                    for item in self.nodes[i].rotation.get_rotation_vector()
                ],
            },
        }


class Beam3kClass(_Beam):
    """Represents a Kirchhoff beam element."""

    nodes_create = [-1, 0, 1]
    beam_type = _mpy.beam.kirchhoff
    valid_material = [_MaterialKirchhoff]

    coupling_fix_dict = {"NUMDOF": 7, "ONOFF": [1, 1, 1, 1, 1, 1, 0]}
    coupling_joint_dict = {"NUMDOF": 7, "ONOFF": [1, 1, 1, 0, 0, 0, 0]}

    def __init__(self, *, weak=True, rotvec=True, is_fad=True, **kwargs):
        _Beam.__init__(self, **kwargs)

        # Set the parameters for this beam.
        self.weak = weak
        self.rotvec = rotvec
        self.is_fad = is_fad

        # Show warning when not using rotvec.
        if not rotvec:
            _warnings.warn(
                "Use rotvec=False with caution, especially when applying the boundary conditions "
                "and couplings."
            )

    def dump_to_list(self):
        """Return a list with the (single) item representing this element."""

        # Check the material.
        self._check_material()

        # TODO here a numpy data type is converted to a standard Python
        # data type. Once FourCIPP can handle non standard data types,
        # this should be removed.
        return {
            "id": self.i_global,
            "cell": {
                "type": "LINE3",
                "connectivity": [int(self.nodes[i].i_global) for i in [0, 2, 1]],
            },
            "data": {
                "type": "BEAM3K",
                "WK": 1 if self.weak else 0,
                "ROTVEC": 1 if self.rotvec else 0,
                "MAT": self.material.i_global,
                "TRIADS": [
                    float(item)
                    for i in [0, 2, 1]
                    for item in self.nodes[i].rotation.get_rotation_vector()
                ],
                **({"USE_FAD": True} if self.is_fad else {}),
            },
        }


def Beam3k(**kwargs_class):
    """This factory returns a function that creates a new Beam3kClass object
    with certain attributes defined.

    The returned function behaves like a call to the object.
    """

    def create_class(**kwargs):
        """The function that will be returned.

        This function should behave like the call to the __init__
        function of the class.
        """
        return Beam3kClass(**kwargs_class, **kwargs)

    return create_class


class Beam3eb(_Beam):
    """Represents a Euler Bernoulli beam element."""

    nodes_create = [-1, 1]
    beam_type = _mpy.beam.euler_bernoulli
    valid_material = [_MaterialEulerBernoulli]

    def dump_to_list(self):
        """Return a list with the (single) item representing this element."""

        # Check the material.
        self._check_material()

        # The two rotations must be the same and the x1 vector must point from
        # the start point to the end point.
        if not self.nodes[0].rotation == self.nodes[1].rotation:
            raise ValueError(
                "The two nodal rotations in Euler Bernoulli beams must be the same, i.e. the beam "
                "has to be straight!"
            )
        direction = self.nodes[1].coordinates - self.nodes[0].coordinates
        t1 = self.nodes[0].rotation * [1, 0, 0]
        if _np.linalg.norm(direction / _np.linalg.norm(direction) - t1) >= _mpy.eps_pos:
            raise ValueError(
                "The rotations do not match the direction of the Euler Bernoulli beam!"
            )

        # TODO here a numpy data type is converted to a standard Python
        # data type. Once FourCIPP can handle non standard data types,
        # this should be removed.
        return {
            "id": self.i_global,
            "cell": {
                "type": "LINE2",
                "connectivity": [int(self.nodes[i].i_global) for i in [0, 1]],
            },
            "data": {
                "type": "BEAM3EB",
                "MAT": self.material.i_global,
            },
        }
