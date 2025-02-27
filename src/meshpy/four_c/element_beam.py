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

import warnings

import numpy as np

from meshpy.core.conf import mpy
from meshpy.core.element_beam import Beam
from meshpy.four_c.material import (
    MaterialEulerBernoulli,
    MaterialKirchhoff,
    MaterialReissner,
    MaterialReissnerElastoplastic,
)


class Beam3rHerm2Line3(Beam):
    """Represents a BEAM3R HERM2LINE3 element."""

    nodes_create = [-1, 0, 1]
    beam_type = mpy.beam.reissner
    valid_material = [MaterialReissner, MaterialReissnerElastoplastic]

    coupling_fix_string = "NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0"
    coupling_joint_string = "NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0"

    def _get_dat(self):
        """Return the line for the input file."""

        string_nodes = ""
        string_triads = ""
        for i in [0, 2, 1]:
            node = self.nodes[i]
            string_nodes += f"{node.i_global} "
            string_triads += " " + node.rotation.get_dat()

        # Check the material.
        self._check_material()

        return (
            f"{self.i_global} BEAM3R HERM2LINE3 {string_nodes}MAT {self.material.i_global} "
            f"TRIADS{string_triads}"
        )


class Beam3rLine2Line2(Beam):
    """Represents a Reissner beam with linear shapefunctions in the rotations
    as well as the displacements."""

    nodes_create = [-1, 1]
    beam_type = mpy.beam.reissner
    valid_material = [MaterialReissner]

    coupling_fix_string = "NUMDOF 6 ONOFF 1 1 1 1 1 1"
    coupling_joint_string = "NUMDOF 6 ONOFF 1 1 1 0 0 0"

    def _get_dat(self):
        """Return the line for the input file."""

        string_nodes = ""
        string_triads = ""
        for i in [0, 1]:
            node = self.nodes[i]
            string_nodes += f"{node.i_global} "
            string_triads += " " + node.rotation.get_dat()

        # Check the material.
        self._check_material()

        return (
            f"{self.i_global} BEAM3R LINE2 {string_nodes}MAT {self.material.i_global} "
            f"TRIADS{string_triads}"
        )


class Beam3kClass(Beam):
    """Represents a Kirchhoff beam element."""

    nodes_create = [-1, 0, 1]
    beam_type = mpy.beam.kirchhoff
    valid_material = [MaterialKirchhoff]

    coupling_fix_string = "NUMDOF 7 ONOFF 1 1 1 1 1 1 0"
    coupling_joint_string = "NUMDOF 7 ONOFF 1 1 1 0 0 0 0"

    def __init__(self, *, weak=True, rotvec=True, is_fad=True, **kwargs):
        Beam.__init__(self, **kwargs)

        # Set the parameters for this beam.
        self.weak = weak
        self.rotvec = rotvec
        self.is_fad = is_fad

        # Show warning when not using rotvec.
        if not rotvec:
            warnings.warn(
                "Use rotvec=False with caution, especially when applying the boundary conditions "
                "and couplings."
            )

    def _get_dat(self):
        """Return the line for the input file."""

        string_nodes = ""
        string_triads = ""
        for i in [0, 2, 1]:
            node = self.nodes[i]
            string_nodes += f"{node.i_global} "
            string_triads += " " + node.rotation.get_dat()

        # Check the material.
        self._check_material()

        string_dat = ("{} BEAM3K LINE3 {} WK {} ROTVEC {} MAT {} TRIADS{}{}").format(
            self.i_global,
            string_nodes,
            "1" if self.weak else "0",
            "1" if self.rotvec else "0",
            self.material.i_global,
            string_triads,
            " FAD" if self.is_fad else "",
        )

        return string_dat


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


class Beam3eb(Beam):
    """Represents a Euler Bernoulli beam element."""

    nodes_create = [-1, 1]
    beam_type = mpy.beam.euler_bernoulli
    valid_material = [MaterialEulerBernoulli]

    def _get_dat(self):
        """Return the line for the input file."""

        # The two rotations must be the same and the x1 vector must point from
        # the start point to the end point.
        if not self.nodes[0].rotation == self.nodes[1].rotation:
            raise ValueError(
                "The two nodal rotations in Euler Bernoulli beams must be the same, i.e. the beam "
                "has to be straight!"
            )
        direction = self.nodes[1].coordinates - self.nodes[0].coordinates
        t1 = self.nodes[0].rotation * [1, 0, 0]
        if np.linalg.norm(direction / np.linalg.norm(direction) - t1) >= mpy.eps_pos:
            raise ValueError(
                "The rotations do not match the direction of the Euler Bernoulli beam!"
            )

        string_nodes = ""
        string_triads = ""
        for i in [0, 1]:
            node = self.nodes[i]
            string_nodes += f"{node.i_global} "
            string_triads += " " + node.rotation.get_dat()

        # Check the material.
        self._check_material()

        return (
            f"{self.i_global} BEAM3EB LINE2 {string_nodes}MAT {self.material.i_global}"
        )
