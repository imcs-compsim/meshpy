# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# MeshPy: A beam finite element input generator
#
# MIT License
#
# Copyright (c) 2018-2024
#     Ivo Steinbrecher
#     Institute for Mathematics and Computer-Based Simulation
#     Universitaet der Bundeswehr Muenchen
#     https://www.unibw.de/imcs-en
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
This file contains the wrapper for the LocSys condition for 4c.
"""


# MeshPy modules
from .. import mpy
from ..boundary_condition import BoundaryCondition


class LocSysCondition(BoundaryCondition):
    """
    This object handles the LocSys condition.
    """

    def __init__(
        self,
        geometry_set,
        rotation,
        **kwargs,
    ):
        """
        Initialize the object.

        Args
        ----
        geometry_set: GeometrySet, int
            Geometry that this boundary condition acts on. An integer can be
            given, in the case a dat file is imported. This integer is only
            temporary and will be replaced with the GeometrySet object.
        rotation: Rotation
            Object that represents the rotation of the coordinate system.
        """

        super().__init__(
            geometry_set,
            "ROTANGLE {} {} {} FUNCT 0 0 0 USEUPDATEDNODEPOS 0 USECONSISTENTNODENORMAL 0".format(
                *rotation.get_rotation_vector()
            ),
            bc_type=mpy.bc.locsys,
            **kwargs,
        )
