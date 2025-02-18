# MeshPy: A beam finite element input generator
#
# MIT License
#
# Copyright (c) 2018-2025
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
"""This file implements basic classes to manage materials in MeshPy."""

import numpy as np

from meshpy.core.base_mesh_item import BaseMeshItemFull


class Material(BaseMeshItemFull):
    """Base class for all materials."""

    def __init__(self, data=None, **kwargs):
        super().__init__(data=data, **kwargs)

    def __deepcopy__(self, memo):
        """When deepcopy is called on a mesh, we do not want the materials to
        be copied, as this will result in multiple equal materials in the input
        file."""

        # Add this object to the memo dictionary.
        memo[id(self)] = self

        # Return this object again, as no copy should be created.
        return self


class MaterialBeam(Material):
    """Base class for all beam materials."""

    def __init__(
        self,
        radius=-1.0,
        material_string=None,
        youngs_modulus=-1.0,
        nu=0.0,
        density=0.0,
        interaction_radius=None,
        **kwargs,
    ):
        """Set the material values that all beams have."""
        super().__init__(**kwargs)

        self.radius = radius
        self.material_string = material_string
        self.youngs_modulus = youngs_modulus
        self.nu = nu
        self.density = density
        self.radius = radius
        self.interaction_radius = interaction_radius
        self.area = None
        self.mom2 = None
        self.mom3 = None
        self.polar = None

    def calc_area_stiffness(self):
        """Calculate the relevant stiffness terms and the area for the given
        beam."""
        area = 4 * self.radius**2 * np.pi * 0.25
        mom2 = self.radius**4 * np.pi * 0.25
        mom3 = mom2
        polar = mom2 + mom3
        return area, mom2, mom3, polar
