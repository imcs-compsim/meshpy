# -*- coding: utf-8 -*-
"""
This module implements a basic class to manage materials in the baci input
file.
"""

# Python modules.
import numpy as np

# Meshpy modules.
from . import BaseMeshItem


class Material(BaseMeshItem):
    """Base class for all materials."""
    pass


class MaterialBeam(Material):
    """Base class for all beam materials."""

    def __init__(self, radius=None, **kwargs):
        """All beams have a radius attribute for visualization."""
        Material.__init__(self, **kwargs)
        self.radius = radius


class MaterialReissner(MaterialBeam):
    """Holds material definition for Reissner beams."""

    def __init__(self, material_string='MAT_BeamReissnerElastHyper',
            youngs_modulus=-1., nu=0., density=0., radius=-1.,
            shear_correction=1):

        MaterialBeam.__init__(self, data=None, is_dat=False)

        self.material_string = material_string
        self.youngs_modulus = youngs_modulus
        self.nu = nu
        self.density = density
        self.radius = radius
        self.area = 4 * self.radius**2 * np.pi * 0.25
        self.mom2 = self.radius**4 * np.pi * 0.25
        self.mom3 = self.mom2
        self.polar = self.mom2 + self.mom3
        self.shear_correction = shear_correction

    def _get_dat(self):
        """Return the line for this material."""
        string = 'MAT {} {} YOUNG {} POISSONRATIO {} DENS {} CROSSAREA {} '
        string += 'SHEARCORR {} MOMINPOL {} MOMIN2 {} MOMIN3 {}'
        return string.format(
            self.n_global,
            self.material_string,
            self.youngs_modulus,
            self.nu,
            self.density,
            self.area,
            self.shear_correction,
            self.polar,
            self.mom2,
            self.mom3
            )
