# -*- coding: utf-8 -*-
"""
This module implements a basic class to manage materials in the baci input
file.
"""

# Python modules.
import numpy as np

# Meshpy modules.
from .base_mesh_item import BaseMeshItem


class Material(BaseMeshItem):
    """Base class for all materials."""
    def __init__(self, data=None, is_dat=False, **kwargs):
        BaseMeshItem.__init__(self, data=data, is_dat=is_dat, **kwargs)

    def __deepcopy__(self, memo):
        """
        When deepcopy is called on a mesh, we do not want the materials to be
        copied, as this will result in multiple equal materials in the input
        file.
        """

        # Add this object to the memo dictionary.
        memo[id(self)] = self

        # Return this object again, as no copy should be created.
        return self


class MaterialBeam(Material):
    """Base class for all beam materials."""

    def __init__(self,
            radius=-1.,
            material_string=None,
            youngs_modulus=-1.,
            nu=0.,
            density=0.,
            **kwargs):
        """Set the material values that all beams have."""
        Material.__init__(self, **kwargs)

        self.radius = radius
        self.material_string = material_string
        self.youngs_modulus = youngs_modulus
        self.nu = nu
        self.density = density
        self.radius = radius
        self.area = 4 * self.radius**2 * np.pi * 0.25
        self.mom2 = self.radius**4 * np.pi * 0.25
        self.mom3 = self.mom2
        self.polar = self.mom2 + self.mom3


class MaterialReissner(MaterialBeam):
    """Holds material definition for Reissner beams."""

    def __init__(self, shear_correction=1, **kwargs):
        MaterialBeam.__init__(self,
            material_string='MAT_BeamReissnerElastHyper', **kwargs)

        # Shear factor for Reissner beam.
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


class MaterialKirchhoff(MaterialBeam):
    """Holds material definition for Kirchhoff beams."""

    def __init__(self, **kwargs):
        MaterialBeam.__init__(self,
            material_string='MAT_BeamKirchhoffElastHyper', **kwargs)

    def _get_dat(self):
        """Return the line for this material."""
        string = 'MAT {} {} YOUNG {} SHEARMOD {} DENS {} CROSSAREA {} '
        string += 'MOMINPOL {} MOMIN2 {} MOMIN3 {}'
        return string.format(
            self.n_global,
            self.material_string,
            self.youngs_modulus,
            self.youngs_modulus / (2. * (1. + self.nu)),
            self.density,
            self.area,
            self.polar,
            self.mom2,
            self.mom3
            )


class MaterialEulerBernoulli(MaterialBeam):
    """Holds material definition for Euler Bernoulli beams."""

    def __init__(self, **kwargs):
        MaterialBeam.__init__(self,
            material_string='MAT_BeamKirchhoffTorsionFreeElastHyper', **kwargs)

    def _get_dat(self):
        """Return the line for this material."""
        string = 'MAT {} {} YOUNG {} DENS {} CROSSAREA {} MOMIN {}'
        return string.format(
            self.n_global,
            self.material_string,
            self.youngs_modulus,
            self.density,
            self.area,
            self.mom2
            )
