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
        super().__init__(data=data, is_dat=is_dat, **kwargs)

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
        super().__init__(**kwargs)

        self.radius = radius
        self.material_string = material_string
        self.youngs_modulus = youngs_modulus
        self.nu = nu
        self.density = density
        self.radius = radius
        self.area = None
        self.mom2 = None
        self.mom3 = None
        self.polar = None

    def calc_area_stiffness(self):
        """
        Calculate the relevant stiffness terms and the area for the given beam.
        """
        area = 4 * self.radius**2 * np.pi * 0.25
        mom2 = self.radius**4 * np.pi * 0.25
        mom3 = mom2
        polar = mom2 + mom3
        return area, mom2, mom3, polar


class MaterialReissner(MaterialBeam):
    """Holds material definition for Reissner beams."""

    def __init__(self, shear_correction=1, **kwargs):
        super().__init__(material_string='MAT_BeamReissnerElastHyper',
            **kwargs)

        # Shear factor for Reissner beam.
        self.shear_correction = shear_correction

    def _get_dat(self):
        """Return the line for this material."""
        if (self.area is None
                and self.mom2 is None
                and self.mom3 is None
                and self.polar is None):
            area, mom2, mom3, polar = self.calc_area_stiffness()
        elif (self.area is not None
                and self.mom2 is not None
                and self.mom3 is not None
                and self.polar is not None):
            area = self.area
            mom2 = self.mom2
            mom3 = self.mom3
            polar = self.polar
        else:
            raise ValueError('Either all relevant material parameters are set '
                + 'by the user, or a circular cross-section will be assumed. '
                + 'A combination is not possible')
        string = 'MAT {} {} YOUNG {} POISSONRATIO {} DENS {} CROSSAREA {} '
        string += 'SHEARCORR {} MOMINPOL {} MOMIN2 {} MOMIN3 {}'
        return string.format(
            self.n_global,
            self.material_string,
            self.youngs_modulus,
            self.nu,
            self.density,
            area,
            self.shear_correction,
            polar,
            mom2,
            mom3
            )


class MaterialKirchhoff(MaterialBeam):
    """Holds material definition for Kirchhoff beams."""

    def __init__(self, **kwargs):
        super().__init__(material_string='MAT_BeamKirchhoffElastHyper',
            **kwargs)

    def _get_dat(self):
        """Return the line for this material."""
        if (self.area is None
                and self.mom2 is None
                and self.mom3 is None
                and self.polar is None):
            area, mom2, mom3, polar = self.calc_area_stiffness()
        elif (self.area is not None
                and self.mom2 is not None
                and self.mom3 is not None
                and self.polar is not None):
            area = self.area
            mom2 = self.mom2
            mom3 = self.mom3
            polar = self.polar
        else:
            raise ValueError('Either all relevant material parameters are set '
                + 'by the user, or a circular cross-section will be assumed. '
                + 'A combination is not possible')
        string = 'MAT {} {} YOUNG {} SHEARMOD {} DENS {} CROSSAREA {} '
        string += 'MOMINPOL {} MOMIN2 {} MOMIN3 {}'
        return string.format(
            self.n_global,
            self.material_string,
            self.youngs_modulus,
            self.youngs_modulus / (2. * (1. + self.nu)),
            self.density,
            area,
            polar,
            mom2,
            mom3
            )


class MaterialEulerBernoulli(MaterialBeam):
    """Holds material definition for Euler Bernoulli beams."""

    def __init__(self, **kwargs):
        super().__init__(
            material_string='MAT_BeamKirchhoffTorsionFreeElastHyper', **kwargs)

    def _get_dat(self):
        """Return the line for this material."""
        area, mom2, _mom3, _polar = self.calc_area_stiffness()
        if (self.area is None and self.mom2 is None):
            area, mom2, _mom3, _polar = self.calc_area_stiffness()
        elif (self.area is not None and self.mom2 is not None):
            area = self.area
            mom2 = self.mom2
        else:
            raise ValueError('Either all relevant material parameters are set '
                + 'by the user, or a circular cross-section will be assumed. '
                + 'A combination is not possible')
        string = 'MAT {} {} YOUNG {} DENS {} CROSSAREA {} MOMIN {}'
        return string.format(
            self.n_global,
            self.material_string,
            self.youngs_modulus,
            self.density,
            area,
            mom2
            )
