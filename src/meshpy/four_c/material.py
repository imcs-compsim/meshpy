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
"""This file implements materials for 4C beams and solids."""

from meshpy.core.material import Material, MaterialBeam


class MaterialReissner(MaterialBeam):
    """Holds material definition for Reissner beams."""

    def __init__(self, shear_correction=1, **kwargs):
        super().__init__(material_string="MAT_BeamReissnerElastHyper", **kwargs)

        # Shear factor for Reissner beam.
        self.shear_correction = shear_correction

    def _get_dat(self):
        """Return the line for this material."""
        if (
            self.area is None
            and self.mom2 is None
            and self.mom3 is None
            and self.polar is None
        ):
            area, mom2, mom3, polar = self.calc_area_stiffness()
        elif (
            self.area is not None
            and self.mom2 is not None
            and self.mom3 is not None
            and self.polar is not None
        ):
            area = self.area
            mom2 = self.mom2
            mom3 = self.mom3
            polar = self.polar
        else:
            raise ValueError(
                "Either all relevant material parameters are set "
                "by the user, or a circular cross-section will be assumed. "
                "A combination is not possible"
            )
        string = "MAT {} {} YOUNG {} POISSONRATIO {} DENS {} CROSSAREA {} "
        string += "SHEARCORR {} MOMINPOL {} MOMIN2 {} MOMIN3 {}"
        if self.interaction_radius is not None:
            string += f" INTERACTIONRADIUS {self.interaction_radius}"
        return string.format(
            self.i_global,
            self.material_string,
            self.youngs_modulus,
            self.nu,
            self.density,
            area,
            self.shear_correction,
            polar,
            mom2,
            mom3,
        )


class MaterialReissnerElastoplastic(MaterialReissner):
    """Holds elasto-plastic material definition for Reissner beams."""

    def __init__(
        self,
        *,
        yield_moment=None,
        isohardening_modulus_moment=None,
        torsion_plasticity=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.material_string = "MAT_BeamReissnerElastPlastic"

        if yield_moment is None or isohardening_modulus_moment is None:
            raise ValueError(
                "The yield moment and the isohardening modulus for moments must be specified "
                "for plasticity."
            )

        self.yield_moment = yield_moment
        self.isohardening_modulus_moment = isohardening_modulus_moment
        self.torsion_plasticity = torsion_plasticity

    def _get_dat(self):
        """Return the line for this material."""
        super_dat = super()._get_dat()
        string = super_dat + " YIELDM {} ISOHARDM {} TORSIONPLAST {}"
        return string.format(
            self.yield_moment,
            self.isohardening_modulus_moment,
            1 if self.torsion_plasticity else 0,
        )


class MaterialKirchhoff(MaterialBeam):
    """Holds material definition for Kirchhoff beams."""

    def __init__(self, is_fad=False, **kwargs):
        super().__init__(material_string="MAT_BeamKirchhoffElastHyper", **kwargs)
        self.is_fad = is_fad

    def _get_dat(self):
        """Return the line for this material."""
        if (
            self.area is None
            and self.mom2 is None
            and self.mom3 is None
            and self.polar is None
        ):
            area, mom2, mom3, polar = self.calc_area_stiffness()
        elif (
            self.area is not None
            and self.mom2 is not None
            and self.mom3 is not None
            and self.polar is not None
        ):
            area = self.area
            mom2 = self.mom2
            mom3 = self.mom3
            polar = self.polar
        else:
            raise ValueError(
                "Either all relevant material parameters are set "
                "by the user, or a circular cross-section will be assumed. "
                "A combination is not possible"
            )
        string = "MAT {} {} YOUNG {} SHEARMOD {} DENS {} CROSSAREA {} "
        string += "MOMINPOL {} MOMIN2 {} MOMIN3 {} FAD {}"
        string = string.format(
            self.i_global,
            self.material_string,
            self.youngs_modulus,
            self.youngs_modulus / (2.0 * (1.0 + self.nu)),
            self.density,
            area,
            polar,
            mom2,
            mom3,
            # TODO: replace this with a common utils function
            "yes" if self.is_fad else "no",
        )
        if self.interaction_radius is not None:
            string += f" INTERACTIONRADIUS {self.interaction_radius}"
        return string


class MaterialEulerBernoulli(MaterialBeam):
    """Holds material definition for Euler Bernoulli beams."""

    def __init__(self, **kwargs):
        super().__init__(
            material_string="MAT_BeamKirchhoffTorsionFreeElastHyper", **kwargs
        )

    def _get_dat(self):
        """Return the line for this material."""
        area, mom2, _mom3, _polar = self.calc_area_stiffness()
        if self.area is None and self.mom2 is None:
            area, mom2, _mom3, _polar = self.calc_area_stiffness()
        elif self.area is not None and self.mom2 is not None:
            area = self.area
            mom2 = self.mom2
        else:
            raise ValueError(
                "Either all relevant material parameters are set "
                "by the user, or a circular cross-section will be assumed. "
                "A combination is not possible"
            )
        string = "MAT {} {} YOUNG {} DENS {} CROSSAREA {} MOMIN {}"
        return string.format(
            self.i_global,
            self.material_string,
            self.youngs_modulus,
            self.density,
            area,
            mom2,
        )


class MaterialString(Material):
    """Holds material definition that is defined by a string."""

    def __init__(self, material_string, **kwargs):
        super().__init__(**kwargs)
        self.material_string = material_string

    def _get_dat(self):
        """Return the line for this material."""
        string = "MAT {} {}"
        return string.format(self.i_global, self.material_string)


class MaterialSolid(Material):
    """Base class for a material for solids."""

    def __init__(
        self, material_string=None, youngs_modulus=-1.0, nu=0.0, density=0.0, **kwargs
    ):
        """Set the material values for a solid."""
        super().__init__(**kwargs)

        self.material_string = material_string
        self.youngs_modulus = youngs_modulus
        self.nu = nu
        self.density = density


class MaterialStVenantKirchhoff(MaterialSolid):
    """Holds material definition for StVenant Kirchhoff solids."""

    def __init__(self, **kwargs):
        super().__init__(material_string="MAT_Struct_StVenantKirchhoff", **kwargs)

    def _get_dat(self):
        """Return the line for this material."""

        string = "MAT {} {} YOUNG {} NUE {} DENS {}"
        return string.format(
            self.i_global,
            self.material_string,
            self.youngs_modulus,
            self.nu,
            self.density,
        )
