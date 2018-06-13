# import python modules
import numpy as np

# import meshpy modules
from . import BaseMeshItem


class Material(BaseMeshItem):
    """ Holds material definition for beams and solids. """
    
    def __init__(
            self,
            material_string,
            youngs_modulus,
            nu,
            density,
            diameter,
            shear_correction=0.75
            ):
        
        BaseMeshItem.__init__(self, data=None, is_dat=False)
        
        self.material_string = material_string
        self.youngs_modulus = youngs_modulus
        self.nu = nu
        self.density = density
        self.diameter = diameter
        self.area = diameter**2 * np.pi * 0.25
        self.mom2 = (diameter*0.5)**4 * np.pi * 0.25
        self.mom3 = self.mom2
        self.polar = self.mom2 + self.mom3
        self.shear_correction = shear_correction
        
    
    def _get_dat(self):
        """ Return the line for the .dat file. """
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