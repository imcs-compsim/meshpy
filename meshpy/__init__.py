# -*- coding: utf-8 -*-
"""
This module defines classes and functions to create and edit a Baci input file.
"""

# Global configuration object.
from .conf import mpy

# 3D rotations for nodes.
from .rotation import Rotation

# Mesh items.
from .function import Function
from .material import (MaterialBeam, MaterialReissner, MaterialKirchhoff,
    MaterialEulerBernoulli)
from .element_beam import Beam3rHerm2Lin3, Beam3k, Beam3eb
from .geometry_set import GeometrySet

# Boundary conditions and couplings for geometry in the mesh.
from .boundary_condition import BoundaryCondition
from .coupling import Coupling

# The mesh class itself and the input file classes.
from .mesh import Mesh
from .inputfile import InputFile, InputSection

# Define the itCouplingems that will be exported by default.
__all__ = [
    # Option object.
    'mpy',
    # Basic stuff.
    'Rotation', 'Function', 'MaterialReissner', 'MaterialKirchhoff',
    'MaterialBeam', 'GeometrySet', 'BoundaryCondition', 'Coupling',
    'MaterialEulerBernoulli',
    # Mesh items
    'Beam3rHerm2Lin3', 'Beam3k', 'Mesh', 'InputFile', 'InputSection', 'Beam3eb'
    ]
