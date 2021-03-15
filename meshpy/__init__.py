# -*- coding: utf-8 -*-
"""
This module defines classes and functions to create and edit a Baci input file.
"""

# Global configuration object.
from .conf import mpy

# Utility functions.
from .utility import flatten, find_close_nodes, find_close_points

# 3D rotations for nodes.
from .rotation import Rotation

# Mesh items.
from .base_mesh_item import BaseMeshItem
from .function import Function
from .material import (MaterialBeam, MaterialReissner, MaterialKirchhoff,
    MaterialEulerBernoulli)
from .element_beam import Beam3rHerm2Line3, Beam3rLine2Line2, Beam3k, Beam3eb
from .geometry_set import GeometrySet

# Boundary conditions and couplings for geometry in the mesh.
from .boundary_condition import BoundaryCondition
from .coupling import Coupling

# The mesh class itself and the input file classes.
from .mesh import Mesh
from .inputfile import InputFile, InputSection

# Functions to set default header options.
from .header_functions import (get_comment, get_yes_no, set_header_static,
    set_runtime_output, set_beam_to_solid_meshtying)

# Define the itCouplingems that will be exported by default.
__all__ = [
    # Option object.
    'mpy',
    # Basic stuff.
    'Rotation', 'BaseMeshItem', 'Function', 'MaterialReissner',
    'MaterialKirchhoff', 'MaterialBeam', 'GeometrySet', 'BoundaryCondition',
    'Coupling', 'MaterialEulerBernoulli',
    # Mesh items.
    'Beam3rHerm2Line3', 'Beam3rLine2Line2', 'Beam3k', 'Mesh', 'InputFile',
    'InputSection', 'Beam3eb',
    # Header functions.
    'set_header_static', 'set_runtime_output', 'set_beam_to_solid_meshtying'
    ]
