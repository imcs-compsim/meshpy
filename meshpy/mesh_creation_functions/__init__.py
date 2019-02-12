# -*- coding: utf-8 -*-
"""
This module stores all functions to create beam meshes. From simple lines to
complex stent craft structures.
"""

# Basic geometry functions
from .beam_basic_geometry import create_beam_mesh_line, \
    create_beam_mesh_arc_segment

# Parametric curve.
from .beam_curve import create_beam_mesh_curve

# Honeycomb.
from .beam_honeycomb import create_beam_mesh_honeycomb_flat, \
    create_beam_mesh_honeycomb

# Honeycomb.
from .beam_stent import create_beam_mesh_stent, create_beam_mesh_stent_flat

# Define the items that will be exported by default.
__all__ = [
    # Base geometry.
    'create_beam_mesh_line',
    'create_beam_mesh_arc_segment',
    # Parametric curve.
    'create_beam_mesh_curve',
    # Honeycomb.
    'create_beam_mesh_honeycomb_flat',
    'create_beam_mesh_honeycomb',
    # Stent
    'create_beam_mesh_stent'
    ]
