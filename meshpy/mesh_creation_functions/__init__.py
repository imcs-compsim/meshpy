# -*- coding: utf-8 -*-
"""
This module stores all functions to create beam meshes. From simple lines to
complex stent craft structures.
"""

# Basic geometry functions
from .beam_basic_geometry import (create_beam_mesh_line,
    create_beam_mesh_arc_segment, create_beam_mesh_arc_segment_2d)

# Parametric curve.
from .beam_curve import create_beam_mesh_curve

# Fibers in rectangle.
from .beam_fibers_in_rectangle import create_fibers_in_rectangle

# Honeycomb.
from .beam_honeycomb import (create_beam_mesh_honeycomb_flat,
    create_beam_mesh_honeycomb)

# Honeycomb.
from .beam_stent import create_beam_mesh_stent, create_beam_mesh_stent_flat

# Wire.
from .beam_wire import create_wire_fibers

# Define the items that will be exported by default.
__all__ = [
    # Base geometry.
    'create_beam_mesh_line',
    'create_beam_mesh_arc_segment',
    'create_beam_mesh_arc_segment_2d',
    # Parametric curve.
    'create_beam_mesh_curve',
    # Honeycomb.
    'create_beam_mesh_honeycomb_flat',
    'create_beam_mesh_honeycomb',
    # Stent
    'create_beam_mesh_stent',
    # Fibers in rectangle
    'create_fibers_in_rectangle',
    # Wire
    'create_wire_fibers'
    ]
