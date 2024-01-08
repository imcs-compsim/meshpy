# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# MeshPy: A beam finite element input generator
#
# MIT License
#
# Copyright (c) 2018-2024
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
# -----------------------------------------------------------------------------
"""
This module stores all functions to create beam meshes. From simple lines to
complex stent craft structures.
"""

# Basic geometry functions
from .beam_basic_geometry import (
    create_beam_mesh_line,
    create_beam_mesh_arc_segment_via_axis,
    create_beam_mesh_arc_segment_via_rotation,
    create_beam_mesh_arc_segment_2d,
    create_beam_mesh_line_at_node,
    create_beam_mesh_arc_at_node,
)

# Parametric curve.
from .beam_curve import create_beam_mesh_curve

# Fibers in rectangle.
from .beam_fibers_in_rectangle import create_fibers_in_rectangle

# Honeycomb.
from .beam_honeycomb import create_beam_mesh_honeycomb_flat, create_beam_mesh_honeycomb

# Create a beam mesh along a NURBS curve.
from .beam_nurbs import create_beam_mesh_from_nurbs

# Honeycomb.
from .beam_stent import create_beam_mesh_stent

# Wire.
from .beam_wire import create_wire_fibers

# Generic NURBS mesh creation function
from .nurbs_generic import add_geomdl_nurbs_to_mesh

# NURBS geometry functions
from .nurbs_geometries import (
    create_nurbs_hollow_cylinder_segment_2d,
    create_nurbs_flat_plate_2d,
    create_nurbs_brick,
    create_nurbs_sphere_surface,
    create_nurbs_hemisphere_surface,
)

# Define the items that will be exported by default.
__all__ = [
    # Base geometry.
    "create_beam_mesh_line",
    "create_beam_mesh_arc_segment_via_axis",
    "create_beam_mesh_arc_segment_via_rotation",
    "create_beam_mesh_arc_segment_2d",
    "create_beam_mesh_line_at_node",
    "create_beam_mesh_arc_at_node",
    # Parametric curve.
    "create_beam_mesh_curve",
    # Honeycomb.
    "create_beam_mesh_honeycomb_flat",
    "create_beam_mesh_honeycomb",
    # Stent
    "create_beam_mesh_stent",
    # Spline based geometry
    "create_beam_mesh_from_nurbs",
    # Fibers in rectangle
    "create_fibers_in_rectangle",
    # Wire
    "create_wire_fibers",
    # Generic NURBS mesh creation function
    "add_geomdl_nurbs_to_mesh",
    # NURBS geometry functions
    "create_nurbs_hollow_cylinder_segment_2d",
    "create_nurbs_flat_plate_2d",
    "create_nurbs_brick",
    "create_nurbs_sphere_surface",
    "create_nurbs_hemisphere_surface",
]
