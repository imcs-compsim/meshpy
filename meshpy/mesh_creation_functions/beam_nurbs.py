# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# MeshPy: A beam finite element input generator
#
# MIT License
#
# Copyright (c) 2021 Ivo Steinbrecher
#                    Institute for Mathematics and Computer-Based Simulation
#                    Universitaet der Bundeswehr Muenchen
#                    https://www.unibw.de/imcs-en
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
Create a beam filament from a nurbs curve represented with NURBS-Python
(geomdl).
"""

# Python packages.
import numpy as np

# Meshpy modules.
from ..conf import mpy
from .beam_curve import create_beam_mesh_curve


def create_beam_mesh_from_nurbs(mesh, beam_object, material, curve, **kwargs):
    """
    Generate a beam from a nurbs curve.

    Args
    ----
    mesh: Mesh
        Mesh that the curve will be added to.
    beam_object: Beam
        Class of beam that will be used for this line.
    material: Material
        Material for this line.
    curve: geomdl object
        Curve that is used to describe the beam centerline.

    **kwargs (for all of them look into create_beam_mesh_function)
    ----
    n_el: int
        Number of equally spaces beam elements along the line.

    Return
    ----
    return_set: GeometryName
        Set with the 'start' and 'end' node of the curve. Also a 'line' set
        with all nodes of the curve.
    """

    # Get start and end values of the curve parameter space.
    curve_start = np.min(curve.knotvector)
    curve_end = np.max(curve.knotvector)

    def eval_r(t):
        """Evaluate the position along the curve."""
        return np.array(curve.derivatives(u=t, order=0)[0])

    def eval_rp(t):
        """Evaluate the derivative along the curve."""
        return np.array(curve.derivatives(u=t, order=1)[1])

    def function(t):
        """Convert the curve to a function that can be used for beam
        generation."""

        # Due to numeric tolerances it is possible that the position has to be
        # evaluated outside the interval.
        if curve_start <= t and t <= curve_end:
            return eval_r(t)
        elif np.abs(curve_start - t) < mpy.eps_pos:
            return eval_r(curve_start)
        elif np.abs(t - curve_end) < mpy.eps_pos:
            return eval_r(curve_end)
        else:
            raise ValueError(
                "Can not evaluate the curve function outside of the interval (plus tolerances)."
            )

    def jacobian(t):
        """Convert the spline to a Jacobian function that can be used for curve
        generation."""

        # In the curve integration it is possible that the Jacobian is
        # evaluated outside the interval. In that case use the values at the
        # limits of the interval.
        if curve_start <= t and t <= curve_end:
            return eval_rp(t)
        elif t < curve_start:
            return eval_rp(curve_start)
        elif curve_end < t:
            return eval_rp(curve_end)

    # Create the beams.
    return create_beam_mesh_curve(
        mesh,
        beam_object,
        material,
        function,
        [curve_start, curve_end],
        function_derivative=jacobian,
        **kwargs
    )
