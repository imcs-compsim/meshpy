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
"""Create a beam filament from a NURBS curve represented with splinepy."""

import numpy as _np

from meshpy.core.conf import mpy as _mpy
from meshpy.mesh_creation_functions.beam_curve import (
    create_beam_mesh_curve as _create_beam_mesh_curve,
)


def get_nurbs_curve_function_and_jacobian_for_integration(curve, tol=None):
    """Return function objects for evaluating the curve and the derivative.
    These functions are used in the curve integration. It can happen that the
    integration algorithm has to evaluate the curve outside of the defined
    domain. This usually leads to errors in common NURBS packages. Therefore,
    we check for this evaluation outside of the parameter domain here and
    perform a linear extrapolation.

    Args
    ----
    curve: splinepy object
        Curve that is used to describe the beam centerline.
    tol: float
        Tolerance for checking if point is close to the start or end of the
        interval. If None is given, use the default tolerance from mpy.

    Return
    ----
    (function, jacobian, curve_start, curve_end):
        function:
            Function for evaluating a position on the curve
        jacobian:
            Function for evaluating the tangent along the curve
        curve_start:
            Parameter coordinate for the start for the NURBS curve
        curve_end:
            Parameter coordinate for the end for the NURBS curve
    """

    if tol is None:
        tol = _mpy.eps_pos

    knot_vector = curve.knot_vectors[0]
    curve_start = _np.min(knot_vector)
    curve_end = _np.max(knot_vector)

    def eval_r(t):
        """Evaluate the position along the curve."""
        return curve.evaluate([[t]])[0]

    def eval_rp(t):
        """Evaluate the derivative along the curve."""
        return curve.derivative([[t]], orders=[1])[0]

    def function(t):
        """Convert the curve to a function that can be used for beam
        generation."""

        if curve_start <= t <= curve_end:
            return eval_r(t)
        elif t < curve_start and _np.abs(t - curve_start) < tol:
            diff = t - curve_start
            return eval_r(curve_start) + diff * eval_rp(curve_start)
        elif t > curve_end and _np.abs(t - curve_end) < tol:
            diff = t - curve_end
            return eval_r(curve_end) + diff * eval_rp(curve_end)
        raise ValueError(
            "Can not evaluate the curve function outside of the interval (plus tolerances).\n"
            f"Abs diff start: {_np.abs(curve_start - t)}\nAbs diff end: {_np.abs(t - curve_end)}"
        )

    def jacobian(t):
        """Convert the spline to a Jacobian function that can be used for curve
        generation.

        There is no tolerance here, since the integration algorithms
        sometimes evaluate the derivative far outside the interval.
        """

        if curve_start <= t <= curve_end:
            return eval_rp(t)
        elif t < curve_start:
            return eval_rp(curve_start)
        elif curve_end < t:
            return eval_rp(curve_end)
        raise ValueError("Should not happen")

    return function, jacobian, curve_start, curve_end


def create_beam_mesh_from_nurbs(
    mesh, beam_object, material, curve, *, tol=None, **kwargs
):
    """Generate a beam from a NURBS curve.

    Args
    ----
    mesh: Mesh
        Mesh that the curve will be added to.
    beam_object: Beam
        Class of beam that will be used for this line.
    material: Material
        Material for this line.
    curve: splinepy object
        Curve that is used to describe the beam centerline.
    tol: float
        Tolerance for checking if point is close to the start or end of the
        interval. If None is given, use the default tolerance from mpy.

    **kwargs (for all of them look into create_beam_mesh_function)
    ----
    n_el: int
        Number of equally spaced beam elements along the line. Defaults to 1.
        Mutually exclusive with l_el.
    l_el: float
        Desired length of beam elements. Mutually exclusive with n_el.
        Be aware, that this length might not be achieved, if the elements are
        warped after they are created.

    Return:
        Return value from create_beam_mesh_function
    """

    (
        function,
        jacobian,
        curve_start,
        curve_end,
    ) = get_nurbs_curve_function_and_jacobian_for_integration(curve, tol=tol)

    # Create the beams
    return _create_beam_mesh_curve(
        mesh,
        beam_object,
        material,
        function,
        [curve_start, curve_end],
        function_derivative=jacobian,
        **kwargs,
    )
