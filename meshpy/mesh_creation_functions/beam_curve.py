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
This file has functions to create a beam from a parametric curve.
"""

# Python packages.
import numpy as np

# Meshpy modules.
from ..conf import mpy
from ..rotation import Rotation


def create_beam_mesh_curve(mesh, beam_object, material, function, interval,
        *, function_rotation=None, **kwargs):
    """
    Generate a beam from a parametric curve. Integration along the beam is
    performed with scipy, and the gradient is calculated with autograd.

    Args
    ----
    mesh: Mesh
        Mesh that the curve will be added to.
    beam_object: Beam
        Class of beam that will be used for this line.
    material: Material
        Material for this line.
    function: function
        3D-parametric curve that represents the beam axis. If only a 2D
        point is returned, the triad creation is simplified. If
        mathematical functions are used, they have to come from the wrapper
        autograd.numpy.
    interval: [start end]
        Start and end values for the parameter of the curve.
    function_rotation: function
        If this argument is given, the triads are computed with this
        function, on the same interval as the position function. Must
        return a Rotation object.

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

    # Packages for AD and numerical integration.
    from autograd import jacobian
    import autograd.numpy as npAD
    import scipy.integrate as integrate
    import scipy.optimize as optimize

    # Check size of position function
    if len(function(interval[0])) == 2:
        is_3d_curve = False
    elif len(function(interval[0])) == 3:
        is_3d_curve = True
    else:
        raise ValueError('Function must return either 2d or 3d curve!')

    # Check rotation function.
    if function_rotation is None:
        is_rot_funct = False
    else:
        is_rot_funct = True

    # Check that the position is an np.array
    if not isinstance(function(float(interval[0])), np.ndarray):
        raise TypeError(
            'Function must be of type np.ndarray, got {}!'.format(
                type(function(float(interval[0])))
                ))

    # Get the derivative of the position function and the increment along
    # the curve.
    rp = jacobian(function)

    # Check which one of the boundaries is larger.
    if (interval[0] > interval[1]):
        # In this case rp needs to be negated.
        def rp_negative(t):
            return -(jacobian(function)(t))
        rp = rp_negative

    def ds(t):
        """Increment along the curve."""
        return npAD.linalg.norm(rp(t))

    def S(t, start_t=None, start_S=None):
        """
        Function that integrates the length until a parameter value.
        A speedup can be achieved by giving start_t and start_S, the
        parameter and Length at a known point.
        """
        if start_t is None and start_S is None:
            st = interval[0]
            sS = 0
        elif start_t is not None and start_S is not None:
            st = start_t
            sS = start_S
        else:
            raise ValueError('Input parameters are wrong!')
        return integrate.quad(ds, st, t)[0] + sS

    def get_t_along_curve(arc_length, t0, **kwargs):
        """
        Calculate the parameter t where the length along the curve is
        arc_length. t0 is the start point for the Newton iteration.
        """
        t_root = optimize.newton(lambda t: S(t, **kwargs) - arc_length, t0,
            fprime=ds)
        return t_root

    def get_beam_functions(length_a, length_b):
        """
        Return a function for the position and rotation along the beam
        axis.
        """

        # Length of the beam element in physical space.
        L = length_b - length_a

        def beam_function(xi):
            """
            Return position and rotation along the beam in the parameter
            coordinate xi.
            """

            # Global values for the start of the element.
            global t_temp, t_start_element, t2_temp

            # Parameter for xi.
            t = get_t_along_curve(
                length_a + 0.5 * (xi + 1) * L,
                t_start_element, start_t=t_start_element, start_S=length_a)
            t_temp = t

            # Position at xi.
            if is_3d_curve:
                pos = function(t)
            else:
                pos = np.zeros(3)
                pos[:2] = function(t)

            # Rotation at xi.
            if is_rot_funct:
                rot = function_rotation(t)
            else:
                if is_3d_curve:
                    rot = Rotation.from_basis(rp(t), t2_temp)
                else:
                    # The rotation simplifies in the 2d case.
                    rprime = rp(t)
                    rot = Rotation(
                        [0, 0, 1],
                        np.arctan2(rprime[1], rprime[0])
                        )

            if np.abs(xi - 1) < mpy.eps_pos:
                # Set start values for the next element.
                t_start_element = t_temp
                t2_temp = rot.get_rotation_matrix()[:, 1]

            # Return the needed values for beam creation.
            return (pos, rot)

        return beam_function

    # Now create the beam.
    # Get the length of the whole segment.
    length = S(interval[1])

    # Create the beams.
    global t_temp, t_start_element, t2_temp
    t_temp = interval[0]
    t_start_element = interval[0]

    # The first t2 basis is the one with the larger projection on rp.
    if is_3d_curve:
        rprime = rp(float(interval[0]))
        if abs(np.dot(rprime, [0, 0, 1])) < \
                abs(np.dot(rprime, [0, 1, 0])):
            t2_temp = [0, 0, 1]
        else:
            t2_temp = [0, 1, 0]

    # Create the beam in the mesh
    return mesh.create_beam_mesh_function(beam_object=beam_object,
        material=material, function_generator=get_beam_functions,
        interval=[0., length], **kwargs)
