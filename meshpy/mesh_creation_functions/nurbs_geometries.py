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
This file has functions to create NURBS geometries using Geomdl.
"""

# Python modules
import numpy as np

# Geomdl modules
from geomdl import NURBS
from geomdl import operations
from geomdl import compatibility as compat


def create_nurbs_hollow_cylinder_segment_2d(
    radius_in, radius_out, angle, *, n_ele_u=1, n_ele_v=1
):
    """
    Creates a patch of a 2 dimensional segment of a hollow cylinder.

    Args
    ----
    radius_in: double
        inner cylinder radius
    radius_out: double
        outer cylinder radius
    angle: double
        angle of the hollow cylinder section (radians)
    n_ele_u: int
        number of elements in the parametric u-direction
    n_ele_v: int
        number of elements in the parametric v-direction


    Return
    ----
    surf: geomdl object
        geomdl object that contains the surface information
    """

    # Check the validity of the input values:
    if radius_in >= radius_out:
        raise ValueError(
            "The external radius should be larger than the internal radius of a hollow cylinder."
        )
    if (angle > np.pi) or (angle < 0):
        raise ValueError(
            "The following algorithm for creating a hollow cylinder section is only valid for 0 < angle <= pi."
        )

    # Create a NURBS surface instance
    surf = NURBS.Surface()

    # Set degrees
    surf.degree_u = 2
    surf.degree_v = 2

    # Control points and set them to the surface
    p_size_u = 3
    p_size_v = 3

    # To calculate the internal control point cp_2 we have to calculate
    # the intersection of the tangents of the points cp_1 and cp_3.
    # As point cp_1 is always defined over the x axis, its tangent is simply x = radius
    # The equation of the tangent to the circle at the cp_3 point is:
    # y - cp_3y = -(cp_3x/cp_3y)*(x - cp_3x).

    # Obtaining the control points that define the external arc of the hollow cylinder
    cp_ext1 = [radius_out, 0.0, 0.0]
    cp_ext3 = [radius_out * np.cos(angle), radius_out * np.sin(angle), 0.0]
    cp_ext2 = [
        radius_out,
        -(cp_ext3[0] / cp_ext3[1]) * (radius_out - cp_ext3[0]) + cp_ext3[1],
        0.0,
    ]

    # Obtaining the control points that define the internal arc of the hollow cylinder
    cp_int1 = [radius_in, 0.0, 0.0]
    cp_int3 = [radius_in * np.cos(angle), radius_in * np.sin(angle), 0.0]
    cp_int2 = [
        radius_in,
        -(cp_int3[0] / cp_int3[1]) * (radius_in - cp_int3[0]) + cp_int3[1],
        0.0,
    ]

    # Obtaining the control points positioned in the middle of the internal and external arches
    cp_middle1 = [(cp_ext1[0] + cp_int1[0]) / 2, (cp_ext1[1] + cp_int1[1]) / 2, 0.0]
    cp_middle2 = [(cp_ext2[0] + cp_int2[0]) / 2, (cp_ext2[1] + cp_int2[1]) / 2, 0.0]
    cp_middle3 = [(cp_ext3[0] + cp_int3[0]) / 2, (cp_ext3[1] + cp_int3[1]) / 2, 0.0]

    ctrlpts = [
        cp_ext1,
        cp_ext2,
        cp_ext3,
        cp_middle1,
        cp_middle2,
        cp_middle3,
        cp_int1,
        cp_int2,
        cp_int3,
    ]

    weights = [
        1.0,
        np.cos(angle / 2),
        1.0,
        1.0,
        np.cos(angle / 2),
        1.0,
        1.0,
        np.cos(angle / 2),
        1.0,
    ]

    t_ctrlptsw = compat.combine_ctrlpts_weights(ctrlpts, weights)
    n_ctrlptsw = compat.flip_ctrlpts_u(t_ctrlptsw, p_size_u, p_size_v)

    surf.ctrlpts_size_u = p_size_u
    surf.ctrlpts_size_v = p_size_v
    surf.ctrlptsw = n_ctrlptsw

    surf.knotvector_u = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    surf.knotvector_v = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

    do_uniform_knot_refinement_surface(surf, n_ele_u, n_ele_v)

    return surf


def create_nurbs_flat_plate_2d(width, length, *, n_ele_u=1, n_ele_v=1):
    """
    Creates a patch of a 2 dimensional flat plate.

    Args
    ----
    width: double
        dimension of the plate in the x-direction
    length: double
        dimension of the plate in the y-direction
    n_ele_u: int
        number of elements in the parametric u-direction
    n_ele_v: int
        number of elements in the parametric v-direction


    Return
    ----
    surf: geomdl object
        geomdl object that contains the surface information
    """

    # Create a NURBS surface instance
    surf = NURBS.Surface()

    # Set degrees
    surf.degree_u = 2
    surf.degree_v = 2

    # Control points and set them to the surface
    p_size_u = 3
    p_size_v = 3

    ctrlpts = [
        [-width / 2, -length / 2, 0.0],
        [0.0, -length / 2, 0.0],
        [width / 2, -length / 2, 0.0],
        [-width / 2, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [width / 2, 0.0, 0.0],
        [-width / 2, length / 2, 0.0],
        [0.0, length / 2, 0.0],
        [width / 2, length / 2, 0.0],
    ]

    weights = [1.0] * 9

    t_ctrlptsw = compat.combine_ctrlpts_weights(ctrlpts, weights)
    n_ctrlptsw = compat.flip_ctrlpts_u(t_ctrlptsw, p_size_u, p_size_v)

    surf.ctrlpts_size_u = p_size_u
    surf.ctrlpts_size_v = p_size_v
    surf.ctrlptsw = n_ctrlptsw

    surf.knotvector_u = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    surf.knotvector_v = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

    do_uniform_knot_refinement_surface(surf, n_ele_u, n_ele_v)

    return surf


def create_nurbs_sphere_surface(radius, n_ele_u=1, n_ele_v=1):
    """
    Generates a patch of a sphere as a NURBS surface.
    This function constructs a segment of a spherical
    surface using Non-Uniform Rational B-Splines (NURBS)
    based on the specified radius and the number of elements
    in the parametric u and v directions.

    Args
    ---
    radius: double
        radius of the sphere
    n_ele_u: int
        number of elements in the parametric u-direction
    n_ele_v: int
        number of elements in the parametric v-direction

    Return
    ----
    surf: geomdl object
        geomdl object that contains the surface information
    """

    # Create a NURBS surface instance
    surf = NURBS.Surface()

    # Set degrees
    surf.degree_u = 2
    surf.degree_v = 2

    # Control points and set them to the surface
    p_size_u = 3
    p_size_v = 3

    dummy = 6.0 * (
        5.0 / 12.0
        + 0.5 * np.sqrt(2.0 / 3.0)
        - 0.25 / np.sqrt(3.0)
        - 0.5 * np.sqrt(2.0 / 3.0) * np.sqrt(3.0) / 2.0
    )

    ctrlpts = [
        [
            2.0 * radius / np.sqrt(6.0) * -np.sin(1.0 / 4.0 * np.pi),
            -radius / np.sqrt(3.0),
            2.0 * radius / np.sqrt(6.0) * np.cos(1.0 / 4.0 * np.pi),
        ],
        [
            radius * np.sqrt(3.0) / (np.sqrt(2.0) * 2.0) * np.cos(1.0 / 4.0 * np.pi)
            + radius * np.sqrt(3.0) / (np.sqrt(2.0) * 2.0) * -np.sin(1.0 / 4.0 * np.pi),
            -np.sqrt(3.0) / 2 * radius,
            radius * np.sqrt(3.0) / (np.sqrt(2.0) * 2.0) * np.cos(1.0 / 4.0 * np.pi)
            + radius * np.sqrt(3.0) / (np.sqrt(2.0) * 2.0) * np.sin(1.0 / 4.0 * np.pi),
        ],
        [
            2.0 * radius / np.sqrt(6.0) * np.cos(1.0 / 4.0 * np.pi),
            -radius / np.sqrt(3.0),
            2.0 * radius / np.sqrt(6.0) * np.sin(1.0 / 4.0 * np.pi),
        ],
        [
            radius * np.sqrt(6.0) / 2.0 * -np.sin(1.0 / 4.0 * np.pi),
            0.0,
            radius * np.sqrt(6.0) / 2.0 * np.cos(1.0 / 4.0 * np.pi),
        ],
        [
            radius * dummy * np.sqrt(2.0) / 2.0 * np.cos(1.0 / 4.0 * np.pi)
            + radius * dummy * np.sqrt(2.0) / 2.0 * -np.sin(1.0 / 4.0 * np.pi),
            0.0,
            radius * dummy * np.sqrt(2.0) / 2.0 * np.cos(1.0 / 4.0 * np.pi)
            + radius * dummy * np.sqrt(2.0) / 2.0 * np.sin(1.0 / 4.0 * np.pi),
        ],
        [
            radius * np.sqrt(6.0) / 2.0 * np.cos(1.0 / 4.0 * np.pi),
            0.0,
            radius * np.sqrt(6.0) / 2.0 * np.sin(1.0 / 4.0 * np.pi),
        ],
        [
            2.0 * radius / np.sqrt(6.0) * -np.sin(1.0 / 4.0 * np.pi),
            2.0 * radius / np.sqrt(6.0) * np.cos(1.0 / 4.0 * np.pi),
            radius / np.sqrt(3.0),
        ],
        [
            radius * np.sqrt(3.0) / (np.sqrt(2.0) * 2.0) * np.cos(1.0 / 4.0 * np.pi)
            + radius * np.sqrt(3.0) / (np.sqrt(2.0) * 2.0) * -np.sin(1.0 / 4.0 * np.pi),
            np.sqrt(3.0) / 2 * radius,
            radius * np.sqrt(3.0) / (np.sqrt(2.0) * 2.0) * np.cos(1.0 / 4.0 * np.pi)
            + radius * np.sqrt(3.0) / (np.sqrt(2.0) * 2.0) * np.sin(1.0 / 4.0 * np.pi),
        ],
        [
            2.0 * radius / np.sqrt(6.0) * np.cos(1.0 / 4.0 * np.pi),
            radius / np.sqrt(3.0),
            2.0 * radius / np.sqrt(6.0) * np.sin(1.0 / 4.0 * np.pi),
        ],
    ]

    weights = [
        1.0,
        2.0 / np.sqrt(6.0),
        1.0,
        2.0 / np.sqrt(6.0),
        2.0 / 3.0,
        2.0 / np.sqrt(6.0),
        1.0,
        2.0 / np.sqrt(6.0),
        1.0,
    ]

    t_ctrlptsw = compat.combine_ctrlpts_weights(ctrlpts, weights)
    n_ctrlptsw = compat.flip_ctrlpts_u(t_ctrlptsw, p_size_u, p_size_v)

    surf.ctrlpts_size_u = p_size_u
    surf.ctrlpts_size_v = p_size_v
    surf.ctrlptsw = n_ctrlptsw

    surf.knotvector_u = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    surf.knotvector_v = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

    do_uniform_knot_refinement_surface(surf, n_ele_u, n_ele_v)

    return surf


def create_nurbs_hemisphere_surface(radius, n_ele_uv=1):
    """
    Generates a hemisphere as a NURBS surface. This function constructs five segments that represent
    the surface of a hemisphere using Non-Uniform Rational B-Splines (NURBS) based on the specified 
    radius and the number of elements in the parametric u and v directions. To secure the connectivity between
    surfaces, all surfaces must have the same parametric representation in any parametric direction.
    Therefore, the number of elements in u- and v- directions must be the same

    This function generates a list of five NURBS geomdl objects.

    Args
    ---
    radius: double
        Radius of the hemisphere
    n_ele_uv: int
        Number of elements in the parametric u- and v- directions

    Return
    ----
    list: list(geomdl object)
        A list of geomdl objects that contains the surface information
    """

    # Create the first section of the hemisphere
    hemisphere_1 = create_nurbs_sphere_surface(radius, n_ele_u=1, n_ele_v=1)

    # Create a temporary section by rotating and translating the
    # first section of the hemisphere
    temp_hemisphere = operations.rotate(hemisphere_1, 90, axis=1)
    temp_hemisphere = operations.translate(
        temp_hemisphere,
        (0, 0, -2.0 * radius / np.sqrt(6.0) * np.sin(1.0 / 4.0 * np.pi) * 2),
    )

    # To create the hemisphere it is necessary to split the temporary
    # sphere section in two pieces. This split is done in u = 0.5. After
    # the split, the second surface is taken as it will be
    # adjacent to the first section of the sphere
    cut_section_sphere = operations.split_surface_u(temp_hemisphere, param=0.5)
    hemisphere_2 = cut_section_sphere[1]

    translation_component = radius * np.sin(1.0 / 4.0 * np.pi) * 2
    # Create the third section. Rotate and translate it accordingly
    hemisphere_3 = operations.rotate(hemisphere_2, 90, axis=2)
    hemisphere_3 = operations.translate(hemisphere_3, (translation_component, 0, 0))

    # Create the forth section. Rotate and translate it accordingly
    hemisphere_4 = operations.rotate(hemisphere_3, 90, axis=2)
    hemisphere_4 = operations.translate(hemisphere_4, (0, translation_component, 0))

    # Create the fifth section. Rotate and translate it accordingly
    hemisphere_5 = operations.rotate(hemisphere_4, 90, axis=2)
    hemisphere_5 = operations.translate(hemisphere_5, (-translation_component, 0, 0))

    patches = [hemisphere_1, hemisphere_2, hemisphere_3, hemisphere_4, hemisphere_5]
    for patch in patches:
        do_uniform_knot_refinement_surface(patch, n_ele_uv, n_ele_uv)
    return patches


def create_nurbs_brick(width, length, height, *, n_ele_u=1, n_ele_v=1, n_ele_w=1):
    """
    Creates a patch of a 3 dimensional brick.

    Args
    ----
    width: double
        dimension of the plate in the x-direction
    length: double
        dimension of the plate in the y-direction
    height: double
        dimension of the plate in the z-direction
    n_ele_u: int
        number of elements in the parametric u-direction
    n_ele_v: int
        number of elements in the parametric v-direction
    n_ele_w: int
        number of elements in the parametric w-direction


    Return
    ----
    vol: geomdl object
        geomdl object that contains the volume information
    """

    # Create a NURBS volume instance
    vol = NURBS.Volume()

    # Set degrees
    vol.degree_u = 2
    vol.degree_v = 2
    vol.degree_w = 2

    # Create control points and set them to the volume
    cp_size_u = 3
    cp_size_v = 3
    cp_size_w = 3

    ctrlpts = [
        [-width / 2, -length / 2, -height / 2],
        [-width / 2, 0.0, -height / 2],
        [-width / 2, length / 2, -height / 2],
        [0.0, -length / 2, -height / 2],
        [0.0, 0.0, -height / 2],
        [0.0, length / 2, -height / 2],
        [width / 2, -length / 2, -height / 2],
        [width / 2, 0.0, -height / 2],
        [width / 2, length / 2, -height / 2],
        [-width / 2, -length / 2, 0.0],
        [-width / 2, 0.0, 0.0],
        [-width / 2, length / 2, 0.0],
        [0.0, -length / 2, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, length / 2, 0.0],
        [width / 2, -length / 2, 0.0],
        [width / 2, 0.0, 0.0],
        [width / 2, length / 2, 0.0],
        [-width / 2, -length / 2, height / 2],
        [-width / 2, 0.0, height / 2],
        [-width / 2, length / 2, height / 2],
        [0.0, -length / 2, height / 2],
        [0.0, 0.0, height / 2],
        [0.0, length / 2, height / 2],
        [width / 2, -length / 2, height / 2],
        [width / 2, 0.0, height / 2],
        [width / 2, length / 2, height / 2],
    ]

    weights = [1.0] * 27

    vol.ctrlpts_size_u = cp_size_u
    vol.ctrlpts_size_v = cp_size_v
    vol.ctrlpts_size_w = cp_size_w

    t_ctrlptsw = compat.combine_ctrlpts_weights(ctrlpts, weights)

    vol.set_ctrlpts(t_ctrlptsw, cp_size_u, cp_size_v, cp_size_w)

    vol.knotvector_u = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    vol.knotvector_v = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    vol.knotvector_w = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

    do_uniform_knot_refinement_volume(vol, n_ele_u, n_ele_v, n_ele_w)

    return vol


def do_uniform_knot_refinement_surface(surf, n_ele_u, n_ele_v):
    """
    This function does an uniform knot refinement in the u- and v- direction.

    Args
    ----
    surf: geomdl object
        geomdl object that contains the surface information
    n_ele_u: int
        number of elements in the parametric u-direction
    n_ele_v: int
        number of elements in the parametric v-direction

    Return
    ----
    surf: geomdl object
    """

    size_of_knotvector_u = 1 / n_ele_u
    size_of_knotvector_v = 1 / n_ele_v

    for i in range(1, n_ele_u):
        operations.insert_knot(surf, [size_of_knotvector_u * i, None], [1, 0])
    for j in range(1, n_ele_v):
        operations.insert_knot(surf, [None, size_of_knotvector_v * j], [0, 1])


def do_uniform_knot_refinement_volume(vol, n_ele_u, n_ele_v, n_ele_w):
    """
    This function does an uniform knot refinement in the u-, v- and w- direction

    Args
    ----
    vol: geomdl object
        geomdl object that contains the volume information
    n_ele_u: int
        number of elements in the parametric u-direction
    n_ele_v: int
        number of elements in the parametric v-direction
    n_ele_w: int
        number of elements in the parametric w-direction

    Return
    ----
    vol: geomdl object
    """

    size_of_knotvector_u = 1 / n_ele_u
    size_of_knotvector_v = 1 / n_ele_v
    size_of_knotvector_w = 1 / n_ele_w

    for i in range(1, n_ele_u):
        operations.insert_knot(vol, [size_of_knotvector_u * i, None, None], [1, 0, 0])
    for j in range(1, n_ele_v):
        operations.insert_knot(vol, [None, size_of_knotvector_v * j, None], [0, 1, 0])
    for k in range(1, n_ele_w):
        operations.insert_knot(vol, [None, None, size_of_knotvector_w * k], [0, 0, 1])
