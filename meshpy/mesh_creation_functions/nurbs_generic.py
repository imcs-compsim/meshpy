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
Generic function used to create NURBS meshes within meshpy.
"""

# Python modules
import numpy as np

# Meshpy modules
from ..conf import mpy
from ..container import GeometryName
from ..geometry_set import GeometrySet
from ..node import ControlPoint
from ..nurbs_patch import NURBSSurface, NURBSVolume


def add_geomdl_nurbs_to_mesh(
    mesh,
    geomdl_obj,
    *,
    material=None,
    element_description=None,
):
    """
    Generic NURBS mesh creation function.

    Args
    ----
    mesh: Mesh
        Mesh that the created NURBS geometry should be added to.
    geomdl_obj: Geomdl object
        NURBS geometry created using Geomdl.
    material: Material
        Material for this geometry.
    element_description:
        Information that will be written after the information of
        the elements.

    Return
    ----
    return_set: GeometryName
        Set with the control points that form the topology of the mesh.

        For a surface, the following information is stored:
            Vertices: 'vertex_u_min_v_min', 'vertex_u_max_v_min', 'vertex_u_min_v_max', 'vertex_u_max_v_max'
            Edges: 'line_v_min', 'line_u_max', 'line_v_max', 'line_u_min'
            Surface: 'surf'

        For a volume, the following information is stored:
            Vertices: 'vertex_u_min_v_min_w_min', 'vertex_u_max_v_min_w_min', 'vertex_u_min_v_max_w_min', 'vertex_u_max_v_max_w_min',
                      'vertex_u_min_v_min_w_max', 'vertex_u_max_v_min_w_max', 'vertex_u_min_v_max_w_max', 'vertex_u_max_v_max_w_max'
            Edges: 'line_v_min_w_min', 'line_u_max_w_min', 'line_v_max_w_min', 'line_u_min_w_min',
                   'line_u_min_v_min', 'line_u_max_v_min', 'line_u_min_v_max', 'line_u_max_v_max'
                   'line_v_min_w_max', 'line_u_max_w_max', 'line_v_max_w_max', 'line_u_min_w_max'
            Surfaces: 'surf_w_min', 'surf_w_max', 'surf_v_min', 'surf_v_max', 'surf_v_max', 'surf_u_min'
            Volume: 'vol'
    """

    # Make sure the material is in the mesh
    mesh.add_material(material)

    # Fill control points
    control_points = []
    nurbs_dimension = len(geomdl_obj.knotvector)
    if nurbs_dimension == 2:
        control_points = create_control_points_surface(geomdl_obj)
    elif nurbs_dimension == 3:
        control_points = create_control_points_volume(geomdl_obj)
    else:
        raise NotImplementedError(
            "Error, not implemented for NURBS with dimension {}!".format(
                nurbs_dimension
            )
        )

    # Fill element
    manifold_dim = len(geomdl_obj.knotvector)

    if manifold_dim == 2:
        nurbs_object = NURBSSurface
    elif manifold_dim == 3:
        nurbs_object = NURBSVolume
    else:
        raise NotImplementedError(
            "Error, not implemented for a NURBS {}!".format(type(geomdl_obj))
        )

    element = nurbs_object(
        geomdl_obj.knotvector,
        geomdl_obj.degree,
        nodes=control_points,
        material=material,
        element_description=element_description,
    )

    # Add element and control points to the mesh
    mesh.elements.append(element)
    mesh.nodes.extend(control_points)

    # Create geometry sets that will be returned
    return_set = create_geometry_sets(element)

    return return_set


def create_control_points_surface(geomdl_obj):
    """
    Creates a list with the ControlPoint objects of a surface created with geomdl
    """
    control_points = []
    for dir_v in range(geomdl_obj.ctrlpts_size_v):
        for dir_u in range(geomdl_obj.ctrlpts_size_u):
            weight = geomdl_obj.ctrlpts2d[dir_u][dir_v][3]

            # As the control points are scaled with their weight, divide them to get
            # their coordinates
            coord = [
                geomdl_obj.ctrlpts2d[dir_u][dir_v][0] / weight,
                geomdl_obj.ctrlpts2d[dir_u][dir_v][1] / weight,
                geomdl_obj.ctrlpts2d[dir_u][dir_v][2] / weight,
            ]

            control_points.append(ControlPoint(coord, weight))

    return control_points


def create_control_points_volume(geomdl_obj):
    """
    Creates a list with the ControlPoint objects of a volume created with geomdl
    """
    control_points = []
    for dir_w in range(geomdl_obj.ctrlpts_size_w):
        for dir_v in range(geomdl_obj.ctrlpts_size_v):
            for dir_u in range(geomdl_obj.ctrlpts_size_u):
                # Obtain the id of the control point
                cp_id = (
                    dir_v
                    + geomdl_obj.ctrlpts_size_v * dir_u
                    + geomdl_obj.ctrlpts_size_u * geomdl_obj.ctrlpts_size_v * dir_w
                )

                weight = geomdl_obj.ctrlptsw[cp_id][3]

                # As the control points are scaled with their weight, divide them to get
                # their coordinates
                coord = [
                    geomdl_obj.ctrlptsw[cp_id][0] / weight,
                    geomdl_obj.ctrlptsw[cp_id][1] / weight,
                    geomdl_obj.ctrlptsw[cp_id][2] / weight,
                ]

                control_points.append(ControlPoint(coord, weight))

    return control_points


def create_geometry_sets(element):
    """
    Function that returns a GeometryName object. For more information of the
    return item, look into add_geomdl_nurbs_to_mesh.
    """

    def get_num_cps_uvw(knot_vectors):
        """Obtain the number of control points on each parametric direction of a patch"""
        num_cps_uvw = np.zeros(len(knot_vectors), dtype=int)

        for direction in range(len(knot_vectors)):
            knotvector_size_dir = len(element.knot_vectors[direction])
            p_dir = element.polynomial_orders[direction]
            cp_size_dir = knotvector_size_dir - p_dir - 1

            num_cps_uvw[direction] = cp_size_dir

        return num_cps_uvw

    def get_patch_vertices(return_set, num_cps_uvw, nurbs_dimension, element):
        """Get the control points positioned over the vertices of a patch"""

        if nurbs_dimension == 2:
            # Vertex 1 is positioned on u = 0, v = 0
            return_set["vertex_u_min_v_min"] = GeometrySet(
                mpy.geo.point, nodes=element.nodes[0]
            )

            # Vertex 2 is positioned on u = 1, v = 0
            return_set["vertex_u_max_v_min"] = GeometrySet(
                mpy.geo.point, nodes=element.nodes[num_cps_uvw[0] - 1]
            )

            # Vertex 3 is positioned on u = 0, v = 1
            return_set["vertex_u_min_v_max"] = GeometrySet(
                mpy.geo.point,
                nodes=element.nodes[num_cps_uvw[0] * (num_cps_uvw[1] - 1)],
            )

            # Vertex 4 is positioned on u = 1, v = 1
            return_set["vertex_u_max_v_max"] = GeometrySet(
                mpy.geo.point, nodes=element.nodes[num_cps_uvw[0] * num_cps_uvw[1] - 1]
            )

        elif nurbs_dimension == 3:
            # Vertex 1 is positioned on u = 0, v = 0, w =
            return_set["vertex_u_min_v_min_w_min"] = GeometrySet(
                mpy.geo.point, nodes=element.nodes[0]
            )

            # Vertex 2 is positioned on u = 1, v = 0, w = 0
            return_set["vertex_u_max_v_min_w_min"] = GeometrySet(
                mpy.geo.point, nodes=element.nodes[num_cps_uvw[0] - 1]
            )

            # Vertex 3 is positioned on u = 0, v = 1, w = 0
            return_set["vertex_u_min_v_max_w_min"] = GeometrySet(
                mpy.geo.point,
                nodes=element.nodes[num_cps_uvw[0] * (num_cps_uvw[1] - 1)],
            )

            # Vertex 4 is positioned on u = 1, v = 1, w = 0
            return_set["vertex_u_max_v_max_w_min"] = GeometrySet(
                mpy.geo.point, nodes=element.nodes[num_cps_uvw[0] * num_cps_uvw[1] - 1]
            )

            # Vertex 5 is positioned on u = 0, v = 0, w = 1
            return_set["vertex_u_min_v_min_w_max"] = GeometrySet(
                mpy.geo.point,
                nodes=element.nodes[
                    num_cps_uvw[0] * num_cps_uvw[1] * (num_cps_uvw[2] - 1)
                ],
            )

            # Vertex 6 is positioned on u = 1, v = 0, w = 1
            return_set["vertex_u_max_v_min_w_max"] = GeometrySet(
                mpy.geo.point,
                nodes=element.nodes[
                    num_cps_uvw[0] * num_cps_uvw[1] * (num_cps_uvw[2] - 1)
                    + (num_cps_uvw[0] - 1)
                ],
            )

            # Vertex 7 is positioned on u = 0, v = 1, w = 1
            return_set["vertex_u_min_v_max_w_max"] = GeometrySet(
                mpy.geo.point,
                nodes=element.nodes[
                    num_cps_uvw[0] * num_cps_uvw[1] * (num_cps_uvw[2] - 1)
                    + num_cps_uvw[0] * (num_cps_uvw[1] - 1)
                ],
            )

            # Vertex 8 is positioned on u = 1, v = 1, w = 1
            return_set["vertex_u_max_v_max_w_max"] = GeometrySet(
                mpy.geo.point,
                nodes=element.nodes[
                    num_cps_uvw[0] * num_cps_uvw[1] * num_cps_uvw[2] - 1
                ],
            )

        else:
            raise NotImplementedError(
                "Error, not implemented for NURBS with dimension {}!".format(
                    nurbs_dimension
                )
            )

    def get_patch_lines(return_set, num_cps_uvw, nurbs_dimension, element):
        """Get the control points positioned over the lines of a patch"""

        control_points_line_1 = []
        control_points_line_2 = []
        control_points_line_3 = []
        control_points_line_4 = []

        if nurbs_dimension == 2:
            # Fill line 1 and line 3 with their control points
            for i in range(num_cps_uvw[0]):
                # Line 1 has the control points on u = [0,1], v = 0
                cpgid_l1 = num_cps_uvw[0] * 0 + i
                control_points_line_1.append(element.nodes[cpgid_l1])

                # Line 3 has the control points on u = [0,1], v = 1
                cpgid_l3 = num_cps_uvw[0] * (num_cps_uvw[1] - 1) + i
                control_points_line_3.append(element.nodes[cpgid_l3])

            # Fill line 2 and line 4 with their control points
            for j in range(num_cps_uvw[1]):
                # Line 2 has the control points on u = 1, v = [0,1]
                cpgid_l2 = num_cps_uvw[0] * j + (num_cps_uvw[0] - 1)
                control_points_line_2.append(element.nodes[cpgid_l2])

                # Line 4 has the control points on u = 0, v = [0,1]
                cpgid_l4 = num_cps_uvw[0] * j + 0
                control_points_line_4.append(element.nodes[cpgid_l4])

            # Create geometric sets for lines
            return_set["line_v_min"] = GeometrySet(
                mpy.geo.line, nodes=control_points_line_1
            )
            return_set["line_u_max"] = GeometrySet(
                mpy.geo.line, nodes=control_points_line_2
            )
            return_set["line_v_max"] = GeometrySet(
                mpy.geo.line, nodes=control_points_line_3
            )
            return_set["line_u_min"] = GeometrySet(
                mpy.geo.line, nodes=control_points_line_4
            )

        elif nurbs_dimension == 3:
            # Define the rest of the lines to define a volume
            control_points_line_5 = []
            control_points_line_6 = []
            control_points_line_7 = []
            control_points_line_8 = []
            control_points_line_9 = []
            control_points_line_10 = []
            control_points_line_11 = []
            control_points_line_12 = []

            # Fill line 1, 3, 9 and 11 with their control points
            for i in range(num_cps_uvw[0]):
                # Line 1 has the control points on u = [0,1], v = 0, w = 0
                cpgid_l1 = num_cps_uvw[0] * 0 + i
                control_points_line_1.append(element.nodes[cpgid_l1])

                # Line 3 has the control points on u = [0,1], v = 1, w = 0
                cpgid_l3 = num_cps_uvw[0] * (num_cps_uvw[1] - 1) + i
                control_points_line_3.append(element.nodes[cpgid_l3])

                # Line 9 has the control points on u = [0,1], v = 0, w = 1
                cpgid_l9 = num_cps_uvw[0] * num_cps_uvw[1] * (num_cps_uvw[2] - 1) + i
                control_points_line_9.append(element.nodes[cpgid_l9])

                # Line 11 has the control points on u = [0,1], v = 1, w = 1
                cpgid_l11 = (
                    num_cps_uvw[0] * num_cps_uvw[1] * (num_cps_uvw[2] - 1)
                    + num_cps_uvw[0] * (num_cps_uvw[1] - 1)
                    + i
                )
                control_points_line_11.append(element.nodes[cpgid_l11])

            # Fill line 2, 4, 10 and 12 with their control points
            for j in range(num_cps_uvw[1]):
                # Line 2 has the control points on u = 1, v = [0,1] , w = 0
                cpgid_l2 = num_cps_uvw[0] * j + (num_cps_uvw[0] - 1)
                control_points_line_2.append(element.nodes[cpgid_l2])

                # Line 4 has the control points on u = 0, v = [0,1] , w = 0
                cpgid_l4 = num_cps_uvw[0] * j + 0
                control_points_line_4.append(element.nodes[cpgid_l4])

                # Line 10 has the control points on u = 1, v = [0,1] , w = 1
                cpgid_l10 = (
                    num_cps_uvw[0] * num_cps_uvw[1] * (num_cps_uvw[2] - 1)
                    + num_cps_uvw[0] * j
                    + (num_cps_uvw[0] - 1)
                )
                control_points_line_10.append(element.nodes[cpgid_l10])

                # Line 12 has the control points on u = 0, v = [0,1] , w = 1
                cpgid_l12 = (
                    num_cps_uvw[0] * num_cps_uvw[1] * (num_cps_uvw[2] - 1)
                    + num_cps_uvw[0] * j
                )
                control_points_line_12.append(element.nodes[cpgid_l12])

            # Fill line 5, 6, 7 and 8 with their control points
            for k in range(num_cps_uvw[2]):
                # Line 5 has the control points on u = 0, v = 0 , w = [0,1]
                cpgid_l5 = num_cps_uvw[0] * num_cps_uvw[1] * k
                control_points_line_5.append(element.nodes[cpgid_l5])

                # Line 6 has the control points on u = 1, v = 0 , w = [0,1]
                cpgid_l6 = num_cps_uvw[0] * num_cps_uvw[1] * k + num_cps_uvw[0] - 1
                control_points_line_6.append(element.nodes[cpgid_l6])

                # Line 7 has the control points on u = 0, v = 1 , w = [0,1]
                cpgid_l7 = num_cps_uvw[0] * num_cps_uvw[1] * k + num_cps_uvw[0] * (
                    num_cps_uvw[1] - 1
                )
                control_points_line_7.append(element.nodes[cpgid_l7])

                # Line 8 has the control points on u = 1, v = 1 , w = [0,1]
                cpgid_l8 = (
                    num_cps_uvw[0] * num_cps_uvw[1] * k
                    + num_cps_uvw[0] * num_cps_uvw[1]
                    - 1
                )
                control_points_line_8.append(element.nodes[cpgid_l8])

            # Create geometric sets for lines
            return_set["line_v_min_w_min"] = GeometrySet(
                mpy.geo.line, nodes=control_points_line_1
            )
            return_set["line_u_max_w_min"] = GeometrySet(
                mpy.geo.line, nodes=control_points_line_2
            )
            return_set["line_v_max_w_min"] = GeometrySet(
                mpy.geo.line, nodes=control_points_line_3
            )
            return_set["line_u_min_w_min"] = GeometrySet(
                mpy.geo.line, nodes=control_points_line_4
            )
            return_set["line_u_min_v_min"] = GeometrySet(
                mpy.geo.line, nodes=control_points_line_5
            )
            return_set["line_u_max_v_min"] = GeometrySet(
                mpy.geo.line, nodes=control_points_line_6
            )
            return_set["line_u_min_v_max"] = GeometrySet(
                mpy.geo.line, nodes=control_points_line_7
            )
            return_set["line_u_max_v_max"] = GeometrySet(
                mpy.geo.line, nodes=control_points_line_8
            )
            return_set["line_v_min_w_max"] = GeometrySet(
                mpy.geo.line, nodes=control_points_line_9
            )
            return_set["line_u_max_w_max"] = GeometrySet(
                mpy.geo.line, nodes=control_points_line_10
            )
            return_set["line_v_max_w_max"] = GeometrySet(
                mpy.geo.line, nodes=control_points_line_11
            )
            return_set["line_u_min_w_max"] = GeometrySet(
                mpy.geo.line, nodes=control_points_line_12
            )

        else:
            raise NotImplementedError(
                "Error, not implemented for NURBS with dimension {}!".format(
                    nurbs_dimension
                )
            )

    def get_patch_surfaces(return_set, num_cps_uvw, nurbs_dimension, element):
        """Get the control points positioned over the surfaces of a patch"""

        control_points_surface_1 = []

        if nurbs_dimension == 2:
            # As there is only one surface, it collects all the control points
            control_points_surface_1.extend(
                element.nodes[: (num_cps_uvw[0] * num_cps_uvw[1])]
            )

            # Create geometric sets for surfaces
            return_set["surf"] = GeometrySet(
                mpy.geo.surface, nodes=control_points_surface_1
            )

        elif nurbs_dimension == 3:
            control_points_surface_2 = []
            control_points_surface_3 = []
            control_points_surface_4 = []
            control_points_surface_5 = []
            control_points_surface_6 = []

            # Surface defined on w = 0
            control_points_surface_1.extend(
                element.nodes[: (num_cps_uvw[0] * num_cps_uvw[1])]
            )

            # Surface defined on w = 1
            control_points_surface_2.extend(
                element.nodes[
                    (num_cps_uvw[0] * num_cps_uvw[1] * (num_cps_uvw[2] - 1)) : (
                        num_cps_uvw[0] * num_cps_uvw[1] * num_cps_uvw[2]
                    )
                ]
            )

            for layers_w_dir in range(num_cps_uvw[2]):
                # Calculate the number of control points on the w-plane
                num_cps_plane_w = num_cps_uvw[0] * num_cps_uvw[1]

                for layers_u_dir in range(num_cps_uvw[0]):
                    # Surface defined on v = 0
                    cpgid_l1 = num_cps_uvw[0] * 0 + layers_u_dir
                    control_points_surface_3.append(
                        element.nodes[cpgid_l1 + num_cps_plane_w * layers_w_dir]
                    )

                    # Surface defined on v = 1
                    cpgid_l3 = num_cps_uvw[0] * (num_cps_uvw[1] - 1) + layers_u_dir
                    control_points_surface_5.append(
                        element.nodes[cpgid_l3 + num_cps_plane_w * layers_w_dir]
                    )

                for layers_v_dir in range(num_cps_uvw[1]):
                    # Surface defined on u = 1
                    cpgid_l2 = num_cps_uvw[0] * layers_v_dir + (num_cps_uvw[0] - 1)
                    control_points_surface_4.append(
                        element.nodes[cpgid_l2 + num_cps_plane_w * layers_w_dir]
                    )

                    # Surface defined on u = 0
                    cpgid_l4 = num_cps_uvw[0] * layers_v_dir + 0
                    control_points_surface_6.append(
                        element.nodes[cpgid_l4 + num_cps_plane_w * layers_w_dir]
                    )

            # Create geometric sets for surfaces
            return_set["surf_w_min"] = GeometrySet(
                mpy.geo.surface, nodes=control_points_surface_1
            )
            return_set["surf_w_max"] = GeometrySet(
                mpy.geo.surface, nodes=control_points_surface_2
            )
            return_set["surf_v_min"] = GeometrySet(
                mpy.geo.surface, nodes=control_points_surface_3
            )
            return_set["surf_u_max"] = GeometrySet(
                mpy.geo.surface, nodes=control_points_surface_4
            )
            return_set["surf_v_max"] = GeometrySet(
                mpy.geo.surface, nodes=control_points_surface_5
            )
            return_set["surf_u_min"] = GeometrySet(
                mpy.geo.surface, nodes=control_points_surface_6
            )

        else:
            raise NotImplementedError(
                "Error, not implemented for NURBS with dimension {}!".format(
                    nurbs_dimension
                )
            )

    def get_patch_volume(return_set, num_cps_uvw, nurbs_dimension, element):
        """Get the control points positioned in the volume of a patch"""

        control_points_volume_1 = []

        if nurbs_dimension == 2:
            # As this is a surface, it's not necessary to get a volume GeometrySet
            pass

        elif nurbs_dimension == 3:
            # As there is only one volume, it collects all the control points
            control_points_volume_1.extend(
                element.nodes[: (num_cps_uvw[0] * num_cps_uvw[1] * num_cps_uvw[2])]
            )

            # Create geometric sets for surfaces
            return_set["vol"] = GeometrySet(
                mpy.geo.volume, nodes=control_points_volume_1
            )

        else:
            raise NotImplementedError(
                "Error, not implemented for NURBS with dimension {}".format(
                    nurbs_dimension
                )
            )

    # Create return set
    return_set = GeometryName()

    # Get the number of control points on each parametric direction that define the patch
    num_cps_uvw = get_num_cps_uvw(element.knot_vectors)

    # Get the NURBS dimension
    nurbs_dimension = len(element.knot_vectors)

    # Obtain the vertices of the patch
    get_patch_vertices(return_set, num_cps_uvw, nurbs_dimension, element)

    # Obtain the lines of the patch
    get_patch_lines(return_set, num_cps_uvw, nurbs_dimension, element)

    # Obtain the surfaces of the patch
    get_patch_surfaces(return_set, num_cps_uvw, nurbs_dimension, element)

    # Obtain the volume of the patch
    get_patch_volume(return_set, num_cps_uvw, nurbs_dimension, element)

    return return_set
