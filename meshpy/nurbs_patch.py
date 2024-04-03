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
This module implements NURBS patches for the mesh.
"""

# Python modules
import numpy as np

# Meshpy modules
from .conf import mpy
from .element import Element

from .material import MaterialString, MaterialStVenantKirchhoff


class NURBSPatch(Element):
    """
    A base class for a NURBS patch
    """

    # A list of valid material types for this element
    valid_material = [MaterialString, MaterialStVenantKirchhoff]

    def __init__(
        self,
        knot_vectors,
        polynomial_orders,
        material=None,
        nodes=None,
        element_description=None,
    ):
        super().__init__(nodes=nodes, material=material)

        # Knot vectors
        self.knot_vectors = knot_vectors

        # Polynomial degrees
        self.polynomial_orders = polynomial_orders

        # Set numbers for elements
        self.n_nurbs_patch = None

        # Set the element description
        self.element_description = element_description

    def add_element_specific_section(self, sections):
        """Return additional information of the NURBS patch"""

        from .inputfile import InputSectionMultiKey

        knotvectors_section = "STRUCTURE KNOTVECTORS"

        if not knotvectors_section in sections.keys():
            sections[knotvectors_section] = InputSectionMultiKey(knotvectors_section)

        section = sections[knotvectors_section]
        section.add("BEGIN NURBSPATCH")
        section.add("ID {}".format(self.n_nurbs_patch))

        for dir_manifold in range(len(self.knot_vectors)):
            num_knotvectors = len(self.knot_vectors[dir_manifold])
            section.add("NUMKNOTS {}".format(num_knotvectors))
            section.add("DEGREE {}".format(self.polynomial_orders[dir_manifold]))

            # Check the type of knot vector, in case that the multiplicity of the first and last
            # knot vectors is not p + 1, then it is a closed (periodic) knot vector, otherwise it
            # is an open (interpolated) knot vector.
            knotvector_type = "Interpolated"

            for i in range(self.polynomial_orders[dir_manifold] - 1):
                if (
                    abs(
                        self.knot_vectors[dir_manifold][i]
                        - self.knot_vectors[dir_manifold][i + 1]
                    )
                    > mpy.eps_knot_vector
                ) or (
                    abs(
                        self.knot_vectors[dir_manifold][num_knotvectors - 2 - i]
                        - self.knot_vectors[dir_manifold][num_knotvectors - 1 - i]
                    )
                    > mpy.eps_knot_vector
                ):
                    knotvector_type = "Periodic"
                    break

            section.add("TYPE {}".format(knotvector_type))

            for knot_vector_val in self.knot_vectors[dir_manifold]:
                section.add("{}".format(knot_vector_val))

        section.add("END NURBSPATCH")

    def get_number_elements(self):
        """Get the number of elements in this patch by checking
        the amount of nonzero knot spans in the knot vector"""

        num_elements_dir = np.zeros(len(self.knot_vectors), dtype=int)

        for dir in range(len(self.knot_vectors)):
            for i in range(len(self.knot_vectors[dir]) - 1):
                if (
                    abs(self.knot_vectors[dir][i] - self.knot_vectors[dir][i + 1])
                    > mpy.eps_knot_vector
                ):
                    num_elements_dir[dir] += 1

        total_num_elements = np.prod(num_elements_dir)

        return total_num_elements

    def _check_material(self):
        """Check if the linked material is valid for this type of NURBS solid element"""
        for material_type in self.valid_material:
            if type(self.material) is material_type:
                break
        else:
            raise TypeError(
                "NURBS solid of type {} can not have a material of type {}!".format(
                    type(self), type(self.material)
                )
            )


class NURBSSurface(NURBSPatch):
    """
    A patch of a NURBS surface
    """

    def _get_dat(self):
        """Return the lines with elements for the input file"""

        # Check the material
        self._check_material()

        # Calculate how many control points are on the u direction
        ctrlpts_size_u = len(self.knot_vectors[0]) - self.polynomial_orders[0] - 1

        def get_ids_ctrlpts_surface(knot_span_u, knot_span_v):
            """For an interpolated patch, calculate control points involved in evaluation of the
            surface point at the knot span (knot_span_u, knot_span_v)"""

            id_u = knot_span_u - self.polynomial_orders[0]
            id_v = knot_span_v - self.polynomial_orders[1]

            element_ctrlpts_ids = []
            for j in range(self.polynomial_orders[1] + 1):
                for i in range(self.polynomial_orders[0] + 1):
                    # Calculating the global index of the control point, based on the book
                    # "Isogeometric Analysis: toward Integration of CAD and FEA" of J. Austin Cottrell, p. 314.
                    index_global = ctrlpts_size_u * (id_v + j) + id_u + i
                    element_ctrlpts_ids.append(index_global)

            return element_ctrlpts_ids

        patch_elements = []

        # Adding an increment j to self.global to obtain the ID of an element in the patch
        j = 0

        # Loop over the knot spans to obtain the elements inside the patch
        for knot_span_v in range(
            self.polynomial_orders[1],
            len(self.knot_vectors[1]) - self.polynomial_orders[1] - 1,
        ):
            for knot_span_u in range(
                self.polynomial_orders[0],
                len(self.knot_vectors[0]) - self.polynomial_orders[0] - 1,
            ):
                element_cps_ids = get_ids_ctrlpts_surface(knot_span_u, knot_span_v)

                string_cps = ""
                for i in element_cps_ids:
                    cp = self.nodes[i]
                    string_cps += "{} ".format(cp.n_global)

                patch_elements.append(
                    "{} WALLNURBS NURBS{} {} MAT {} {}".format(
                        self.n_global + j,
                        (self.polynomial_orders[0] + 1)
                        * (self.polynomial_orders[1] + 1),
                        string_cps,
                        self.material.n_global,
                        self.element_description,
                    )
                )
                j += 1

        return patch_elements


class NURBSVolume(NURBSPatch):
    """
    A patch of a NURBS volume
    """

    def _get_dat(self):
        """Return the lines with elements for the input file"""

        # Check the material
        self._check_material()

        # Calculate how many control points are on the u and v directions
        ctrlpts_size_u = len(self.knot_vectors[0]) - self.polynomial_orders[0] - 1
        ctrlpts_size_v = len(self.knot_vectors[1]) - self.polynomial_orders[1] - 1

        def get_ids_ctrlpts_volume(knot_span_u, knot_span_v, knot_span_w):
            """For an interpolated patch, calculate control points involved in evaluation of the
            surface point at the knot span (knot_span_u, knot_span_v, knot_span_w)"""

            id_u = knot_span_u - self.polynomial_orders[0]
            id_v = knot_span_v - self.polynomial_orders[1]
            id_w = knot_span_w - self.polynomial_orders[2]

            element_ctrlpts_ids = []

            for k in range(self.polynomial_orders[2] + 1):
                for j in range(self.polynomial_orders[1] + 1):
                    for i in range(self.polynomial_orders[0] + 1):
                        # Calculating the global index of the control point, based on the paper
                        # "Isogeometric analysis: an overview and computer implementation aspects" of Vinh-Phu Nguyen, Mathematics and Computers in Simulation, Jun-2015.
                        index_global = (
                            ctrlpts_size_u * ctrlpts_size_v * (id_w + k)
                            + ctrlpts_size_u * (id_v + j)
                            + id_u
                            + i
                        )
                        element_ctrlpts_ids.append(index_global)

            return element_ctrlpts_ids

        patch_elements = []

        # Adding an increment to self.global to obtain the ID of an element in the patch
        increment_ele = 0

        # Loop over the knot spans to obtain the elements inside the patch
        for knot_span_w in range(
            self.polynomial_orders[2],
            len(self.knot_vectors[2]) - self.polynomial_orders[2] - 1,
        ):
            for knot_span_v in range(
                self.polynomial_orders[1],
                len(self.knot_vectors[1]) - self.polynomial_orders[1] - 1,
            ):
                for knot_span_u in range(
                    self.polynomial_orders[0],
                    len(self.knot_vectors[0]) - self.polynomial_orders[0] - 1,
                ):
                    element_cps_ids = get_ids_ctrlpts_volume(
                        knot_span_u, knot_span_v, knot_span_w
                    )

                    string_cps = ""
                    for i in element_cps_ids:
                        cp = self.nodes[i]
                        string_cps += "{} ".format(cp.n_global)

                    num_cp_in_element = (
                        (self.polynomial_orders[0] + 1)
                        * (self.polynomial_orders[1] + 1)
                        * (self.polynomial_orders[2] + 1)
                    )

                    patch_elements.append(
                        "{} SONURBS{} NURBS{} {} MAT {} {}".format(
                            self.n_global + increment_ele,
                            num_cp_in_element,
                            num_cp_in_element,
                            string_cps,
                            self.material.n_global,
                            self.element_description,
                        )
                    )
                    increment_ele += 1

        return patch_elements
