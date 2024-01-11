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
This function creates the code for the definition of the geometry in unit
tests.
"""

# Meshpy modules.
from .. import (
    Beam,
    VolumeElement,
    VolumeHEX8,
    VolumeHEX20,
    VolumeHEX27,
    VolumeTET4,
    VolumeTET10,
)


def get_unit_test_code(mesh, function_name):
    """
    Create the C++ code for the definition of the positional DOF vector in the
    unit tests. This currently only works, if there is a single type of solid
    element in the mesh.
    """

    # Empty lists with code lines.
    list_variables_definition = []
    list_function_definition = []
    list_beam = []
    list_beam_elements = []
    list_solid = []
    list_solid_elements = []

    # Counter for created elements.
    element_counter = 0

    # Loop over beam elements.
    for i, element in enumerate(
        [element for element in mesh.elements if isinstance(element, Beam)]
    ):
        if i == 0:
            # At the first element, initialize vectors.
            list_variables_definition.append(
                "std::vector<LINALG::TMatrix<double, 12, 1>> q_line_elements;"
            )
            list_function_definition.append(
                "std::vector<LINALG::TMatrix<double, 12, 1>>& q_line_elements"
            )
            list_variables_definition.append(
                "std::vector<LINALG::TMatrix<double, 9, 1>> q_rot_line_elements;"
            )
            list_function_definition.append(
                "std::vector<LINALG::TMatrix<double, 9, 1>>& q_rot_line_elements"
            )
            list_beam_elements.append("const int dummy_node_ids[2] = {0, 1};")

        # Add beam element.
        list_beam_elements.append(
            "line_elements.push_back(Teuchos::rcp(new DRT::ELEMENTS::Beam3r({}, 0)));".format(
                element_counter
            )
        )
        list_beam_elements.append(
            "line_elements.back()->SetNodeIds(2, dummy_node_ids);"
        )

        # Add position and tangents of the centerline nodes.
        list_beam.append("q_line_elements.push_back(LINALG::TMatrix<double, 12, 1>());")
        counter_dof = 0
        for node in [element.nodes[0], element.nodes[2]]:
            # Set the position dofs for the current node.
            for coord in node.coordinates:
                list_beam.append(
                    ("q_line_elements.back()({}) = {};").format(counter_dof, coord)
                )
                counter_dof += 1

            # Set the rotational dofs for the current node.
            t = node.rotation * [1, 0, 0]
            for coord in t:
                list_beam.append(
                    ("q_line_elements.back()({}) = {};").format(counter_dof, coord)
                )
                counter_dof += 1

        # Add the rotation vectors for the calculation of the reference length.
        list_beam.append(
            "q_rot_line_elements.push_back(LINALG::TMatrix<double, 9, 1>());"
        )
        counter_dof = 0
        for node in [element.nodes[0], element.nodes[2], element.nodes[1]]:
            # Set the rotational dofs for the current node.
            for coord in node.rotation.get_rotation_vector():
                list_beam.append(
                    "q_rot_line_elements.back()({}) = {};".format(counter_dof, coord)
                )
                counter_dof += 1

        element_counter += 1

    # Loop over solid elements.
    for i, element in enumerate(
        [element for element in mesh.elements if isinstance(element, VolumeElement)]
    ):
        # Number of nodes for this element.
        n_nodes = len(element.nodes)

        if i == 0:
            # At the first element, initialize vectors.
            list_variables_definition.append(
                "std::vector<LINALG::TMatrix<double, {}, 1>> q_volume_elements;".format(
                    3 * n_nodes
                )
            )
            list_function_definition.append(
                "std::vector<LINALG::TMatrix<double, {}, 1>>& q_volume_elements".format(
                    3 * n_nodes
                )
            )
            list_variables_definition.append(
                "std::vector<Teuchos::RCP<GEOMETRYPAIR::GeometryPairLineToVolumeSegmentation<double, 2, 2, {}, 1>>> geometry_pairs;".format(
                    n_nodes
                )
            )
            list_function_definition.append(
                "std::vector<Teuchos::RCP<GEOMETRYPAIR::GeometryPairLineToVolumeSegmentation<double, 2, 2, {}, 1>>>& geometry_pairs".format(
                    n_nodes
                )
            )

        # Add the element.
        if isinstance(element, VolumeHEX8):
            list_solid_elements.append(
                "volume_elements.push_back(Teuchos::rcp(new DRT::ELEMENTS::So_hex8({}, 0)));".format(
                    element_counter
                )
            )
        elif isinstance(element, VolumeHEX20):
            list_solid_elements.append(
                "volume_elements.push_back(Teuchos::rcp(new DRT::ELEMENTS::So_hex20({}, 0)));".format(
                    element_counter
                )
            )
        elif isinstance(element, VolumeHEX27):
            list_solid_elements.append(
                "volume_elements.push_back(Teuchos::rcp(new DRT::ELEMENTS::So_hex27({}, 0)));".format(
                    element_counter
                )
            )
        elif isinstance(element, VolumeTET4):
            list_solid_elements.append(
                "volume_elements.push_back(Teuchos::rcp(new DRT::ELEMENTS::So_tet4({}, 0)));".format(
                    element_counter
                )
            )
        elif isinstance(element, VolumeTET10):
            list_solid_elements.append(
                "volume_elements.push_back(Teuchos::rcp(new DRT::ELEMENTS::So_tet10({}, 0)));".format(
                    element_counter
                )
            )
        else:
            raise TypeError("Element type not implemented!")

        # Add new vectors for this element.
        list_solid.append(
            "q_volume_elements.push_back(LINALG::TMatrix<double, {}, 1>());".format(
                3 * n_nodes
            )
        )

        # Add position of the nodes.
        counter_dof = 0
        for node in element.nodes:
            # Set the position dofs for the current node.
            for coord in node.coordinates:
                list_solid.append(
                    ("q_volume_elements.back()({}) = {};").format(counter_dof, coord)
                )
                counter_dof += 1

        element_counter += 1

    return "\n".join(
        [
            "// Definition of variables for this test case.",
            "\n".join(list_variables_definition),
            "\n\n\n/**",
            "* \\brief The following code part is generated with meshpy. The function defines element coordinates for unit test examples.",
            "*/",
            "void {}(std::vector<Teuchos::RCP<DRT::Element>>& line_elements, std::vector<Teuchos::RCP<DRT::Element>>& volume_elements, {}){{".format(
                function_name, ", ".join(list_function_definition)
            ),
            "// Create the elements.",
            "\n".join(list_beam_elements),
            "\n".join(list_solid_elements),
            "",
            "// Positional and tangent DOFs of the line(s).",
            "\n".join(list_beam),
            "\n// Positional DOFs of the solid(s).",
            "\n".join(list_solid),
            "}",
        ]
    )
