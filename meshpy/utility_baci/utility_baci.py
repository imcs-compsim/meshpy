# -*- coding: utf-8 -*-
"""
This function creates the code for the definition of the geometry in unit
tests.
"""

# Meshpy modules.
from .. import Beam, SolidElement, SolidHEX8, SolidHEX20, SolidHEX27, \
    SolidTET4, SolidTET10


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
    for i, element in enumerate([element for element in mesh.elements if
            isinstance(element, Beam)]):

        if i == 0:
            # At the first element, initialize vectors.
            list_variables_definition.append('std::vector<LINALG::TMatrix<double, 12, 1>> q_line_elements;')
            list_function_definition.append('std::vector<LINALG::TMatrix<double, 12, 1>>& q_line_elements')
            list_variables_definition.append('std::vector<LINALG::TMatrix<double, 9, 1>> q_rot_line_elements;')
            list_function_definition.append('std::vector<LINALG::TMatrix<double, 9, 1>>& q_rot_line_elements')
            list_beam_elements.append('const int dummy_node_ids[2] = {0, 1};')

        # Add beam element.
        list_beam_elements.append('line_elements.push_back(Teuchos::rcp(new DRT::ELEMENTS::Beam3r({}, 0)));'.format(element_counter))
        list_beam_elements.append('line_elements.back()->SetNodeIds(2, dummy_node_ids);')

        # Add position and tangents of the centerline nodes.
        list_beam.append('q_line_elements.push_back(LINALG::TMatrix<double, 12, 1>());')
        counter_dof = 0
        for node in [element.nodes[0], element.nodes[2]]:
            # Set the position dofs for the current node.
            for coord in node.coordinates:
                list_beam.append(('q_line_elements.back()({}) = {};').format(
                        counter_dof, coord))
                counter_dof += 1

            # Set the rotational dofs for the current node.
            t = node.rotation * [1, 0, 0]
            for coord in t:
                list_beam.append(('q_line_elements.back()({}) = {};').format(
                        counter_dof, coord))
                counter_dof += 1

        # Add the rotation vectors for the calculation of the reference length.
        list_beam.append('q_rot_line_elements.push_back(LINALG::TMatrix<double, 9, 1>());')
        counter_dof = 0
        for node in [element.nodes[0], element.nodes[2], element.nodes[1]]:
            # Set the rotational dofs for the current node.
            for coord in node.rotation.get_rotation_vector():
                list_beam.append(('q_rot_line_elements.back()({}) = '
                    + '{};').format(
                        counter_dof, coord))
                counter_dof += 1

        element_counter += 1

    # Loop over solid elements.
    for i, element in enumerate([element for element in mesh.elements if
            isinstance(element, SolidElement)]):

        # Number of nodes for this element.
        n_nodes = len(element.nodes)

        if i == 0:
            # At the first element, initialize vectors.
            list_variables_definition.append('std::vector<LINALG::TMatrix<double, {}, 1>> q_volume_elements;'.format(3 * n_nodes))
            list_function_definition.append('std::vector<LINALG::TMatrix<double, {}, 1>>& q_volume_elements'.format(3 * n_nodes))
            list_variables_definition.append('std::vector<Teuchos::RCP<GEOMETRYPAIR::GeometryPairLineToVolumeSegmentation<double, 2, 2, {}, 1>>> geometry_pairs;'.format(n_nodes))
            list_function_definition.append('std::vector<Teuchos::RCP<GEOMETRYPAIR::GeometryPairLineToVolumeSegmentation<double, 2, 2, {}, 1>>>& geometry_pairs'.format(n_nodes))

        # Add the element.
        if isinstance(element, SolidHEX8):
            list_solid_elements.append('volume_elements.push_back(Teuchos::rcp(new DRT::ELEMENTS::So_hex8({}, 0)));'.format(element_counter))
        elif isinstance(element, SolidHEX20):
            list_solid_elements.append('volume_elements.push_back(Teuchos::rcp(new DRT::ELEMENTS::So_hex20({}, 0)));'.format(element_counter))
        elif isinstance(element, SolidHEX27):
            list_solid_elements.append('volume_elements.push_back(Teuchos::rcp(new DRT::ELEMENTS::So_hex27({}, 0)));'.format(element_counter))
        elif isinstance(element, SolidTET4):
            list_solid_elements.append('volume_elements.push_back(Teuchos::rcp(new DRT::ELEMENTS::So_tet4({}, 0)));'.format(element_counter))
        elif isinstance(element, SolidTET10):
            list_solid_elements.append('volume_elements.push_back(Teuchos::rcp(new DRT::ELEMENTS::So_tet10({}, 0)));'.format(element_counter))
        else:
            raise TypeError('Element type not implemented!')

        # Add new vectors for this element.
        list_solid.append('q_volume_elements.push_back(LINALG::TMatrix<double, {}, 1>());'.format(3 * n_nodes))

        # Add position of the nodes.
        counter_dof = 0
        for node in element.nodes:
            # Set the position dofs for the current node.
            for coord in node.coordinates:
                list_solid.append(('q_volume_elements.back()({}) = {};'
                        ).format(counter_dof, coord))
                counter_dof += 1

        element_counter += 1

    return '\n'.join([
        '// Definition of variables for this test case.',
        '\n'.join(list_variables_definition),
        '\n\n\n'
        '/**',
        '* \\brief The following code part is generated with meshpy. The function defines element coordinates for unit test examples.',
        '*/',
        'void {}(std::vector<Teuchos::RCP<DRT::Element>>& line_elements, std::vector<Teuchos::RCP<DRT::Element>>& volume_elements, {}){{'.format(function_name, ', '.join(list_function_definition)),
        '// Create the elements.',
        '\n'.join(list_beam_elements),
        '\n'.join(list_solid_elements),
        '',
        '// Positional and tangent DOFs of the line(s).',
        '\n'.join(list_beam),
        '\n// Positional DOFs of the solid(s).',
        '\n'.join(list_solid),
        '}',
        ])
