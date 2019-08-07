# -*- coding: utf-8 -*-
"""
This script is used to simulate baci input files created with meshpy.
"""

# Python imports.
import os
import unittest
import numpy as np
import subprocess
import shutil
import glob

# Testing imports.
from tests.testing_utility import (baci_release, testing_temp, testing_path,
    testing_input)

# Meshpy imports.
from meshpy import (mpy, Rotation, InputFile, InputSection, MaterialReissner,
    Function, Beam3rHerm2Lin3, BoundaryCondition, Mesh, set_header_static)

# Geometry functions.
from meshpy.mesh_creation_functions.beam_basic_geometry import (
    create_beam_mesh_line)
from meshpy.mesh_creation_functions.beam_honeycomb import (
    create_beam_mesh_honeycomb)


class TestFullBaci(unittest.TestCase):
    """
    Create and run input files in Baci. They are test files and Baci should
    return 0.
    """

    def setUp(self):
        """
        This method is called before each test. Check if BACI was found.
        If not, skip the test.
        """
        if baci_release is None:
            self.skipTest('BACI path was not found!')

    def run_baci_test(self, name, mesh, n_proc=2):
        """
        Run Baci with a input file and check the output. If the test passes,
        the created files are deleted.

        Args
        ----
        name: str
            Name of the test case.
        mesh: InputFile
            The InputFile object that contains the simulation.
        n_proc: int
            Number of processors to run Baci on.
        """

        # Check if temp directory exists.
        os.makedirs(testing_temp, exist_ok=True)

        # Create input file.
        input_file = os.path.join(testing_temp, name + '.dat')
        mesh.write_input_file(input_file)

        # Run Baci with the input file.
        child = subprocess.Popen([
            'mpirun', '-np', str(n_proc),
            baci_release,
            os.path.join(testing_path, input_file),
            os.path.join(testing_temp, 'xxx_' + name)
            ], cwd=testing_temp, stdout=subprocess.PIPE)
        child.communicate()[0]
        self.assertEqual(0, child.returncode,
            msg='Test {} failed!'.format(name))

        # If successful delete created files directory.
        if int(child.returncode) == 0:
            os.remove(input_file)
            items = glob.glob(testing_temp + '/xxx_' + name + '*')
            for item in items:
                if os.path.isdir(item):
                    shutil.rmtree(item)
                else:
                    os.remove(item)

    def test_honeycomb_sphere_as_input(self):
        """
        Test the honeycomb sphere model with different types of mesh import.
        """

        mpy.set_default_values()
        self.create_honeycomb_sphere_as_input('honeycomb_sphere')

        mpy.set_default_values()
        mpy.import_mesh_full = not mpy.import_mesh_full
        self.create_honeycomb_sphere_as_input('honeycomb_sphere_full_input')

    def create_honeycomb_sphere_as_input(self, name):
        """
        Create the same honeycomb mesh as defined in
        /Input/beam3r_herm2lin3_static_point_coupling_BTSPH_contact_stent_\
        honeycomb_stretch_r01_circ10.dat
        The honeycomb beam is in contact with a rigid sphere, the sphere is
        moved compared to the original test file, since there are some problems
        with the contact convergence. The sphere is imported as an existing
        mesh.
        """

        # Read input file with information of the sphere and simulation.
        input_file = InputFile(
            maintainer='Ivo Steinbrecher',
            description='honeycomb beam in contact with sphere',
            dat_file=os.path.join(testing_input,
                'baci_input_honeycomb_sphere.dat'))

        # Modify the time step options.
        input_file.add(InputSection(
            'STRUCTURAL DYNAMIC',
            'NUMSTEP 5',
            'TIMESTEP 0.2',
            option_overwrite=True
            ))

        # Delete the results given in the input file.
        input_file.delete_section('RESULT DESCRIPTION')
        input_file.add('-----RESULT DESCRIPTION')

        # Add result checks.
        displacement = [0.0, -8.09347205557697258, 2.89298034569662965]

        nodes = [268, 188, 182]
        for node in nodes:
            for i, direction in enumerate(['x', 'y', 'z']):
                input_file.add(
                    InputSection('RESULT DESCRIPTION',
                        ('STRUCTURE DIS structure NODE {} QUANTITY disp{} '
                        + 'VALUE {} TOLERANCE 1e-10').format(
                            node, direction, displacement[i]
                        )
                    )
                )

        # Material for the beam.
        material = MaterialReissner(
            youngs_modulus=2.07e2,
            radius=0.1,
            shear_correction=1.1)

        # Create the honeycomb mesh.
        mesh_honeycomb = Mesh()
        honeycomb_set = create_beam_mesh_honeycomb(mesh_honeycomb,
            Beam3rHerm2Lin3, material, 50.0, 10, 4, n_el=1, closed_top=False,
            add_sets=True)
        mesh_honeycomb.rotate(Rotation([0, 0, 1], 0.5 * np.pi))

        # Functions for the boundary conditions
        ft = Function(
            'COMPONENT 0 FUNCTION a\n'
            + 'VARIABLE 0 NAME a TYPE linearinterpolation NUMPOINTS 3 '
            + 'TIMES 0.0 0.2 1.0 VALUES 0.0 1.0 1.0'
            )
        mesh_honeycomb.add(ft)

        # Change the sets to lines, only for purpose of matching the test file
        honeycomb_set['bottom'].geo_type = mpy.geo.line
        honeycomb_set['top'].geo_type = mpy.geo.line
        mesh_honeycomb.add(
            BoundaryCondition(honeycomb_set['bottom'],
                'NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0 VAL 0 0 0 0 0 0 0 0 0 '
                + 'FUNCT 0 0 0 0 0 0 0 0 0',
                bc_type=mpy.bc.dirichlet))
        mesh_honeycomb.add(
            BoundaryCondition(honeycomb_set['top'],
                'NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0 VAL 0 0 5.0 0 0 0 0 0 0 '
                + 'FUNCT 0 0 {} 0 0 0 0 0 0',
                format_replacement=[ft],
                bc_type=mpy.bc.dirichlet))

        # Add the mesh to the imported solid mesh.
        input_file.add(mesh_honeycomb)

        # Run the input file in Baci.
        self.run_baci_test(name, input_file)

    def test_beam_and_solid_tube(self):
        """
        Test the honeycomb sphere model with different types of mesh import.
        """

        mpy.set_default_values()
        self.create_beam_and_solid_tube('beam_and_solid_tube')

        mpy.set_default_values()
        mpy.import_mesh_full = not mpy.import_mesh_full
        self.create_beam_and_solid_tube('beam_and_solid_tube')

    def create_beam_and_solid_tube(self, name):
        """Merge a solid tube with a beam tube and simulate them together."""

        # Create the input file and read solid mesh data.
        input_file = InputFile(
            maintainer='Ivo Steinbrecher',
            description='Solid tube with beam tube')
        input_file.read_dat(os.path.join(testing_input, 'baci_input_tube.dat'))

        # Add options for beam_output.
        input_file.add(InputSection(
            'IO/RUNTIME VTK OUTPUT/BEAMS',
            '''
            OUTPUT_BEAMS                    Yes
            DISPLACEMENT                    Yes
            USE_ABSOLUTE_POSITIONS          Yes
            TRIAD_VISUALIZATIONPOINT        Yes
            STRAINS_GAUSSPOINT              Yes
            '''))

        # Add functions for boundary conditions and material.
        sin = Function('COMPONENT 0 FUNCTION sin(t*2*pi)')
        cos = Function('COMPONENT 0 FUNCTION cos(t*2*pi)')
        material = MaterialReissner(
            youngs_modulus=1e9,
            radius=0.25,
            shear_correction=0.75)
        input_file.add(sin, cos, material)

        # Add a straight beam.
        input_file.add(material)
        cantilever_set = create_beam_mesh_line(input_file, Beam3rHerm2Lin3,
            material, [2, 0, -5], [2, 0, 5], n_el=3)

        # Add boundary conditions.
        input_file.add(
            BoundaryCondition(
                cantilever_set['start'],
                'NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL 0 0 0 0 0 0 0 0 0 ' + \
                'FUNCT 0 0 0 0 0 0 0 0 0',
                bc_type=mpy.bc.dirichlet
                )
            )
        input_file.add(
            BoundaryCondition(
                cantilever_set['end'],
                'NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL 3. 3. 0 0 0 0 0 0 0 ' + \
                'FUNCT {} {} 0 0 0 0 0 0 0',
                format_replacement=[cos, sin],
                bc_type=mpy.bc.dirichlet
                )
            )

        # Add result checks.
        displacement = [
            [1.50796091342925, 1.31453288915877e-8, 0.0439008100184687],
            [0.921450108160878, 1.41113401669104e-15, 0.0178350143764099]
            ]

        nodes = [35, 69]
        for j, node in enumerate(nodes):
            for i, direction in enumerate(['x', 'y', 'z']):
                input_file.add(
                    InputSection('RESULT DESCRIPTION',
                        ('STRUCTURE DIS structure NODE {} QUANTITY disp{} '
                        + 'VALUE {} TOLERANCE 1e-10').format(
                            node, direction, displacement[j][i]
                        )
                    )
                )

        # Call get_unique_geometry_sets to check that this does not affect the
        # mesh creation.
        input_file.get_unique_geometry_sets(link_nodes=True)

        # Run the input file in Baci.
        self.run_baci_test(name, input_file)

    def test_honeycomb_variants(self):
        """
        Create a few different honeycomb structures.
        """

        # Set default values for global parameters.
        mpy.set_default_values()

        # Create input file.
        input_file = InputFile(
            maintainer='Ivo Steinbrecher',
            description='Varieties of honeycomb'
            )

        # Set options with different syntaxes.
        input_file.add(InputSection('PROBLEM SIZE', 'DIM 3'))
        input_file.add('''
        ------------------------------------PROBLEM TYP
        PROBLEMTYP                            Structure
        RESTART                               0
        --------------------------------------IO
        OUTPUT_BIN                            No
        STRUCT_DISP                           No
        FILESTEPS                             1000
        VERBOSITY                             Standard
        ''')
        input_file.add(InputSection(
            'IO/RUNTIME VTK OUTPUT',
            '''
            OUTPUT_DATA_FORMAT                    binary
            INTERVAL_STEPS                        1
            EVERY_ITERATION                       No
            '''))
        input_file.add('''
            ------------------------------------STRUCTURAL DYNAMIC
            LINEAR_SOLVER                         1
            INT_STRATEGY                          Standard
            DYNAMICTYP                            Statics
            RESULTSEVRY                           1
            NLNSOL                                fullnewton
            PREDICT                               TangDis
            TIMESTEP                              1.
            NUMSTEP                               666
            MAXTIME                               10.0
            TOLRES                                1.0E-4
            TOLDISP                               1.0E-11
            NORM_RESF                             Abs
            NORM_DISP                             Abs
            NORMCOMBI_RESFDISP                    And
            MAXITER                               20
            ''')
        input_file.add(InputSection('STRUCTURAL DYNAMIC', 'NUMSTEP 1',
            option_overwrite=True))
        input_file.add(InputSection(
            'SOLVER 1',
            '''
            NAME                                  Structure_Solver
            SOLVER                                UMFPACK
            '''))
        input_file.add(InputSection(
            'IO/RUNTIME VTK OUTPUT/BEAMS',
            '''
            OUTPUT_BEAMS                    Yes
            DISPLACEMENT                    Yes
            USE_ABSOLUTE_POSITIONS          Yes
            TRIAD_VISUALIZATIONPOINT        Yes
            STRAINS_GAUSSPOINT              Yes
            '''))

        # Create four meshes with different types of honeycomb structure.
        mesh = Mesh()
        material = MaterialReissner(
            youngs_modulus=2.07e2,
            radius=0.1,
            shear_correction=1.1)
        ft = []
        ft.append(Function('COMPONENT 0 FUNCTION t'))
        ft.append(Function('COMPONENT 0 FUNCTION t'))
        ft.append(Function('COMPONENT 0 FUNCTION t'))
        ft.append(Function('COMPONENT 0 FUNCTION t'))
        mesh.add(ft)

        counter = 0
        for vertical in [False, True]:
            for closed_top in [False, True]:
                mesh.translate(17 * np.array([1, 0, 0]))
                honeycomb_set = create_beam_mesh_honeycomb(mesh,
                    Beam3rHerm2Lin3, material, 10, 6, 3, n_el=2,
                    vertical=vertical, closed_top=closed_top)
                mesh.add(
                    BoundaryCondition(honeycomb_set['bottom'],
                        'NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0 VAL '
                        + '0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0',
                        bc_type=mpy.bc.dirichlet))
                mesh.add(
                    BoundaryCondition(honeycomb_set['top'],
                        'NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0 VAL {1} {1} {1} '
                        + '0 0 0 0 0 0 FUNCT {0} {0} {0} 0 0 0 0 0 0',
                        format_replacement=[ft[counter], 0.0001],
                        bc_type=mpy.bc.neumann,
                        double_nodes=mpy.double_nodes.remove))
                counter += 1

        # Add mesh to input file.
        input_file.add(mesh)

        # Add result checks.
        displacement = [
            [1.319174933218672e-1, 1.993351916746171e-1, 6.922088404929467e-2],
            [1.329830005897997e-1, 2.005554487583632e-1, 6.970029733865361e-2],
            [7.692745378041407e-2, 1.249938079679132e-1, 5.867996421231717e-2],
            [6.988029670094088e-2, 1.098925957039341e-1, 4.835259164485453e-2]
            ]

        nodes = [190, 470, 711, 1071]
        for j, node in enumerate(nodes):
            for i, direction in enumerate(['x', 'y', 'z']):
                input_file.add(
                    InputSection('RESULT DESCRIPTION',
                        ('STRUCTURE DIS structure NODE {} QUANTITY disp{} '
                        + 'VALUE {} TOLERANCE 1e-10').format(
                            node, direction, displacement[j][i]
                        )
                    )
                )

        # Run the input file in Baci.
        self.run_baci_test('honeycomb_variants', input_file)

    def test_rotated_beam_axis(self):
        """
        Create three beams that consist of two connected lines.
        - The first case uses the same nodes for the connection of the lines,
          and the nodes are equal in this case.
        - The second case uses the same nodes for the connection of the lines,
          but the nodes have a different rotation along the basis vector 1.
        - The third case uses two nodes at the connection between the lines,
          and couples them with a coupling.
        """

        # Set default values for global parameters.
        mpy.set_default_values()

        # Create input file.
        input_file = InputFile(
            maintainer='Ivo Steinbrecher',
            description='Rotation of beam along axis'
            )

        # Set header
        set_header_static(input_file,
            time_step=0.05,
            n_steps=20
            )

        # Define linear function over time.
        ft = Function('COMPONENT 0 FUNCTION t')
        input_file.add(ft)

        # Set beam material.
        mat = MaterialReissner(
            youngs_modulus=2.07e2,
            radius=0.1,
            shear_correction=1.1)

        # Direction of the lines and the rotation between the beams.
        direction = np.array([0.5, 1, 2])
        alpha = np.pi / 27 * 7
        force_fac = 0.01

        # Create mesh.
        for i in range(3):
            mesh = Mesh()

            # Create the first line.
            set_1 = create_beam_mesh_line(mesh, Beam3rHerm2Lin3, mat,
                [0, 0, 0], 1. * direction, n_el=3)

            if not i == 0:
                # In the second case rotate the line, so the triads do not
                # match any more.
                mesh.rotate(Rotation(direction, alpha))

            if i == 2:
                # The third line is with couplings.
                start_node = None
            else:
                start_node = set_1['end']

            # Add the second line.
            set_2 = create_beam_mesh_line(mesh, Beam3rHerm2Lin3, mat,
                1. * direction,
                2. * direction,
                n_el=3, start_node=start_node
                )

            # Add boundary conditions.
            mesh.add(BoundaryCondition(set_1['start'],
                'NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL '
                + '0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0',
                bc_type=mpy.bc.dirichlet))
            mesh.add(BoundaryCondition(set_2['end'],
                'NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL '
                + '{1} {1} {1} {1} {1} {1} 0 0 0 FUNCT {0} {0} {0} {0} {0} {0}'
                + ' 0 0 0',
                bc_type=mpy.bc.neumann,
                format_replacement=[ft, force_fac]))

            if i == 2:
                # In the third case add a coupling.
                mesh.couple_nodes()

            # Add the mesh to the input file.
            input_file.add(mesh)

            # Each time move the whole mesh.
            input_file.translate([1, 0, 0])

        # Add result checks.
        displacement = [1.5015284845, 0.35139255451, -1.0126517891]
        nodes = [13, 26, 40]
        for node in nodes:
            for i, direction in enumerate(['x', 'y', 'z']):
                input_file.add(
                    InputSection('RESULT DESCRIPTION',
                        ('STRUCTURE DIS structure NODE {} QUANTITY disp{} '
                        + 'VALUE {} TOLERANCE 1e-10').format(
                            node, direction, displacement[i]
                        )
                    )
                )

        # Run the input file in Baci.
        self.run_baci_test('rotated_beam_axis', input_file)


if __name__ == '__main__':
    # Execution part of script.
    unittest.main()
