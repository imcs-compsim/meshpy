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
This script is used to test the functionality of the meshpy module.
"""

# Python imports.
import os
import unittest
import numpy as np
import autograd.numpy as npAD
import random
import warnings
import vtk

# Meshpy imports.
from meshpy import (
    mpy,
    Rotation,
    InputFile,
    MaterialReissner,
    MaterialReissnerElastoplastic,
    MaterialBeam,
    BoundaryCondition,
    MaterialKirchhoff,
    Mesh,
    Coupling,
    Beam3rHerm2Line3,
    Function,
    MaterialEulerBernoulli,
    Beam3eb,
    InputSection,
    Beam3k,
    set_header_static,
    set_beam_to_solid_meshtying,
    set_runtime_output,
    Beam3rLine2Line2,
    MaterialStVenantKirchhoff,
)
from meshpy.node import Node, NodeCosserat
from meshpy.vtk_writer import VTKWriter
from meshpy.geometry_set import GeometrySet, GeometrySetNodes
from meshpy.container import GeometryName
from meshpy.element_beam import Beam
from meshpy.utility import (
    get_single_node,
    get_min_max_coordinates,
)

# Geometry functions.
from meshpy.mesh_creation_functions.beam_basic_geometry import (
    create_beam_mesh_line,
    create_beam_mesh_arc_segment_via_rotation,
)
from meshpy.mesh_creation_functions.beam_honeycomb import create_beam_mesh_honeycomb
from meshpy.mesh_creation_functions.beam_curve import create_beam_mesh_curve

# Testing imports.
from utils import (
    skip_fail_test,
    testing_temp,
    testing_input,
    compare_strings,
    compare_test_result,
    compare_vtk,
)


def create_test_mesh(mesh):
    """Fill the mesh with a couple of test nodes and elements."""

    # Set the seed for the pseudo random numbers
    random.seed(0)

    # Add material to mesh.
    material = MaterialReissner()
    mesh.add(material)

    # Add three test nodes and add them to a beam element
    for _j in range(3):
        mesh.add(
            NodeCosserat(
                [100 * random.uniform(-1, 1) for _i in range(3)],
                Rotation(
                    [100 * random.uniform(-1, 1) for _i in range(3)],
                    100 * random.uniform(-1, 1),
                ),
            )
        )
    beam = Beam3rHerm2Line3(material=material, nodes=mesh.nodes)
    mesh.add(beam)

    # Add a beam line with three elements
    create_beam_mesh_line(
        mesh,
        Beam3rHerm2Line3,
        material,
        [100 * random.uniform(-1, 1) for _i in range(3)],
        [100 * random.uniform(-1, 1) for _i in range(3)],
        n_el=3,
    )


class TestMeshpy(unittest.TestCase):
    """Test various stuff from the meshpy module."""

    def setUp(self):
        """
        This method is called before each test and sets the default meshpy
        values for each test. The values can be changed in the individual
        tests.
        """

        # Set default values for global parameters.
        mpy.set_default_values()

    def test_meshpy_rotations(self):
        """
        Check if the Mesh function rotation gives the same results as rotating
        each node it self.
        """

        mesh_1 = InputFile()
        create_test_mesh(mesh_1)

        mesh_2 = InputFile()
        create_test_mesh(mesh_2)

        # Set the seed for the pseudo random numbers
        random.seed(0)
        rot = Rotation(
            [100 * random.uniform(-1, 1) for _i in range(3)],
            100 * random.uniform(-1, 1),
        )
        origin = [100 * random.uniform(-1, 1) for _i in range(3)]

        for node in mesh_1.nodes:
            node.rotate(rot, origin=origin)

        mesh_2.rotate(rot, origin=origin)

        # Compare the output for the two meshes.
        compare_strings(
            self, mesh_1.get_string(header=False), mesh_2.get_string(header=False)
        )

    def test_meshpy_mesh_rotations_individual(self):
        """
        Check if the Mesh function rotation gives the same results as rotating
        each node it self, when an array is passed with different rotations.
        """

        mesh_1 = InputFile()
        create_test_mesh(mesh_1)

        mesh_2 = InputFile()
        create_test_mesh(mesh_2)

        # Set the seed for the pseudo random numbers
        random.seed(0)

        # Rotate each node with a different rotation
        rotations = np.zeros([len(mesh_1.nodes), 4])
        origin = [100 * random.uniform(-1, 1) for _i in range(3)]
        for j, node in enumerate(mesh_1.nodes):
            rot = Rotation(
                [100 * random.uniform(-1, 1) for _i in range(3)],
                100 * random.uniform(-1, 1),
            )
            rotations[j, :] = rot.get_quaternion()
            node.rotate(rot, origin=origin)

        mesh_2.rotate(rotations, origin=origin)

        # Compare the output for the two meshes.
        compare_strings(
            self, mesh_1.get_string(header=False), mesh_2.get_string(header=False)
        )

    def test_meshpy_mesh_reflection(self):
        """Check the Mesh().reflect function."""

        def compare_reflection(origin=False, flip=False):
            """
            Create a mesh, and its mirrored counterpart and then compare the
            dat files.
            """

            # Rotations to be applied.
            rot_1 = Rotation([0, 1, 1], np.pi / 6)
            rot_2 = Rotation([1, 2.455, -1.2324], 1.2342352)

            mesh_ref = InputFile()
            mesh = InputFile()
            mat = MaterialReissner(radius=0.1)

            # Create the reference mesh.
            if not flip:
                create_beam_mesh_line(
                    mesh_ref, Beam3rHerm2Line3, mat, [0, 0, 0], [1, 0, 0], n_el=1
                )
                create_beam_mesh_line(
                    mesh_ref, Beam3rHerm2Line3, mat, [1, 0, 0], [1, 1, 0], n_el=1
                )
                create_beam_mesh_line(
                    mesh_ref, Beam3rHerm2Line3, mat, [1, 1, 0], [1, 1, 1], n_el=1
                )
            else:
                create_beam_mesh_line(
                    mesh_ref, Beam3rHerm2Line3, mat, [1, 0, 0], [0, 0, 0], n_el=1
                )
                create_beam_mesh_line(
                    mesh_ref, Beam3rHerm2Line3, mat, [1, 1, 0], [1, 0, 0], n_el=1
                )
                create_beam_mesh_line(
                    mesh_ref, Beam3rHerm2Line3, mat, [1, 1, 1], [1, 1, 0], n_el=1
                )

                # Reorder the internal nodes.
                old = mesh_ref.nodes.copy()
                mesh_ref.nodes[0] = old[2]
                mesh_ref.nodes[2] = old[0]
                mesh_ref.nodes[3] = old[5]
                mesh_ref.nodes[5] = old[3]
                mesh_ref.nodes[6] = old[8]
                mesh_ref.nodes[8] = old[6]

            mesh_ref.rotate(rot_1)

            # Create the mesh that will be mirrored.
            create_beam_mesh_line(
                mesh, Beam3rHerm2Line3, mat, [0, 0, 0], [-1, 0, 0], n_el=1
            )
            create_beam_mesh_line(
                mesh, Beam3rHerm2Line3, mat, [-1, 0, 0], [-1, 1, 0], n_el=1
            )
            create_beam_mesh_line(
                mesh, Beam3rHerm2Line3, mat, [-1, 1, 0], [-1, 1, 1], n_el=1
            )
            mesh.rotate(rot_1.inv())

            # Rotate everything, to show generalized reflection.
            mesh_ref.rotate(rot_2)
            mesh.rotate(rot_2)

            if origin:
                # Translate everything so the reflection plane is not in the
                # origin.
                r = [1, 2.455, -1.2324]
                mesh_ref.translate(r)
                mesh.translate(r)
                mesh.reflect(2 * (rot_2 * [1, 0, 0]), origin=r, flip_beams=flip)
            else:
                mesh.reflect(2 * (rot_2 * [1, 0, 0]), flip_beams=flip)

            # Compare the dat files.
            compare_strings(
                self, mesh_ref.get_string(header=False), mesh.get_string(header=False)
            )

        # Compare all 4 possible variations.
        for flip in [True, False]:
            for origin in [True, False]:
                compare_reflection(origin=origin, flip=flip)

    def create_comments_in_solid(self, full_import):
        """
        Check if comments in the solid file are handled correctly if they are
        inside a mesh section.
        """

        # Convert the solid mesh to meshpy objects.
        mpy.import_mesh_full = full_import

        solid_file = os.path.join(testing_input, "test_meshpy_comments_in_solid.dat")
        mesh = InputFile(dat_file=solid_file)

        # Add one element with BCs.
        mat = MaterialReissner()
        sets = create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 0, 0], [1, 2, 3])
        mesh.add(BoundaryCondition(sets["start"], "test", bc_type=mpy.bc.dirichlet))
        mesh.add(BoundaryCondition(sets["end"], "test", bc_type=mpy.bc.neumann))

        # Compare the output of the mesh.
        compare_test_result(self, mesh.get_string(header=False).strip())

    def test_meshpy_comments_in_solid(self):
        """Test case with full classical import."""
        self.create_comments_in_solid(False)

    def test_meshpy_comments_in_solid_full(self):
        """Test case with full solid import."""
        self.create_comments_in_solid(True)

    def test_meshpy_mesh_transformations_with_solid(self):
        """Test the different mesh transformation methods in combination with solid elements."""

        def base_test_mesh_translations(
            *, import_full=False, radius=None, reflect=True
        ):
            """
            Create the line and wrap it with passing radius to the wrap
            function.
            """

            # Set default values for global parameters.
            mpy.set_default_values()

            # Convert the solid mesh to meshpy objects.
            mpy.import_mesh_full = import_full

            # Create the mesh.
            mesh = InputFile(
                dat_file=os.path.join(testing_input, "4C_input_solid_cuboid.dat")
            )
            mat = MaterialReissner(radius=0.05)

            # Create the line.
            create_beam_mesh_line(
                mesh,
                Beam3rHerm2Line3,
                mat,
                [0.2, 0, 0],
                [0.2, 5 * 0.2 * 2 * np.pi, 4],
                n_el=3,
            )

            # Transform the mesh.
            mesh.wrap_around_cylinder(radius=radius)
            mesh.translate([1, 2, 3])
            mesh.rotate(Rotation([1, 2, 3], np.pi * 17.0 / 27.0))
            if reflect:
                mesh.reflect([0.1, -2, 1])

            # Check the output.
            ref_file = os.path.join(
                testing_input,
                "test_meshpy_mesh_transformations_with_solid_"
                + ("full" if import_full else "dat")
                + "_reference.dat",
            )
            compare_test_result(
                self,
                mesh.get_string(header=False),
                additional_identifier="full" if import_full else "dat",
            )

        base_test_mesh_translations(import_full=False, radius=None)
        base_test_mesh_translations(import_full=False, radius=0.2)
        base_test_mesh_translations(import_full=True, radius=0.2, reflect=False)

        # Not specifying or specifying the wrong radius should raise an error
        # In this case because everything is on one plane ("no" solid nodes in this case)
        # and we specify the radius
        self.assertRaises(
            ValueError,
            base_test_mesh_translations,
            import_full=False,
            radius=666,
            reflect=False,
        )
        # In this case because we need to specify a radius because with the solid nodes there
        # is no clear radius
        self.assertRaises(
            ValueError,
            base_test_mesh_translations,
            import_full=True,
            radius=None,
            reflect=False,
        )

    def test_meshpy_fluid_element_section(self):
        """Add beam elements to an input file containing fluid elements"""

        input_file = InputFile(
            dat_file=os.path.join(testing_input, "fluid_element_input.dat")
        )

        beam_mesh = Mesh()
        material = MaterialEulerBernoulli(youngs_modulus=1e8, radius=0.001, density=10)
        beam_mesh.add(material)

        create_beam_mesh_line(
            beam_mesh, Beam3eb, material, [0, -0.5, 0], [0, 0.2, 0], n_el=5
        )
        input_file.add(beam_mesh)

        # Check the output.
        compare_test_result(self, input_file.get_string(header=False))

    def test_meshpy_domain_geometry_sets(self):
        """Add geometry set based on a 4C internal domain"""

        input_file = InputFile()

        input_file.add(
            """
        -----------------------------------------------FLUID DOMAIN
        LOWER_BOUND     -1.5 -0.5 -0.5
        UPPER_BOUND     1.5 0.5 0.5
        INTERVALS       48 16 16
        ELEMENTS        FLUID HEX8 MAT 1 NA Euler
        PARTITION       structured
        -----------------------------------------------DLINE-NODE TOPOLOGY
        EDGE fluid y+ z+ DLINE 1
        -----------------------------------------------DSURF-NODE TOPOLOGY
        SIDE fluid z+ DSURFACE 1
        """
        )

        # Compare the output of the mesh.
        compare_test_result(self, input_file.get_string(header=False).strip())

    def test_meshpy_wrap_cylinder_not_on_same_plane(self):
        """Create a helix that is itself wrapped around a cylinder."""

        # Ignore the warnings from wrap around cylinder.
        warnings.filterwarnings("ignore")

        # Create the mesh.
        mesh = InputFile()
        mat = MaterialReissner(radius=0.05)

        # Create the line and bend it to a helix.
        create_beam_mesh_line(
            mesh,
            Beam3rHerm2Line3,
            mat,
            [0.2, 0, 0],
            [0.2, 5 * 0.2 * 2 * np.pi, 4],
            n_el=20,
        )
        mesh.wrap_around_cylinder()

        # Move the helix so its axis is in the y direction and goes through
        # (2 0 0). The helix is also moved by a lot in y-direction, this only
        # affects the angle phi when wrapping around a cylinder, not the shape
        # of the beam.
        mesh.rotate(Rotation([1, 0, 0], -0.5 * np.pi))
        mesh.translate([2, 666.666, 0])

        # Wrap the helix again.
        mesh.wrap_around_cylinder(radius=2.0)

        # Check the output.
        compare_test_result(self, mesh.get_string(header=False))

    def test_meshpy_get_nodes_by_function(self):
        """
        Check if the get_nodes_by_function method of Mesh works properly.
        """

        def get_nodes_at_x(node, x_value):
            """True for all coordinates at a certain x value."""
            if np.abs(node.coordinates[0] - x_value) < 1e-10:
                return True
            else:
                return False

        mat = MaterialReissner()

        mesh = Mesh()
        create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 0, 0], [5, 0, 0], n_el=5)
        create_beam_mesh_line(
            mesh, Beam3rHerm2Line3, mat, [0, 1, 0], [10, 1, 0], n_el=10
        )

        nodes = mesh.get_nodes_by_function(get_nodes_at_x, 1.0)
        self.assertTrue(2 == len(nodes))
        for node in nodes:
            self.assertTrue(np.abs(1.0 - node.coordinates[0]) < 1e-10)

    def test_meshpy_get_min_max_coordinates(self):
        """
        Test if the get_min_max_coordinates function works properly.
        """

        # Create the mesh.
        mpy.import_mesh_full = True
        mesh = InputFile(
            dat_file=os.path.join(testing_input, "4C_input_solid_cuboid.dat")
        )
        mat = MaterialReissner(radius=0.05)
        create_beam_mesh_line(
            mesh, Beam3rHerm2Line3, mat, [0, 0, 0], [2, 3, 4], n_el=10
        )

        # Check the results.
        min_max = get_min_max_coordinates(mesh.nodes)
        ref_solution = [-0.5, -1.0, -1.5, 2.0, 3.0, 4.0]
        self.assertTrue(np.linalg.norm(min_max - ref_solution) < 1e-10)

    def test_meshpy_geometry_sets(self):
        """Test functionality of the GeometrySet objects"""

        mesh = InputFile()
        for i in range(6):
            mesh.add(NodeCosserat([i, 2 * i, 3 * i], Rotation([i, 2 * i, 3 * i], i)))

        set_1 = GeometrySetNodes(
            mpy.geo.point, [mesh.nodes[0], mesh.nodes[1], mesh.nodes[2]]
        )
        set_2 = GeometrySetNodes(
            mpy.geo.point, [mesh.nodes[2], mesh.nodes[3], mesh.nodes[4]]
        )
        set_12 = GeometrySetNodes(mpy.geo.point)
        set_12.add(set_1)
        set_12.add(set_2)
        set_3 = GeometrySet(set_1.get_points())

        mesh.add(set_1, set_2, set_12, set_3)

        # Check the output.
        compare_test_result(self, mesh.get_string(header=False))

    def test_meshpy_reissner_beam(self):
        """
        Test that the input file for all types of Reissner beams is generated
        correctly.
        """

        # Create input file.
        material = MaterialReissner(
            radius=0.1, youngs_modulus=1000, interaction_radius=2.0
        )
        input_file = InputFile()

        # Create a beam arc with the different Reissner beam types.
        for i, beam_type in enumerate([Beam3rHerm2Line3, Beam3rLine2Line2]):
            create_beam_mesh_arc_segment_via_rotation(
                input_file,
                beam_type,
                material,
                [0.0, 0.0, i],
                Rotation([0.0, 0.0, 1.0], np.pi / 2.0),
                2.0,
                np.pi / 2.0,
                n_el=2,
            )

        # Compare with the reference solution.
        compare_test_result(self, input_file.get_string(header=False))

    def test_meshpy_reissner_elasto_plastic(self):
        """Test the elasto plastic Reissner beam material"""

        kwargs = {
            "radius": 0.1,
            "youngs_modulus": 1000,
            "interaction_radius": 2.0,
            "shear_correction": 5.0 / 6.0,
            "yield_moment": 2.3,
            "isohardening_modulus_moment": 4.5,
            "torsion_plasticity": False,
        }

        ref_string = "MAT 69 MAT_BeamReissnerElastPlastic YOUNG 1000 POISSONRATIO 0.0 DENS 0.0 CROSSAREA 0.031415926535897934 SHEARCORR 0.8333333333333334 MOMINPOL 0.00015707963267948968 MOMIN2 7.853981633974484e-05 MOMIN3 7.853981633974484e-05 INTERACTIONRADIUS 2.0 YIELDM 2.3 ISOHARDM 4.5 TORSIONPLAST "

        mat = MaterialReissnerElastoplastic(**kwargs)
        mat.n_global = 69
        self.assertEqual(mat.get_dat_lines(), [ref_string + "0"])

        kwargs["torsion_plasticity"] = True
        mat = MaterialReissnerElastoplastic(**kwargs)
        mat.n_global = 69
        self.assertEqual(mat.get_dat_lines(), [ref_string + "1"])

    def test_meshpy_kirchhoff_beam(self):
        """
        Test that the input file for all types of Kirchhoff beams is generated
        correctly.
        """

        # Create input file.
        material = MaterialKirchhoff(radius=0.1, youngs_modulus=1000)
        input_file = InputFile()

        with warnings.catch_warnings():
            # Ignore the warnings for the rotvec beams.
            warnings.simplefilter("ignore")

            # Loop over options.
            for is_fad in (True, False):
                for weak in (True, False):
                    for rotvec in (True, False):
                        # Define the beam object factory function for the
                        # creation functions.
                        BeamObject = Beam3k(weak=weak, rotvec=rotvec, is_fad=is_fad)

                        # Create a beam.
                        set_1 = create_beam_mesh_line(
                            input_file,
                            BeamObject,
                            material,
                            [0, 0, 0],
                            [1, 0, 0],
                            n_el=2,
                        )
                        set_2 = create_beam_mesh_line(
                            input_file,
                            BeamObject,
                            material,
                            [1, 0, 0],
                            [2, 0, 0],
                            n_el=2,
                        )

                        # Couple the nodes.
                        if rotvec:
                            input_file.couple_nodes(
                                nodes=[
                                    get_single_node(set_1["end"]),
                                    get_single_node(set_2["start"]),
                                ]
                            )

                        # Move the mesh away from the next created beam.
                        input_file.translate([0, 0.5, 0])

        # Compare with the reference solution.
        compare_test_result(self, input_file.get_string(header=False))

    def test_meshpy_euler_bernoulli(self):
        """
        Recreate the 4C test case beam3eb_static_endmoment_quartercircle.dat
        This tests the implementation for Euler Bernoulli beams.
        """

        # Create the input file and add function and material.
        input_file = InputFile()
        fun = Function("COMPONENT 0 SYMBOLIC_FUNCTION_OF_SPACE_TIME t")
        input_file.add(fun)
        mat = MaterialEulerBernoulli(youngs_modulus=1.0, density=1.3e9)

        # Set the parameters that are also set in the test file.
        mat.area = 1
        mat.mom2 = 1e-4

        # Create the beam.
        beam_set = create_beam_mesh_line(
            input_file, Beam3eb, mat, [-1, 0, 0], [1, 0, 0], n_el=16
        )

        # Add boundary conditions.
        input_file.add(
            BoundaryCondition(
                beam_set["start"],
                "NUMDOF 6 ONOFF 1 1 1 0 1 1 VAL 0.0 0.0 0.0 0.0 0.0 0.0 FUNCT 0 0 0 0 0 0",
                bc_type=mpy.bc.dirichlet,
            )
        )
        input_file.add(
            BoundaryCondition(
                beam_set["end"],
                "NUMDOF 6 ONOFF 0 0 0 0 0 1 VAL 0.0 0.0 0.0 0.0 0.0 7.8539816339744e-05 FUNCT 0 0 0 0 0 {}",
                bc_type=mpy.bc.moment_euler_bernoulli,
                format_replacement=[fun],
            )
        )

        # Compare with the reference solution.
        compare_test_result(self, input_file.get_string(header=False, check_nox=False))

        # Test consistency checks.
        rot = Rotation([1, 2, 3], 2.3434)
        input_file.nodes[-1].rotation = rot
        with self.assertRaises(ValueError):
            # This raises an error because not all rotation in the beams are
            # the same.
            input_file.get_string(header=False)

        for node in input_file.nodes:
            node.rotation = rot
        with self.assertRaises(ValueError):
            # This raises an error because the rotations do not match the
            # director between the nodes.
            input_file.get_string(header=False)

    def test_meshpy_close_beam(self):
        """
        Create a circle with different methods.
        - Create the mesh manually by creating the nodes and connecting them to
          the elements.
        - Create one full circle and connect it to its beginning.
        - Create two half circle and connect their start / end nodes.
        All of those methods should give the exact same mesh.
        Both variants are also tried with different rotations at the beginning.
        """

        # Parameters for this test case.
        n_el = 3
        R = 1.235
        additional_rotation = Rotation([0, 1, 0], 0.5)

        # Define material.
        mat = MaterialReissner(radius=0.1)

        def create_mesh_manually(start_rotation):
            """Create the full circle manually."""
            input_file = InputFile()
            input_file.add(mat)

            # Add nodes.
            for i in range(4 * n_el):
                basis = start_rotation * Rotation([0, 0, 1], np.pi * 0.5)
                r = [R, 0, 0]
                node = NodeCosserat(r, basis)
                rotation = Rotation([0, 0, 1], 0.5 * i * np.pi / n_el)
                node.rotate(rotation, origin=[0, 0, 0])
                input_file.nodes.append(node)

            # Add elements.
            for i in range(2 * n_el):
                node_index = [2 * i, 2 * i + 1, 2 * i + 2]
                nodes = []
                for index in node_index:
                    if index == len(input_file.nodes):
                        nodes.append(input_file.nodes[0])
                    else:
                        nodes.append(input_file.nodes[index])
                element = Beam3rHerm2Line3(mat, nodes)
                input_file.add(element)

            # Add sets.
            geom_set = GeometryName()
            geom_set["start"] = GeometrySet(input_file.nodes[0])
            geom_set["end"] = GeometrySet(input_file.nodes[0])
            geom_set["line"] = GeometrySet(input_file.elements)
            input_file.add(geom_set)
            return input_file

        def one_full_circle_closed(function, argument_list, additional_rotation=None):
            """Create one full circle and connect it to itself."""

            input_file = InputFile()

            if additional_rotation is not None:
                start_rotation = additional_rotation * Rotation([0, 0, 1], np.pi * 0.5)
                input_file.add(NodeCosserat([R, 0, 0], start_rotation))
                function(
                    input_file,
                    start_node=input_file.nodes[0],
                    end_node=True,
                    add_sets=True,
                    **(argument_list),
                )
            else:
                function(input_file, end_node=True, add_sets=True, **(argument_list))
            return input_file

        def two_half_circles_closed(function, argument_list, additional_rotation=None):
            """
            Create two half circles and close them, by reusing the connecting
            nodes.
            """

            input_file = InputFile()

            if additional_rotation is not None:
                start_rotation = additional_rotation * Rotation([0, 0, 1], np.pi * 0.5)
                input_file.add(NodeCosserat([R, 0, 0], start_rotation))
                set_1 = function(
                    input_file, start_node=input_file.nodes[0], **(argument_list[0])
                )
            else:
                set_1 = function(input_file, **(argument_list[0]))

            set_2 = function(
                input_file,
                start_node=set_1["end"],
                end_node=set_1["start"],
                **(argument_list[1]),
            )

            # Add sets.
            geom_set = GeometryName()
            geom_set["start"] = GeometrySet(set_1["start"])
            geom_set["end"] = GeometrySet(set_2["end"])
            geom_set["line"] = GeometrySet([set_1["line"], set_2["line"]])
            input_file.add(geom_set)

            return input_file

        def get_arguments_arc_segment(circle_type):
            """Return the arguments for the arc segment function."""
            if circle_type == 0:
                # Full circle.
                arg_rot_angle = np.pi / 2
                arg_angle = 2 * np.pi
                arg_n_el = 2 * n_el
            elif circle_type == 1:
                # First half circle.
                arg_rot_angle = np.pi / 2
                arg_angle = np.pi
                arg_n_el = n_el
            elif circle_type == 2:
                # Second half circle.
                arg_rot_angle = 3 * np.pi / 2
                arg_angle = np.pi
                arg_n_el = n_el
            return {
                "beam_object": Beam3rHerm2Line3,
                "material": mat,
                "center": [0, 0, 0],
                "axis_rotation": Rotation([0, 0, 1], arg_rot_angle),
                "radius": R,
                "angle": arg_angle,
                "n_el": arg_n_el,
            }

        def circle_function(t):
            """Function for the circle."""
            return R * npAD.array([npAD.cos(t), npAD.sin(t)])

        def get_arguments_curve(circle_type):
            """Return the arguments for the curve function."""
            if circle_type == 0:
                # Full circle.
                arg_interval = [0, 2 * np.pi]
                arg_n_el = 2 * n_el
            elif circle_type == 1:
                # First half circle.
                arg_interval = [0, np.pi]
                arg_n_el = n_el
            elif circle_type == 2:
                # Second half circle.
                arg_interval = [np.pi, 2 * np.pi]
                arg_n_el = n_el
            return {
                "beam_object": Beam3rHerm2Line3,
                "material": mat,
                "function": circle_function,
                "interval": arg_interval,
                "n_el": arg_n_el,
            }

        # Check the meshes without additional rotation.
        compare_test_result(
            self, create_mesh_manually(Rotation()).get_string(header=False)
        )
        compare_test_result(
            self,
            one_full_circle_closed(
                create_beam_mesh_arc_segment_via_rotation, get_arguments_arc_segment(0)
            ).get_string(header=False),
        )
        compare_test_result(
            self,
            two_half_circles_closed(
                create_beam_mesh_arc_segment_via_rotation,
                [get_arguments_arc_segment(1), get_arguments_arc_segment(2)],
            ).get_string(header=False),
        )
        compare_test_result(
            self,
            one_full_circle_closed(
                create_beam_mesh_curve, get_arguments_curve(0)
            ).get_string(header=False),
        )
        compare_test_result(
            self,
            two_half_circles_closed(
                create_beam_mesh_curve, [get_arguments_curve(1), get_arguments_curve(2)]
            ).get_string(header=False),
        )

        # Check the meshes with additional rotation.
        additional_identifier = "rotation"
        compare_test_result(
            self,
            create_mesh_manually(additional_rotation).get_string(header=False),
            additional_identifier=additional_identifier,
        )
        compare_test_result(
            self,
            one_full_circle_closed(
                create_beam_mesh_arc_segment_via_rotation,
                get_arguments_arc_segment(0),
                additional_rotation=additional_rotation,
            ).get_string(header=False),
            additional_identifier=additional_identifier,
        )
        compare_test_result(
            self,
            two_half_circles_closed(
                create_beam_mesh_arc_segment_via_rotation,
                [get_arguments_arc_segment(1), get_arguments_arc_segment(2)],
                additional_rotation=additional_rotation,
            ).get_string(header=False),
            additional_identifier=additional_identifier,
        )
        compare_test_result(
            self,
            one_full_circle_closed(
                create_beam_mesh_curve,
                get_arguments_curve(0),
                additional_rotation=additional_rotation,
            ).get_string(header=False),
            additional_identifier=additional_identifier,
        )
        compare_test_result(
            self,
            two_half_circles_closed(
                create_beam_mesh_curve,
                [get_arguments_curve(1), get_arguments_curve(2)],
                additional_rotation=additional_rotation,
            ).get_string(header=False),
            additional_identifier=additional_identifier,
        )

    def test_meshpy_replace_nodes_geometry_set(self):
        """
        Test case for coupling of nodes, and reusing the identical nodes.
        This test case uses geometry-based sets.
        """
        self.x_test_replace_nodes(False)

    def test_meshpy_replace_nodes_geometry_set_nodes(self):
        """
        Test case for coupling of nodes, and reusing the identical nodes.
        This test case uses node-based sets.
        """
        self.x_test_replace_nodes(True)

    def x_test_replace_nodes(self, use_nodal_geometry_sets=False):
        """Test case for coupling of nodes, and reusing the identical nodes."""

        mpy.check_overlapping_elements = False

        mat = MaterialReissner(radius=0.1, youngs_modulus=1)
        rot = Rotation([1, 2, 43], 213123)

        def create_mesh():
            """Create two empty meshes."""
            return InputFile(), InputFile()

        # Create a beam with two elements. Once immediately and once as two
        # beams with couplings.
        mesh_ref, mesh_couple = create_mesh()

        # Create a simple beam.
        create_beam_mesh_line(
            mesh_ref, Beam3rHerm2Line3, mat, [0, 0, 0], [2, 0, 0], n_el=2
        )
        create_beam_mesh_line(mesh_couple, Beam3rHerm2Line3, mat, [0, 0, 0], [1, 0, 0])
        create_beam_mesh_line(mesh_couple, Beam3rHerm2Line3, mat, [1, 0, 0], [2, 0, 0])

        ref_nodes = list(mesh_ref.nodes)
        coupling_nodes = list(mesh_couple.nodes)

        # Add a set with all nodes, to check that the nodes in the
        # boundary condition are replaced correctly.
        if use_nodal_geometry_sets:
            mesh_ref.add(GeometrySetNodes(mpy.geo.point, ref_nodes))
            mesh_couple.add(GeometrySetNodes(mpy.geo.point, coupling_nodes))
        else:
            mesh_ref.add(GeometrySet(ref_nodes))
            mesh_couple.add(GeometrySet(coupling_nodes))

        # Add another set with all nodes, this time only the coupling node
        # that will be kept is in this set.
        coupling_nodes_without_replace_node = list(coupling_nodes)
        del coupling_nodes_without_replace_node[3]
        if use_nodal_geometry_sets:
            mesh_ref.add(GeometrySetNodes(mpy.geo.point, ref_nodes))
            mesh_couple.add(
                GeometrySetNodes(mpy.geo.point, coupling_nodes_without_replace_node)
            )
        else:
            mesh_ref.add(GeometrySet(ref_nodes))
            mesh_couple.add(GeometrySet(coupling_nodes_without_replace_node))

        # Add another set with all nodes, this time only the coupling node
        # that will be replaced is in this set.
        coupling_nodes_without_replace_node = list(coupling_nodes)
        del coupling_nodes_without_replace_node[2]
        if use_nodal_geometry_sets:
            mesh_ref.add(GeometrySetNodes(mpy.geo.point, ref_nodes))
            mesh_couple.add(
                GeometrySetNodes(mpy.geo.point, coupling_nodes_without_replace_node)
            )
        else:
            mesh_ref.add(GeometrySet(ref_nodes))
            mesh_couple.add(GeometrySet(coupling_nodes_without_replace_node))

        # Rotate both meshes
        mesh_ref.rotate(rot)
        mesh_couple.rotate(rot)

        # Couple the coupling mesh.
        mesh_couple.couple_nodes(
            coupling_dof_type=mpy.coupling_dof.fix, reuse_matching_nodes=True
        )

        # Compare the meshes.
        compare_strings(
            self,
            mesh_ref.get_string(header=False),
            mesh_couple.get_string(header=False),
        )

        # Create two overlapping beams. This is to test that the middle nodes
        # are not coupled.
        mesh_ref, mesh_couple = create_mesh()

        # Create a simple beam.
        set_ref = create_beam_mesh_line(
            mesh_ref, Beam3rHerm2Line3, mat, [0, 0, 0], [1, 0, 0]
        )
        create_beam_mesh_line(
            mesh_ref,
            Beam3rHerm2Line3,
            mat,
            [0, 0, 0],
            [1, 0, 0],
            start_node=set_ref["start"],
            end_node=set_ref["end"],
        )
        create_beam_mesh_line(mesh_couple, Beam3rHerm2Line3, mat, [0, 0, 0], [1, 0, 0])
        create_beam_mesh_line(mesh_couple, Beam3rHerm2Line3, mat, [0, 0, 0], [1, 0, 0])

        # Rotate both meshes
        mesh_ref.rotate(rot)
        mesh_couple.rotate(rot)

        # Couple the coupling mesh.
        mesh_couple.couple_nodes(
            coupling_dof_type=mpy.coupling_dof.fix, reuse_matching_nodes=True
        )

        # Compare the meshes.
        compare_strings(
            self,
            mesh_ref.get_string(header=False),
            mesh_couple.get_string(header=False),
        )

        # Create a beam with two elements. Once immediately and once as two
        # beams with couplings.
        mesh_ref, mesh_couple = create_mesh()

        # Create a simple beam.
        create_beam_mesh_line(
            mesh_ref, Beam3rHerm2Line3, mat, [0, 0, 0], [2, 0, 0], n_el=2
        )
        create_beam_mesh_line(mesh_couple, Beam3rHerm2Line3, mat, [0, 0, 0], [1, 0, 0])
        create_beam_mesh_line(mesh_couple, Beam3rHerm2Line3, mat, [1, 0, 0], [2, 0, 0])

        # Create set with all the beam nodes.
        if use_nodal_geometry_sets:
            node_set_1_ref = GeometrySetNodes(mpy.geo.line, mesh_ref.nodes)
            node_set_2_ref = GeometrySetNodes(mpy.geo.line, mesh_ref.nodes)
            node_set_1_couple = GeometrySetNodes(mpy.geo.line, mesh_couple.nodes)
            node_set_2_couple = GeometrySetNodes(mpy.geo.line, mesh_couple.nodes)
        else:
            node_set_1_ref = GeometrySet(mesh_ref.elements)
            node_set_2_ref = GeometrySet(mesh_ref.elements)
            node_set_1_couple = GeometrySet(mesh_couple.elements)
            node_set_2_couple = GeometrySet(mesh_couple.elements)

        # Create connecting beams.
        create_beam_mesh_line(mesh_ref, Beam3rHerm2Line3, mat, [1, 0, 0], [2, 2, 2])
        create_beam_mesh_line(mesh_ref, Beam3rHerm2Line3, mat, [1, 0, 0], [2, -2, -2])
        create_beam_mesh_line(mesh_couple, Beam3rHerm2Line3, mat, [1, 0, 0], [2, 2, 2])
        create_beam_mesh_line(
            mesh_couple, Beam3rHerm2Line3, mat, [1, 0, 0], [2, -2, -2]
        )

        # Rotate both meshes
        mesh_ref.rotate(rot)
        mesh_couple.rotate(rot)

        # Couple the mesh.
        mesh_ref.couple_nodes(coupling_dof_type=mpy.coupling_dof.fix)
        mesh_couple.couple_nodes(
            coupling_dof_type=mpy.coupling_dof.fix, reuse_matching_nodes=True
        )

        # Add the node sets.
        mesh_ref.add(node_set_1_ref)
        mesh_couple.add(node_set_1_couple)

        # Add BCs.
        mesh_ref.add(BoundaryCondition(node_set_2_ref, "BC1", bc_type=mpy.bc.neumann))
        mesh_couple.add(
            BoundaryCondition(node_set_2_couple, "BC1", bc_type=mpy.bc.neumann)
        )

        # Compare the meshes.
        compare_strings(
            self,
            mesh_ref.get_string(header=False),
            mesh_couple.get_string(header=False),
        )

    def create_beam_to_solid_conditions_model(self):
        """
        Create the input file for the beam-to-solid input conditions tests.
        """

        # Create input file.
        input_file = InputFile(
            dat_file=os.path.join(
                testing_input, "test_meshpy_btsvm_coupling_solid_mesh.dat"
            ),
        )

        # Add beams to the model.
        beam_mesh = Mesh()
        material = MaterialReissner(youngs_modulus=1000, radius=0.05)
        create_beam_mesh_line(
            beam_mesh, Beam3rHerm2Line3, material, [0, 0, 0], [0, 0, 1], n_el=3
        )
        create_beam_mesh_line(
            beam_mesh, Beam3rHerm2Line3, material, [0, 0.5, 0], [0, 0.5, 1], n_el=3
        )

        # Set beam-to-solid coupling conditions.
        line_set = GeometrySet(beam_mesh.elements)
        beam_mesh.add(
            BoundaryCondition(
                line_set,
                bc_type=mpy.bc.beam_to_solid_volume_meshtying,
                bc_string="COUPLING_ID 1",
            )
        )
        beam_mesh.add(
            BoundaryCondition(
                line_set,
                bc_type=mpy.bc.beam_to_solid_surface_meshtying,
                bc_string="COUPLING_ID 2",
            )
        )

        # Add the beam to the solid mesh.
        input_file.add(beam_mesh)
        return input_file

    def test_meshpy_beam_to_solid_conditions(self):
        """
        Create beam-to-solid input conditions.
        """

        # Create input file.
        input_file = self.create_beam_to_solid_conditions_model()

        # Compare with the reference file.
        compare_test_result(self, input_file.get_string(header=False))

    def test_meshpy_beam_to_solid_conditions_with_design_description(self):
        """
        Create beam-to-solid input conditions with the old design description section in 4C
        """

        # Create input file.
        input_file = self.create_beam_to_solid_conditions_model()

        # Compare with the reference file.
        compare_test_result(
            self, input_file.get_string(header=False, design_description=True)
        )

    def test_meshpy_beam_to_solid_conditions_full(self):
        """
        Create beam-to-solid input conditions with full import.
        """

        # Create input file.
        mpy.import_mesh_full = True
        input_file = self.create_beam_to_solid_conditions_model()

        # Compare with the reference file.
        compare_test_result(self, input_file.get_string(header=False))

    def test_meshpy_surface_to_surface_contact_import(self):
        """Test that surface-to-surface contact problems can be imported as expected"""

        # Create input file.
        mpy.import_mesh_full = True
        input_file = InputFile(
            dat_file=os.path.join(
                testing_input, self._testMethodName + "_solid_mesh.dat"
            )
        )

        # Compare with the reference file.
        compare_test_result(self, input_file.get_string(header=False))

    def test_meshpy_nurbs_import(self):
        """
        Test if the import of a NURBS mesh works as expected.
        This script generates the 4C test case:
        beam3r_herm2line3_static_beam_to_solid_volume_meshtying_nurbs27_mortar_penalty_line4
        """

        # Create beam mesh and load solid file.
        input_file = InputFile(
            dat_file=os.path.join(
                testing_input, "test_meshpy_nurbs_import_solid_mesh.dat"
            ),
        )
        set_header_static(
            input_file,
            time_step=0.5,
            n_steps=2,
            tol_residuum=1e-14,
            tol_increment=1e-8,
            option_overwrite=True,
        )
        set_beam_to_solid_meshtying(
            input_file,
            mpy.beam_to_solid.volume_meshtying,
            contact_discretization="mortar",
            mortar_shape="line4",
            penalty_parameter=1000,
            n_gauss_points=6,
            segmentation=True,
            binning_bounding_box=[-3, -3, -1, 3, 3, 5],
            binning_cutoff_radius=1,
        )
        set_runtime_output(input_file, output_solid=False)
        input_file.add(
            InputSection(
                "IO",
                """
            OUTPUT_BIN     yes
            STRUCT_DISP    yes
            FILESTEPS      1000
            VERBOSITY      Standard
            """,
                option_overwrite=True,
            )
        )
        fun = Function("COMPONENT 0 SYMBOLIC_FUNCTION_OF_SPACE_TIME t")
        input_file.add(fun)

        # Create the beam material.
        material = MaterialReissner(youngs_modulus=1000, radius=0.05)

        # Create the beams.
        set_1 = create_beam_mesh_line(
            input_file, Beam3rHerm2Line3, material, [0, 0, 0.95], [1, 0, 0.95], n_el=2
        )
        set_2 = create_beam_mesh_line(
            input_file,
            Beam3rHerm2Line3,
            material,
            [-0.25, -0.3, 0.85],
            [-0.25, 0.5, 0.85],
            n_el=2,
        )

        # Add boundary conditions on the beams.
        input_file.add(
            BoundaryCondition(
                set_1["start"],
                "NUMDOF 9 ONOFF 0 0 0 1 1 1 0 0 0 VAL 0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0",
                bc_type=mpy.bc.dirichlet,
            )
        )
        input_file.add(
            BoundaryCondition(
                set_1["end"],
                "NUMDOF 9 ONOFF 0 1 0 0 0 0 0 0 0 VAL 0 0.02 0 0 0 0 0 0 0 FUNCT 0 {} 0 0 0 0 0 0 0",
                format_replacement=[fun],
                bc_type=mpy.bc.neumann,
            )
        )
        input_file.add(
            BoundaryCondition(
                set_2["start"],
                "NUMDOF 9 ONOFF 0 0 0 1 1 1 0 0 0 VAL 0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0",
                bc_type=mpy.bc.dirichlet,
            )
        )
        input_file.add(
            BoundaryCondition(
                set_2["end"],
                "NUMDOF 9 ONOFF 1 0 0 0 0 0 0 0 0 VAL -0.06 0 0 0 0 0 0 0 0 "
                "FUNCT {} 0 0 0 0 0 0 0 0",
                format_replacement=[fun],
                bc_type=mpy.bc.neumann,
            )
        )

        # Add result checks.
        displacement = [
            [
                -5.14451531793581718e-01,
                -1.05846397858073843e-01,
                -1.77822866851472888e-01,
            ]
        ]

        nodes = [64]
        for j, node in enumerate(nodes):
            for i, direction in enumerate(["x", "y", "z"]):
                input_file.add(
                    InputSection(
                        "RESULT DESCRIPTION",
                        (
                            "STRUCTURE DIS structure NODE {} QUANTITY disp{} "
                            "VALUE {} TOLERANCE 1e-10"
                        ).format(node, direction, displacement[j][i]),
                    )
                )

        # Compare with the reference solution.
        compare_test_result(self, input_file.get_string(header=False, check_nox=False))

    def test_meshpy_stvenantkirchhoff_solid(self):
        """
        Test that the input file for a solid with St. Venant Kirchhoff material properties is generated
        correctly
        """

        # Create materials
        material_1 = MaterialStVenantKirchhoff(
            youngs_modulus=157, nu=0.17, density=6.1e-7
        )

        material_2 = MaterialStVenantKirchhoff(
            youngs_modulus=370, nu=0.20, density=5.2e-7
        )

        # Create input file
        input_file = InputFile()
        input_file.add(material_1)
        input_file.add(material_2)

        # Compare with the reference file
        compare_test_result(self, input_file.get_string(header=False))

    def test_meshpy_point_couplings(self):
        """
        Test that the different point coupling types can be created.
        """

        # The "old" way of coupling points.
        input_file = self.x_test_point_couplings(
            mpy.bc.point_coupling, mpy.coupling_dof.fix
        )
        compare_test_result(
            self, input_file.get_string(header=False), additional_identifier="exact"
        )

        # The "new" way of coupling points.
        input_file = self.x_test_point_couplings(
            mpy.bc.point_coupling_penalty, "PENALTY_VALUE"
        )
        compare_test_result(
            self, input_file.get_string(header=False), additional_identifier="penalty"
        )

    def x_test_point_couplings(self, coupling_type, coupling_dof_type):
        """
        Create the input file for the test_point_couplings method.
        """

        # Create input file.
        material = MaterialReissner(
            radius=0.1, youngs_modulus=1000, interaction_radius=2.0
        )
        input_file = InputFile()

        # Create a 2x2 grid of beams.
        for i in range(3):
            for j in range(2):
                create_beam_mesh_line(
                    input_file, Beam3rHerm2Line3, material, [j, i, 0.0], [j + 1, i, 0.0]
                )
                create_beam_mesh_line(
                    input_file, Beam3rHerm2Line3, material, [i, j, 0.0], [i, j + 1, 0.0]
                )

        # Couple the beams.
        input_file.couple_nodes(
            reuse_matching_nodes=True,
            coupling_type=coupling_type,
            coupling_dof_type=coupling_dof_type,
        )

        return input_file

    def test_meshpy_vtk_writer(self):
        """Test the output created by the VTK writer."""

        # Initialize writer.
        writer = VTKWriter()

        # Add poly line.
        indices = writer.add_points([[0, 0, -2], [1, 1, -2], [2, 2, -1]])
        writer.add_cell(vtk.vtkPolyLine, indices)

        # Add quadratic quad.
        cell_data = {}
        cell_data["cell_data_1"] = 3
        cell_data["cell_data_2"] = [66, 0, 1]
        point_data = {}
        point_data["point_data_1"] = [1, 2, 3, 4, 5, -2, -3, 0]
        point_data["point_data_2"] = [
            [0.25, 0, -0.25],
            [1, 0.25, 0],
            [2, 0, 0],
            [2.25, 1.25, 0.5],
            [2, 2.25, 0],
            [1, 2, 0.5],
            [0, 2.25, 0],
            [0, 1, 0.5],
        ]
        indices = writer.add_points(
            [
                [0.25, 0, -0.25],
                [1, 0.25, 0],
                [2, 0, 0],
                [2.25, 1.25, 0.5],
                [2, 2.25, 0],
                [1, 2, 0.5],
                [0, 2.25, 0],
                [0, 1, 0.5],
            ],
            point_data=point_data,
        )
        writer.add_cell(
            vtk.vtkQuadraticQuad, indices[[0, 2, 4, 6, 1, 3, 5, 7]], cell_data=cell_data
        )

        # Add tetrahedron.
        cell_data = {}
        cell_data["cell_data_2"] = [5, 0, 10]
        point_data = {}
        point_data["point_data_1"] = [1, 2, 3, 4]
        indices = writer.add_points(
            [[3, 3, 3], [4, 4, 3], [4, 3, 3], [4, 4, 4]], point_data=point_data
        )
        writer.add_cell(vtk.vtkTetra, indices[[0, 2, 1, 3]], cell_data=cell_data)

        # Before we can write the data to file we have to store the cell and
        # point data in the grid
        writer.complete_data()

        # Write to file.
        ref_file = os.path.join(testing_input, "test_meshpy_vtk_writer_reference.vtu")
        vtk_file = os.path.join(testing_temp, "test_meshpy_vtk_writer.vtu")
        writer.write_vtk(vtk_file, binary=False)

        # Compare the vtk files.
        compare_vtk(self, ref_file, vtk_file)

    def test_meshpy_vtk_writer_beam(self):
        """Create a sample mesh and check the VTK output."""

        # Create the mesh.
        mesh = Mesh()

        # Add content to the mesh.
        mat = MaterialBeam(radius=0.05)
        create_beam_mesh_honeycomb(
            mesh, Beam3rHerm2Line3, mat, 2.0, 2, 3, n_el=2, add_sets=True
        )

        # Write VTK output, with coupling sets."""
        ref_file = os.path.join(testing_input, "test_meshpy_vtk_beam_reference.vtu")
        vtk_file = os.path.join(testing_temp, "test_meshpy_vtk_beam.vtu")
        mesh.write_vtk(
            output_name="test_meshpy_vtk",
            coupling_sets=True,
            output_directory=testing_temp,
            binary=False,
        )

        # Compare the vtk files.
        compare_vtk(self, ref_file, vtk_file, tol_float=mpy.eps_pos)

        # Write VTK output, without coupling sets."""
        ref_file = os.path.join(
            testing_input, "test_meshpy_vtk_no_coupling_beam_reference.vtu"
        )
        vtk_file = os.path.join(testing_temp, "test_meshpy_vtk_no_coupling_beam.vtu")
        mesh.write_vtk(
            output_name="test_meshpy_vtk_no_coupling",
            coupling_sets=False,
            output_directory=testing_temp,
            binary=False,
        )

        # Compare the vtk files.
        compare_vtk(self, ref_file, vtk_file, tol_float=mpy.eps_pos)

    def test_meshpy_vtk_writer_solid(self):
        """Import a solid mesh and check the VTK output."""

        # Convert the solid mesh to meshpy objects. Without this parameter no
        # solid VTK file would be written.
        mpy.import_mesh_full = True

        # Create the input file and read solid mesh data.
        input_file = InputFile()
        input_file.read_dat(os.path.join(testing_input, "4C_input_solid_tube.dat"))

        # Write VTK output.
        ref_file = os.path.join(testing_input, "test_meshpy_vtk_solid_reference.vtu")
        vtk_file = os.path.join(testing_temp, "test_meshpy_vtk_solid.vtu")
        if os.path.isfile(vtk_file):
            os.remove(vtk_file)
        input_file.write_vtk(
            output_name="test_meshpy_vtk", output_directory=testing_temp, binary=False
        )

        # Compare the vtk files.
        compare_vtk(self, ref_file, vtk_file)

    def test_meshpy_vtk_writer_solid_elements(self):
        """
        Import a solid mesh with all solid types and check the VTK output.
        """

        # Convert the solid mesh to meshpy objects. Without this parameter no
        # solid VTK file would be written.
        mpy.import_mesh_full = True

        # Create the input file and read solid mesh data.
        input_file = InputFile()
        input_file.read_dat(os.path.join(testing_input, "4C_input_solid_elements.dat"))

        # Write VTK output.
        ref_file = os.path.join(
            testing_input, "test_meshpy_vtk_solid_elements_reference.vtu"
        )
        vtk_file = os.path.join(testing_temp, "test_meshpy_vtk_elements_solid.vtu")
        if os.path.isfile(vtk_file):
            os.remove(vtk_file)
        input_file.write_vtk(
            output_name="test_meshpy_vtk_elements",
            output_directory=testing_temp,
            binary=False,
        )

        # Compare the vtk files.
        compare_vtk(self, ref_file, vtk_file)

    def test_meshpy_vtk_curve_cell_data(self):
        """
        Test that when creating a beam, cell data can be given.
        This test also checks, that the nan values in vtk can be explicitly
        given.
        """

        # Create the mesh.
        mesh = Mesh()
        mpy.vtk_nan_float = 69.69
        mpy.vtk_nan_int = 69

        # Add content to the mesh.
        mat = MaterialBeam(radius=0.05)
        create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 0, 0], [2, 0, 0], n_el=2)
        create_beam_mesh_line(
            mesh,
            Beam3rHerm2Line3,
            mat,
            [0, 1, 0],
            [2, 1, 0],
            n_el=2,
            vtk_cell_data={"cell_data": (1, mpy.vtk_type.int)},
        )
        create_beam_mesh_arc_segment_via_rotation(
            mesh,
            Beam3rHerm2Line3,
            mat,
            [0, 2, 0],
            Rotation([1, 0, 0], np.pi),
            1.5,
            np.pi / 2.0,
            n_el=2,
            vtk_cell_data={"cell_data": (2, mpy.vtk_type.int), "other_data": 69},
        )

        # Write VTK output, with coupling sets."""
        ref_file = os.path.join(
            testing_input, "test_meshpy_vtk_curve_cell_data_reference.vtu"
        )
        vtk_file = os.path.join(
            testing_temp, "test_meshpy_vtk_curve_cell_data_beam.vtu"
        )
        mesh.write_vtk(
            output_name="test_meshpy_vtk_curve_cell_data",
            output_directory=testing_temp,
            binary=False,
        )

        # Compare the vtk files.
        compare_vtk(self, ref_file, vtk_file)

    def test_meshpy_cubitpy_import(self):
        """
        Check that a import from a cubitpy object is the same as importing the
        dat file.
        """

        # Check if cubitpy can be loaded.
        import importlib

        found = importlib.util.find_spec("cubitpy") is not None
        if not found:
            # In this case skip the test.
            skip_fail_test(self, "CubitPy could not be loaded!")

        # Load the mesh creation functions.
        from meshpy_testing.create_cubit_input import create_tube, create_tube_cubit

        # Create the input file and read the file.
        file_path = os.path.join(testing_temp, "test_cubitpy_import.dat")
        create_tube(file_path)
        input_file = InputFile(dat_file=file_path)

        # Create the input file and read the cubit object.
        input_file_cubit = InputFile(cubit=create_tube_cubit())

        # Load the file from the reference folder.
        file_path_ref = os.path.join(testing_input, "4C_input_solid_tube.dat")
        input_file_ref = InputFile(dat_file=file_path_ref)

        # Compare the input files.
        compare_strings(
            self,
            input_file.get_string(header=False),
            input_file_cubit.get_string(header=False),
        )
        compare_strings(
            self,
            input_file.get_string(header=False),
            input_file_ref.get_string(header=False),
            rtol=1e-14,
        )

    def test_meshpy_deep_copy(self):
        """
        Thist test checks that the deep copy function on a mesh does not copy
        the materials or functions.
        """

        # Create material and function object.
        mat = MaterialReissner(youngs_modulus=1, radius=1)
        fun = Function("COMPONENT 0 SYMBOLIC_FUNCTION_OF_SPACE_TIME t")

        def create_mesh(mesh):
            """Add material and function to the mesh and create a beam."""
            mesh.add(fun, mat)
            set1 = create_beam_mesh_line(
                mesh, Beam3rHerm2Line3, mat, [0, 0, 0], [1, 0, 0]
            )
            set2 = create_beam_mesh_line(
                mesh, Beam3rHerm2Line3, mat, [1, 0, 0], [1, 1, 0]
            )
            mesh.add(BoundaryCondition(set1["line"], "fix", bc_type=mpy.bc.dirichlet))
            mesh.add(BoundaryCondition(set2["line"], "load", bc_type=mpy.bc.neumann))
            mesh.couple_nodes()

        # The second mesh will be translated and rotated with those vales.
        translate = [1.0, 2.34535435, 3.345353]
        rotation = Rotation([1, 0.2342342423, -2.234234], np.pi / 15 * 27)

        # First create the mesh twice, move one and get the input file.
        mesh_ref_1 = Mesh()
        mesh_ref_2 = Mesh()
        create_mesh(mesh_ref_1)
        create_mesh(mesh_ref_2)
        mesh_ref_2.rotate(rotation)
        mesh_ref_2.translate(translate)
        input_file_ref = InputFile()
        input_file_ref.add(mesh_ref_1, mesh_ref_2)

        # Now copy the first mesh and add them together in the input file.
        mesh_copy_1 = Mesh()
        create_mesh(mesh_copy_1)
        mesh_copy_2 = mesh_copy_1.copy()
        mesh_copy_2.rotate(rotation)
        mesh_copy_2.translate(translate)
        input_file_copy = InputFile()
        input_file_copy.add(mesh_copy_1, mesh_copy_2)

        # Check that the input files are the same.
        compare_strings(
            self,
            input_file_ref.get_string(
                header=False, dat_header=False, add_script_to_header=False
            ),
            input_file_copy.get_string(
                header=False, dat_header=False, add_script_to_header=False
            ),
        )

    def test_meshpy_mesh_add_checks(self):
        """
        This test checks that Mesh raises an error when double objects are
        added to the mesh.
        """

        # Mesh instance for this test.
        mesh = Mesh()

        # Create basic objects that will be added to the mesh.
        node = Node([0, 1.0, 2.0])
        element = Beam()
        mesh.add(node)
        mesh.add(element)

        # Create objects based on basic mesh items.
        coupling = Coupling(mesh.nodes, mpy.bc.point_coupling, mpy.coupling_dof.fix)
        coupling_penalty = Coupling(
            mesh.nodes, mpy.bc.point_coupling_penalty, mpy.coupling_dof.fix
        )
        geometry_set = GeometrySet(mesh.elements)
        mesh.add(coupling)
        mesh.add(coupling_penalty)
        mesh.add(geometry_set)

        # Add the objects again and check for errors.
        self.assertRaises(ValueError, mesh.add, node)
        self.assertRaises(ValueError, mesh.add, element)
        self.assertRaises(ValueError, mesh.add, coupling)
        self.assertRaises(ValueError, mesh.add, coupling_penalty)
        self.assertRaises(ValueError, mesh.add, geometry_set)

    def test_meshpy_check_two_couplings(self):
        """
        The current implementation can handle more than one coupling on a
        node correctly, therefore we check this here.
        """

        # Create mesh object
        mesh = InputFile()
        mat = MaterialReissner()
        mesh.add(mat)

        # Add two beams to create an elbow structure. The beams each have a
        # node at the intersection
        create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 0, 0], [1, 0, 0])
        create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [1, 0, 0], [1, 1, 0])

        # Call coupling twice -> this will create two coupling objects for the
        # corner node
        mesh.couple_nodes()
        mesh.couple_nodes()

        # Create the input file
        compare_test_result(self, mesh.get_string(header=False))

    def xtest_meshpy_check_multiple_node_penalty_coupling(self, reuse_nodes=True):
        """For point penalty coupling constraints, we add multiple coupling conditions.
        This is checked in this test case. This method creates the flag reuse_nodes
        decides if equal nodes are unified to a single node."""

        # Create mesh object
        mesh = InputFile()
        mat = MaterialReissner()
        mesh.add(mat)

        # Add three beams that have one common point
        create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 0, 0], [1, 0, 0])
        create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [1, 0, 0], [2, 0, 0])
        create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [1, 0, 0], [2, -1, 0])

        mesh.couple_nodes(
            reuse_matching_nodes=reuse_nodes,
            coupling_type=mpy.bc.point_coupling_penalty,
        )
        return mesh

    def test_meshpy_check_multiple_node_penalty_coupling_reuse(self):
        """Test the xtest_meshpy_check_multiple_node_penalty_coupling test and replace
        equal nodes."""

        mesh = self.xtest_meshpy_check_multiple_node_penalty_coupling(reuse_nodes=True)
        compare_test_result(self, mesh.get_string(header=False))

    def test_meshpy_check_multiple_node_penalty_coupling(self):
        """Test the xtest_meshpy_check_multiple_node_penalty_coupling test and do not
        replace equal nodes."""

        mesh = self.xtest_meshpy_check_multiple_node_penalty_coupling(reuse_nodes=False)
        compare_test_result(self, mesh.get_string(header=False))

    def test_meshpy_check_double_elements(self):
        """
        Check if there are overlapping elements in a mesh.
        """

        # Create mesh object.
        mesh = InputFile()
        mat = MaterialReissner()
        mesh.add(mat)

        # Add two beams to create an elbow structure. The beams each have a
        # node at the intersection.
        create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 0, 0], [2, 0, 0], n_el=2)
        create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 0, 0], [1, 0, 0])

        # Rotate the mesh with an arbitrary rotation.
        mesh.rotate(Rotation([1, 2, 3.24313], 2.2323423), [1, 3, -2.23232323])

        # The elements in the created mesh are overlapping, check that an error
        # is thrown.
        self.assertRaises(ValueError, mesh.check_overlapping_elements)

        # Check if the overlapping elements are written to the vtk output.
        warnings.filterwarnings("ignore")
        ref_file = os.path.join(
            testing_input, "test_meshpy_vtk_element_overlap_reference.vtu"
        )
        vtk_file = os.path.join(
            testing_temp, "test_meshpy_vtk_element_overlap_beam.vtu"
        )
        mesh.write_vtk(
            output_name="test_meshpy_vtk_element_overlap",
            output_directory=testing_temp,
            binary=False,
            overlapping_elements=True,
        )

        # Compare the vtk files.
        compare_vtk(self, ref_file, vtk_file)

    def perform_test_check_overlapping_coupling_nodes(self, check=True):
        """
        Per default, we check that coupling nodes are at the same physical
        position. This check can be deactivated with the keyword
        check_overlapping_nodes when creating a Coupling.
        """

        # Create mesh object.
        mesh = InputFile()
        mat = MaterialReissner()
        mesh.add(mat)

        # Add two beams to create an elbow structure. The beams each have a
        # node at the intersection.
        set_1 = create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 0, 0], [1, 0, 0])
        set_2 = create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [2, 0, 0], [3, 0, 0])

        # Couple two nodes that are not at the same position.

        # Create the input file. This will cause an error, as there are two
        # couplings for one node.
        args = [
            [set_1["start"], set_2["end"]],
            mpy.bc.point_coupling,
            "coupling_type_string",
        ]
        if check:
            self.assertRaises(ValueError, Coupling, *args)
        else:
            Coupling(*args, check_overlapping_nodes=False)

    def test_meshpy_check_overlapping_coupling_nodes(self):
        """
        Perform the test that the coupling nodes can be tested if they are at
        the same position.
        """
        self.perform_test_check_overlapping_coupling_nodes(True)
        self.perform_test_check_overlapping_coupling_nodes(False)

    def test_meshpy_check_start_end_node_error(self):
        """
        Check that an error is raised if wrong start and end nodes are given to a mesh
        creation function.
        """

        # Create mesh object.
        mesh = Mesh()
        mat = MaterialReissner()
        mesh.add(mat)

        # Try to create a line with a starting node that is not in the mesh.
        node = NodeCosserat([0, 0, 0], Rotation())
        args = [mesh, Beam3rHerm2Line3, mat, [0, 0, 0], [1, 0, 0]]
        kwargs = {"start_node": node}
        self.assertRaises(ValueError, create_beam_mesh_line, *args, **kwargs)
        node.coordinates = [1, 0, 0]
        kwargs = {"end_node": node}
        self.assertRaises(ValueError, create_beam_mesh_line, *args, **kwargs)

    def test_meshpy_userdefined_boundary_condition(self):
        """
        Check if an user defined boundary condition can be added.
        """

        mesh = InputFile()

        mat = MaterialReissner()
        sets = create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 0, 0], [1, 2, 3])
        mesh.add(BoundaryCondition(sets["line"], "test", bc_type="USER SECTION FOR BC"))

        # Compare the output of the mesh.
        compare_test_result(self, mesh.get_string(header=False).strip())

    def test_meshpy_display_pyvista(self):
        """Test that the display in pyvista function does not lead to errors

        TODO: Add a check for the created visualziation"""

        mpy.import_mesh_full = True
        mesh = self.create_beam_to_solid_conditions_model()
        _plotter = mesh.display_pyvista(is_testing=True, resolution=3)


if __name__ == "__main__":
    # Execution part of script.
    unittest.main()
