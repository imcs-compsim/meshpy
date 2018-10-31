# -*- coding: utf-8 -*-
"""
Create a couple of different mesh cases and test the performance.
"""


# Python imports.
import time
import numpy as np
import os
import socket
import warnings

# Meshpy imports.
from meshpy import mpy, InputFile, Mesh, MaterialReissner, Beam3rHerm2Lin3, \
    Rotation

from meshpy.mesh_creation_functions.beam_basic_geometry import \
    create_beam_mesh_line

# Cubitpy imports.
from cubitpy import cupy, CubitPy, get_methods


# Directories and files for testing.
testing_path = os.path.abspath(os.path.dirname(__file__))
testing_temp = os.path.join(testing_path, 'testing-tmp')
testing_solid_block = os.path.join(testing_temp,
    'parformancne_testing_solid.dat')
testing_beam = os.path.join(testing_temp, 'parformancne_testing_beam.dat')


def create_solid_block(file_path, nx, ny, nz):
    """
    Create a solid block (1 x 1 x 1) with (nx * ny * nz) elements.
    """

    # Initialize cubit.
    cubit = CubitPy()

    # Create brick.
    brick = cubit.brick(1)

    # Set mesh parameters.
    mesh_size = [
        [nx, [2, 4, 6, 8]],
        [ny, [1, 3, 5, 7]],
        [nz, [9, 10, 11, 12]]
        ]
    for [n_el, curves] in mesh_size:
        for i in curves:
            cubit.set_line_interval(brick.curves()[i - 1], n_el)
    brick.volumes()[0].mesh()

    # Add block and sets.
    cubit.add_element_type(brick.volumes()[0], 'HEX8', name='brick', bc=[
        'STRUCTURE',
        'MAT 1 KINEM nonlinear EAS none',
        'SOLIDH8'
        ])
    counter = 0
    for item in brick.vertices():
        cubit.add_node_set(item, name='node_set_' + str(counter), bc=[
            'DESIGN POINT NEUMANN CONDITIONS',
            'NUMDOF 6 ONOFF 1 1 1 0 0 0 VAL 3.0 3.0 0.0 0.0 0.0 0.0 FUNCT 1 2 0 0 0 0'])
        counter += 1
    for item in brick.curves():
        cubit.add_node_set(item, name='node_set_' + str(counter), bc=[
            'DESIGN LINE DIRICH CONDITIONS',
            'NUMDOF 6 ONOFF 1 1 1 0 0 0 VAL 3.0 3.0 0.0 0.0 0.0 0.0 FUNCT 1 2 0 0 0 0'])
        counter += 1
    for item in brick.surfaces():
        cubit.add_node_set(item, name='node_set_' + str(counter), bc=[
            'DESIGN SURF NEUMANN CONDITIONS',
            'NUMDOF 6 ONOFF 1 1 1 0 0 0 VAL 3.0 3.0 0.0 0.0 0.0 0.0 FUNCT 1 2 0 0 0 0'])
        counter += 1

    # Export mesh
    cubit.create_dat(file_path)


def load_solid(solid_file, full_import):
    """
    Load a solid into an input file.
    """

    mpy.set_default_values()
    mpy.import_mesh_full = full_import
    InputFile(dat_file=solid_file)


def create_large_beam_mesh(nx, ny, nz, n_el):
    """
    Create a beam grid on the domain (1 x 1 x 1) with (nx * ny * nz)
    "grid cells".
    """

    mesh = InputFile()
    material = MaterialReissner(radius=0.25 / np.max([nx, ny, nz]))

    for ix in range(nx + 1):
        for iy in range(ny + 1):
            create_beam_mesh_line(mesh, Beam3rHerm2Lin3, material,
                [ix / nx, iy / ny, 0],
                [ix / nx, iy / ny, 1],
                n_el=nz * n_el)
    for iy in range(ny + 1):
        for iz in range(nz + 1):
            create_beam_mesh_line(mesh, Beam3rHerm2Lin3, material,
                [0, iy / ny, iz / nz],
                [1, iy / ny, iz / nz],
                n_el=nx * n_el)
    for iz in range(nz + 1):
        for ix in range(nx + 1):
            create_beam_mesh_line(mesh, Beam3rHerm2Lin3, material,
                [ix / nx, 0, iz / nz],
                [ix / nx, 1, iz / nz],
                n_el=ny * n_el)
    return mesh


def time_function(name, funct, args=None, kwargs=None):
    """
    Execute a function and check if the time is as expected.
    """

    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    # Get the expected time for this function.
    host = socket.gethostname()
    if host in expected_times.keys():
        if name in expected_times[host].keys():
            expected_time = expected_times[host][name]
        else:
            raise ValueError('Function name {} not found!'.format(name))
    else:
        raise ValueError('Host {} not found!'.format(host))

    # Time before the execution.
    start_time = time.time()

    # Execute the function.
    return_val = funct(*args, **kwargs)

    # Check the elapsed time.
    elapsed_time = time.time() - start_time
    print(
        'Times for {}:\n'.format(name)
        + '    Expected: {:.3g}sec\n'.format(expected_time)
        + '    Actual:   {:.3g}sec'.format(elapsed_time)
        )
    if expected_time > elapsed_time:
        print('    OK')
    else:
        warnings.warn('Expected time not reached in '
            + 'function {}!'.format(name))

    # Return what the function would have given.
    return return_val


if __name__ == '__main__':
    """Execute part of script."""

    expected_times = {}
    expected_times['adonis'] = {
        'cubitpy_create_solid': 3.2,
        'meshpy_load_solid': 1.1,
        'meshpy_load_solid_full': 3.5,
        'meshpy_create_beams': 7.2,
        'meshpy_rotate': 0.6,
        'meshpy_translate': 0.6,
        'meshpy_reflect': 0.7,
        'meshpy_wrap_around_cylinder': 0.9,
        'meshpy_get_close_nodes': 1.1,
        'meshpy_write_dat': 12.5,
        'meshpy_write_vtk': 19
        }

    time_function(
        'cubitpy_create_solid',
        create_solid_block,
        args=[testing_solid_block, 100, 100, 10]
        )

    time_function(
        'meshpy_load_solid',
        load_solid,
        args=[testing_solid_block, False]
        )

    time_function(
        'meshpy_load_solid_full',
        load_solid,
        args=[testing_solid_block, True]
        )

    mesh = time_function(
        'meshpy_create_beams',
        create_large_beam_mesh,
        args=[40, 40, 10, 2]
        )

    time_function(
        'meshpy_rotate',
        Mesh.rotate,
        args=[mesh, Rotation([1, 1, 0], np.pi / 3)]
        )

    time_function(
        'meshpy_translate',
        Mesh.translate,
        args=[mesh, [0.5, 0, 0]]
        )

    time_function(
        'meshpy_reflect',
        Mesh.reflect,
        args=[mesh, [0.5, 0.4, 0.1]]
        )

    time_function(
        'meshpy_wrap_around_cylinder',
        Mesh.wrap_around_cylinder,
        args=[mesh],
        kwargs={'radius': 1.}
        )

    time_function(
        'meshpy_get_close_nodes',
        Mesh.get_close_nodes,
        args=[mesh, mesh.nodes]
        )

    time_function(
        'meshpy_write_dat',
        InputFile.write_input_file,
        args=[mesh, testing_beam]
        )

    time_function(
        'meshpy_write_vtk',
        Mesh.write_vtk,
        args=[mesh],
        kwargs={
            'output_name': 'performance_testing_beam',
            'output_directory': testing_temp
            }
        )
