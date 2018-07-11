# -*- coding: utf-8 -*-
"""
This module provides a class that is used to write VTK files.
"""

# Python modules.
import vtk
import numpy as np
import numbers
import os
import warnings

# Meshpy modules.
from . import mpy


class VTKWriter(object):
    """A class that manages VTK cells and data and can also create them."""

    def __init__(self):

        # Initialize VTK objects.
        self.points = vtk.vtkPoints()
        self.grid = vtk.vtkUnstructuredGrid()

        # Link points to grid.
        self.grid.SetPoints(self.points)

        # Container for output data.
        self.data = {}
        for key1 in mpy.vtk_geom_types:
            for key2 in mpy.vtk_data_types:
                self.data[key1, key2] = {}

    def add_poly_line(self, coordinates, **kwargs):
        """Add a poly line. The line will connect the points in coordinates."""

        # Connectivity of points in cell.
        topology = list(range(len(coordinates)))

        # Add cell.
        self._add_cell(vtk.vtkPolyLine, coordinates, topology, **kwargs)

    def _add_cell(self, cell_type, coordinates, topology, cell_data=None,
            point_data=None):
        """
        Create a cell and add it to the global array.

        Args
        ----
        cell_type: VTK_type
            Type of cell that will be created.
        coordinates: [3d vector]
            Coordinated of points for this cell.
        topology: [int]
            The connectivity between the cell and the coordinates.
        cell_data, point_data: dic
            A dictionary containing data that will be added to this cell,
            either as cell data, or point data for each point of the cell.
            There are some checks in place, but the length of the data should
            be chosen carefully. If some cells do not have a filed, that field
            will be set to 0 for this cell / points.
        """

        # Consistency checks.
        # Size of coordinates and topology.
        n_points = len(coordinates)
        if not n_points == len(topology):
            raise ValueError(('Coordinates is of size {}, '
                + 'while topology is of size {}!').format(
                    n_points,
                    len(topology)
                    ))

        # Check if point data containers are of the correct size.
        if point_data is not None:
            for key, value in point_data.items():
                if not len(value) == n_points:
                    raise IndexError(('The len of coordinates is {}, the len '
                        + 'of {} is {}, does not match!').format(
                            n_points,
                            key,
                            len(value)
                            ))

        # Check if data container already exists. If not, add it and also add
        # previous entries.
        for data_container, vtk_geom_type in [
                (input_data, vtk_geom_type) for (input_data, vtk_geom_type)
                in [(cell_data, mpy.vtk_cell), (point_data, mpy.vtk_point)]
                if input_data is not None]:

            # Loop through output fields.
            for key, value in data_container.items():

                # Data type.
                if vtk_geom_type == mpy.vtk_cell:
                    vtk_data_type = self._get_vtk_data_type(value)
                else:
                    for item in value:
                        vtk_data_type = self._get_vtk_data_type(item)

                # Check if key already exists.
                if key not in self.data[vtk_geom_type, vtk_data_type].keys():

                    # Set up the VTK data array.
                    data = vtk.vtkDoubleArray()
                    data.SetName(key)
                    if vtk_data_type == mpy.vtk_scalar:
                        data.SetNumberOfComponents(1)
                    else:
                        data.SetNumberOfComponents(3)

                    # Add the empty values for all previous cells / points.
                    if vtk_geom_type == mpy.vtk_cell:
                        n_items = self.grid.GetNumberOfCells()
                    else:
                        n_items = self.grid.GetNumberOfPoints()
                    for i in range(n_items):
                        self._add_empty_data(data, vtk_data_type)
                    self.data[vtk_geom_type, vtk_data_type][key] = data

        # Create the cell.
        geometry_item = cell_type()
        geometry_item.GetPointIds().SetNumberOfIds(n_points)

        # Create the connection between the coordinates.
        for i, coord in enumerate(coordinates):

            # Add the coordinate to the global list of coordinates.
            self.points.InsertNextPoint(coord[0], coord[1], coord[2])

            # Set the local connectivity.
            geometry_item.GetPointIds().SetId(
                i,
                self.points.GetNumberOfPoints() - 1
                )

        # Add to global cells.
        self.grid.InsertNextCell(
            geometry_item.GetCellType(),
            geometry_item.GetPointIds()
            )

        # Add to global data. Loop over data items and check if there is
        # something to be added in this cell. If not an empty value is added.
        for [key_geom, key_data], data in self.data.items():

            # Get input data container.
            if key_geom == mpy.vtk_cell:
                data_container = cell_data
            else:
                data_container = point_data
            if data_container is None:
                continue

            for key, value in data.items():

                # Check if an existing field is also given for this function.
                if key in data_container.keys():
                    # Add the given data.
                    if key_geom == mpy.vtk_cell:
                        self._add_data(value, key_data,
                            non_zero_data=data_container[key])
                    else:
                        for item in data_container[key]:
                            self._add_data(value, key_data, non_zero_data=item)
                else:
                    # Add empty data.
                    if key_geom == mpy.vtk_cell:
                        self._add_data(value, key_data)
                    else:
                        for item in data_container[key]:
                            self._add_data(value, key_data)

    def _get_vtk_data_type(self, data):
        """Return the type of data. Check if data matches an expected case."""

        if isinstance(data, list) or isinstance(data, np.ndarray):
            if len(data) == 3:
                return mpy.vtk_vector
            else:
                raise IndexError('Only 3d vectors are implemented yet! Got '
                    + 'len(data) = {}'.format(len(data)))
        elif isinstance(data, numbers.Number):
            return mpy.vtk_scalar

        raise ValueError('Data {} did not match any expected case!'.format(
            data))

    def _add_data(self, data, vtk_data_type, non_zero_data=None):
        """Add data to a VTK data array."""
        if vtk_data_type == mpy.vtk_scalar:
            if non_zero_data is None:
                data.InsertNextTuple1(0)
            else:
                data.InsertNextTuple1(non_zero_data)
        else:
            if non_zero_data is None:
                data.InsertNextTuple3(0, 0, 0)
            else:
                data.InsertNextTuple3(non_zero_data[0], non_zero_data[1],
                    non_zero_data[2])

    def write_vtk(self, filepath, ascii=False):
        """
        Write the VTK geometry and data to a file.

        Args
        ----
        filepath: str
            Path to output file. The file extension should be vtu.
        ascii: bool
            If the data should be compressed or written in human readable text.
        """

        # Add data to grid.
        for (key_geom, key_data), value in self.data.items():
            for vtk_data in value.values():
                if key_geom == mpy.vtk_cell:
                    self.grid.GetCellData().AddArray(vtk_data)
                else:
                    self.grid.GetPointData().AddArray(vtk_data)

        # Initialize VTK writer.
        writer = vtk.vtkXMLUnstructuredGridWriter()

        # Set the ascii flag.
        if ascii:
            writer.SetDataModeToAscii()

        # Check the file extension.
        filename, file_extension = os.path.splitext(filepath)
        if not file_extension.lower() == '.vtu':
            warnings.warn('The extension should be "vtu", got {}!'.format(
                file_extension))

        # Write geometry and data to file.
        writer.SetFileName(filepath)
        writer.SetInputData(self.grid)
        writer.Write()
