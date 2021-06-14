# -*- coding: utf-8 -*-
"""
Define utility functions for the testing process.
"""

# Python imports.
import os
import numpy as np
import shutil
import subprocess
import warnings
import xml.etree.ElementTree as ET
import vtk
from vtk.util import numpy_support as vtk_numpy


# Global variable if this test is run by GitLab.
if ('TESTING_GITLAB' in os.environ.keys()
        and os.environ['TESTING_GITLAB'] == '1'):
    TESTING_GITLAB = True
else:
    TESTING_GITLAB = False


def skip_fail_test(self, message):
    """
    Skip or fail the test depending if the test are run in GitLab or not.
    """
    if TESTING_GITLAB:
        self.skipTest(message)
    else:
        self.skipTest(message)


def get_baci_path():
    """Look for and return a path to baci-release."""

    if 'BACI_RELEASE' in os.environ.keys():
        path = os.environ['BACI_RELEASE']
    else:
        path = ''

    # Check if the path exists.
    if os.path.isfile(path):
        return path
    else:
        # In the case that no path was found, check if the script is performed
        # by a GitLab runner.
        if TESTING_GITLAB:
            raise ValueError('Path to baci-release not found!')
        else:
            warnings.warn('Path to baci-release not found. Did you set the ' +
                'environment variable BACI_RELEASE?')
            return None


# Define the testing paths.
testing_path = os.path.abspath(os.path.dirname(__file__))
testing_input = os.path.join(testing_path, 'reference-files')
testing_temp = os.path.join(testing_path, 'testing-tmp')
baci_release = get_baci_path()

# Check and clean the temporary directory.
os.makedirs(testing_temp, exist_ok=True)


def empty_testing_directory():
    """Delete all files in the testing directory, if it exists."""
    if os.path.isdir(testing_temp):
        for the_file in os.listdir(testing_temp):
            file_path = os.path.join(testing_temp, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(e)


def compare_strings(self, name, reference, compare):
    """
    Compare two stings. If they are not identical open kompare and show
    differences.
    """

    # Check if the input data is a file that exists.
    reference_is_file = os.path.isfile(reference)
    compare_is_file = os.path.isfile(compare)

    # Get the correct data
    if reference_is_file:
        with open(reference, 'r') as myfile:
            reference_string = myfile.read()
    else:
        reference_string = reference

    if compare_is_file:
        with open(compare, 'r') as myfile:
            compare_string = myfile.read()
    else:
        compare_string = compare

    # Check if the strings are equal, if not compare the differences and
    # fail the test.
    is_equal = reference_string.strip() == compare_string.strip()
    if not is_equal and not TESTING_GITLAB:

        # Check if temporary directory exists, and creates it if necessary.
        os.makedirs(testing_temp, exist_ok=True)

        # Get the paths of the files to compare. If a string was given
        # create a file with the string in it.
        if reference_is_file:
            reference_file = reference
        else:
            reference_file = os.path.join(testing_temp,
                '{}_reference.dat'.format(name))
            with open(reference_file, 'w') as input_file:
                input_file.write(reference_string)

        if compare_is_file:
            compare_file = compare
        else:
            compare_file = os.path.join(testing_temp,
                '{}_compare.dat'.format(name))
            with open(compare_file, 'w') as input_file:
                input_file.write(compare_string)

        child = subprocess.Popen(
            ['meld', reference_file, compare_file],
            stderr=subprocess.PIPE)
        child.communicate()

    # Check the results.
    self.assertTrue(is_equal, name)


def xml_to_dict(xml, tol_float):
    """Convert a XML to a nested dictionary."""

    def item_with_tol(item):
        """
        Check if item is a number and if so, format it to match the given
        tolerance.
        """

        if tol_float is None:
            return item
        else:
            try:
                number = float(item)
                if np.abs(float(number)) < tol_float:
                    return '0.0'
                else:
                    # We check the numbers to a precision of 13.
                    number_float = float(number)
                    return '{:.13e}'.format(number_float)
            except ValueError:
                return item

    # Get and sort keys.
    keys = xml.keys()
    keys.sort()

    # Get string for this XML element.
    string = '<' + xml.tag
    if 'Name' in keys:
        # If there is a key "Name" put this one first.
        index = keys.index('Name')
        if index == 0:
            pass
        else:
            keys[0], keys[index] = keys[index], keys[0]
    for key in keys:
        string += ' '
        string += key
        string += '="'
        string += item_with_tol(xml.get(key))
        string += '"'
    string += '>'

    # Get data for this item.
    xml_dict = {}
    n_childs = len(xml.getchildren())
    is_text = not xml.text.strip() == ''
    if n_childs > 0:
        # Add a child xml construct.
        for child in xml.getchildren():
            key, value = xml_to_dict(child, tol_float)
            xml_dict[key] = value

    if is_text:
        # Add data.
        data = xml.text.split('\n')
        if tol_float is None:
            data_new = [line.strip() for line in data
                if not line.strip() == '']
        else:
            data_new = []
            for line in data:
                if line.strip() == '':
                    continue
                for number in line.strip().split(' '):
                    data_new.append(item_with_tol(number))
        data_string = '\n'.join(data_new)
        xml_dict[''] = data_string

    # Return key for this item and all child items.
    return string, xml_dict


def xml_dict_to_string(item):
    """The nested XML dictionary to a string."""

    # Sort the keys.
    keys = list(item.keys())
    keys.sort()

    # Return the keys and the values.
    string = ''
    for key in keys:
        if key == '':
            string += item[key]
        else:
            # Add content.
            string += key
            string += '\n'
            string += xml_dict_to_string(item[key])
            string += '\n'

            # Get the name of the section from the key.
            section = key[1:].split(' ')[0]
            if section[-1] == '>':
                section = section[:-1]
            string += '</{}>\n'.format(section)

    # Return the value.
    return string.strip()


def compare_xml(path1, path2, tol_float=None):
    """
    Compare the xml files at path1 and path2, by checking the xml data of the
    files.

    Args
    ----
    tol_float: None / float
        If it is None, the numbers are not changed.
        If it is a number, the numbers in the xml file are set to 0 when the
        absolute value is smaller that tol_float.
    """

    # Check that both arguments are paths and exist.
    if not (os.path.isfile(path1) and os.path.isfile(path2)):
        raise ValueError('The paths given are not OK!')

    tree1 = ET.parse(path1)
    tree2 = ET.parse(path2)

    key, value = xml_to_dict(tree1.getroot(), tol_float)
    string1 = xml_dict_to_string({key: value})
    hash1 = hash(string1)

    key, value = xml_to_dict(tree2.getroot(), tol_float)
    string2 = xml_dict_to_string({key: value})
    hash2 = hash(string2)

    if hash1 == hash2:
        return True, None, None
    else:
        return False, string1, string2


def compare_vtk_data(path1, path2, *, raise_error=False, tol_float=None):
    """
    Compare the vtk files at path1 and path2, by compairing the stored data.

    Args
    ----
    raise_error: bool
        If true, then an error will be raised in case the files do not match.
        Otherwise False will be returned.
    tol_float: None / float
        If given, numbers will be considered equal if the difference between
        them is smaller than tol_float.
    """

    # Check that both arguments are paths and exist.
    if not (os.path.isfile(path1) and os.path.isfile(path2)):
        raise ValueError('The paths given are not OK!')

    # Default value for the numerical tolerance.
    if tol_float is None:
        tol_float = 1e-16

    def get_vtk(path):
        """
        Return a vtk object for the file at path.
        """
        reader = vtk.vtkXMLGenericDataObjectReader()
        reader.SetFileName(path)
        reader.Update()
        return reader.GetOutput()

    def compare_arrays(array1, array2, name=None):
        """
        Compare two vtk arrays.
        """

        diff = vtk_numpy.vtk_to_numpy(array1) - vtk_numpy.vtk_to_numpy(array2)
        if not np.max(np.abs(diff)) < tol_float:
            error_string = 'VTK array comparison failed!'
            if name is not None:
                error_string += ' Name of the array: {}'.format(name)
            raise ValueError(error_string)

    def compare_data_sets(data1, data2):
        """
        Compare data sets obtained from vtk objects.
        """

        # Both data sets need to have the same number of arrays.
        if not data1.GetNumberOfArrays() == data2.GetNumberOfArrays():
            raise ValueError('Length of vtk data objects do not match!')

        # Compare each array.
        for i in range(data1.GetNumberOfArrays()):

            # Get the arrays with the same name.
            name = data1.GetArrayName(i)
            array1 = data1.GetArray(name)
            array2 = data2.GetArray(name)
            compare_arrays(array1, array2, name=name)

    # Perform all checks, catch errors.
    try:
        # Load the vtk files.
        data1 = get_vtk(path1)
        data2 = get_vtk(path2)

        # Compare the point positions.
        compare_arrays(
            data1.GetPoints().GetData(),
            data2.GetPoints().GetData(),
            name='point_positions'
            )

        # Compare the cell and point data of the array.
        compare_data_sets(data1.GetCellData(), data2.GetCellData())
        compare_data_sets(data1.GetPointData(), data2.GetPointData())

        # Compare the cell connectivity.
        compare_arrays(
            data1.GetCells().GetData(),
            data2.GetCells().GetData(),
            name='cell_connectivity')

        # Compare the cell types.
        compare_arrays(
            data1.GetCellTypesArray(),
            data2.GetCellTypesArray(),
            name='cell_type')

    except Exception as error:
        if raise_error:
            raise error
        return False

    return True


def compare_vtk(self, name, ref_file, vtk_file, tol_float=None):
    """
    Compare two vtk files. Two different methods are used. Once the xml
    structure is compared and once the vtk data is compared.
    """

    # Compare the xml structure.
    is_equal_xml, _string_ref, _string_vtk = compare_xml(ref_file,
        vtk_file, tol_float=tol_float)

    # Compare the raw data.
    is_equal_data = compare_vtk_data(ref_file, vtk_file, tol_float=tol_float)

    # Check that both comparisons yield true.
    error_string = name + ('\ncompare results: is_equal_xml={}, '
        + 'is_equal_data={}').format(is_equal_xml, is_equal_data)
    self.assertTrue(is_equal_xml and is_equal_data, error_string)
