# -*- coding: utf-8 -*-
"""
Define utility functions for the testing process.
"""

# Python imports.
import os
import shutil
import subprocess
import warnings


# Global variable if this test is run by GitLab.
if ('TESTING_GITLAB' in os.environ.keys()
        and os.environ['TESTING_GITLAB'] == '1'):
    TESTING_GITLAB = True
else:
    TESTING_GITLAB = False


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

# Check and clean the temp directory.
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

        # Check if temp directory exists, and creates it if necessary.
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
            ['kompare', reference_file, compare_file],
            stderr=subprocess.PIPE)
        child.communicate()

    # Check the results.
    self.assertTrue(is_equal, name)
