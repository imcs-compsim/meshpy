# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# MeshPy: A beam finite element input generator.
#
# Copyright (c) 2021 Ivo Steinbrecher
#                    Institute for Mathematics and Computer-Based Simulation
#                    Universitaet der Bundeswehr Muenchen
#                    https://www.unibw.de/imcs-en
#
# TODO: Add license.
# -----------------------------------------------------------------------------
"""
This script is used to test that all headers in the repository are correct.
"""

import unittest
from utility import check_license


class TestHeaders(unittest.TestCase):
    """This class tests the headers in the repository."""

    def test_headers(self):
        """
        Check if all headers are correct.
        """

        wrong_headers = check_license()
        wrong_headers_string = 'Wrong headers in: ' + ', '.join(wrong_headers)
        self.assertTrue(len(wrong_headers) == 0, wrong_headers_string)


if __name__ == '__main__':
    # Execution part of script.
    unittest.main()
