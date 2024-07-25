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
Test utilities of MeshPy.
"""

# Python imports
import unittest
import numpy as np
import numpy.testing as npt
# Meshpy imports
from meshpy.utility import (is_node_on_plane,linear_transformations)

from meshpy.node import Node


class TestUtilities(unittest.TestCase):
    """Test utilities from the meshpy.utility module."""

    def test_is_node_on_plane(self):
        """Test if node on plane function works properly."""

        # node on plane with origin_distance
        node = Node([1.0, 1.0, 1.0])
        self.assertTrue(
            is_node_on_plane(node, normal=[0.0, 0.0, 1.0], origin_distance=1.0)
        )

        # node on plane with point_on_plane
        node = Node([1.0, 1.0, 1.0])
        self.assertTrue(
            is_node_on_plane(
                node, normal=[0.0, 0.0, 5.0], point_on_plane=[5.0, 5.0, 1.0]
            )
        )

        # node not on plane with origin_distance
        node = Node([13.5, 14.5, 15.5])
        self.assertFalse(
            is_node_on_plane(node, normal=[0.0, 0.0, 1.0], origin_distance=5.0)
        )

        # node not on plane with point_on_plane
        node = Node([13.5, 14.5, 15.5])
        self.assertFalse(
            is_node_on_plane(
                node, normal=[0.0, 0.0, 5.0], point_on_plane=[5.0, 5.0, 1.0]
            )
        )


    def test_transformations_scaling(self):
        """
        test the scaling between the intvervall of the function
        it starts with a funciton in the intervall between [0,1] and transforms them
        """


        # starting time array
        time=np.array([0,0.5,0.75,1.0])

        # corresponding values 3 values per time step
        force=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])

        # end time point
        t_end=100;

        # first result is simply the attached point at the end
        time_result=np.append(time,t_end)

        # with the value vector containing the last entry twice
        force_result=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[10,11,12]])

        # base case no scaling only end points should be attached
        time_trans,force_trans=linear_transformations(time,force,[0,1,t_end],False)

        # check solution
        self.assertEqual(time_result.tolist(),time_trans.tolist())
        self.assertEqual(force_trans.tolist(),force_result.tolist())

        # transform back to intervall [0, 1]
        time_trans,force_trans=linear_transformations(2*time,force,[0,1,100],False)

        # result is again the same, since it is transformed back to the initial state
        self.assertEqual(time_result.tolist(),time_trans.tolist())
        self.assertEqual(force_trans.tolist(),force_result.tolist())

        # new force result
        force_result=np.array([[1,2,3],[1,2,3],[4,5,6],[7,8,9],[10,11,12],[10,11,12]])

        # test now an shift to the intervall [1 ,2]
        time_trans,force_trans=linear_transformations(time,force,[1,2,100],False)
        self.assertEqual(np.append(np.append(0,1+time),t_end).tolist(),time_trans.tolist())
        self.assertEqual(force_trans.tolist(),force_result.tolist())

        # test offset and scaling
        # scales the time function in from 0.5 back to the intervall [2,3]
        time_trans,force_trans=linear_transformations(0.5*time,force,[2,3,100],False)
        self.assertEqual(np.append(np.append(0,2+time),t_end).tolist(),time_trans.tolist())
        self.assertEqual(force_trans.tolist(),force_result.tolist())

    def test_transformations_flip(self):
        """
        test the flip flag to mirror the function
        :return:
        """

        # base case no scaling only end points should be attached
        # starting time array
        time=np.array([0,0.5,0.75,1.0])

        # corresponding values:  3 values per time step
        force=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])

        # end time point
        t_end=100;

        # first result is simply the attached point at the end
        time_result=np.array([0,0.25,0.5,1.0,t_end])

        # with the value vector containing the last entry twice
        force_result=np.array([[10,11,12],[7,8,9],[4,5,6],[1,2,3],[1,2,3]])

        # base case no scaling only end points should be attached
        time_trans,force_trans=linear_transformations(time,force,[0,1,t_end],True)

        # check solution
        self.assertEqual(time_result.tolist(),time_trans.tolist())
        self.assertEqual(force_trans.tolist(),force_result.tolist())

        # transform back to intervall [0, 1]
        time_trans,force_trans=linear_transformations(2*time,force,[0,1,100],True)

        # result is again the same, since it is transformed back to the initial state
        self.assertEqual(time_result.tolist(),time_trans.tolist())
        self.assertEqual(force_trans.tolist(),force_result.tolist())

        # new force result
        force_result=np.array([[10,11,12],[10,11,12],[7,8,9], [4,5,6],[1,2,3],[1,2,3],])

        # subtract t_end-1 so that we can later add simply +1 to obtain the desired value
        time_result=np.array([0,0.25,0.5,1.0,t_end-1])
        # test now an shift to the intervall [1 ,2]
        time_trans,force_trans=linear_transformations(time,force,[1,2,100],True)
        self.assertEqual(np.append(0,1+time_result).tolist(),time_trans.tolist())
        self.assertEqual(force_trans.tolist(),force_result.tolist())

        # same trick as above but with 2
        time_result=np.array([0,0.25,0.5,1.0,t_end-2])

        # test offset and scaling
        # scales the time function in from 0.5 back to the intervall [2,3]
        time_trans,force_trans=linear_transformations(0.5*time,force,[2,3,100],True)
        self.assertEqual(np.append(0,2+time_result).tolist(),time_trans.tolist())
        self.assertEqual(force_trans.tolist(),force_result.tolist())


if __name__ == "__main__":
    # Execution part of script.
    unittest.main()
