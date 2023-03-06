// -----------------------------------------------------------------------------
// MeshPy: A beam finite element input generator
//
// MIT License
//
// Copyright (c) 2021 Ivo Steinbrecher
//                    Institute for Mathematics and Computer-Based Simulation
//                    Universitaet der Bundeswehr Muenchen
//                    https://www.unibw.de/imcs-en
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// -----------------------------------------------------------------------------

#include <ArborX.hpp>
#include <type_traits>

#include "find_close_points.h"

namespace GeometricSearch
{
    /*
     * Define the types used for the data in this file.
     * The second entry in the closest_point_type pair is the radius for the cluster search.
     * TODO: Template this also on the scalar type of the numpy array.
     */
    template <int n_dim>
    struct CoordinatesClosestPoints
    {
        const pybind11::detail::unchecked_reference<double, 2>* coordinates_;
        const double tol_;
    };
}  // namespace GeometricSearch

/*
 * Define how ArborX accesses the point data provided by the find_close_points function.
 */
namespace ArborX
{
    template <int n_dim>
    struct AccessTraits<GeometricSearch::CoordinatesClosestPoints<n_dim>, PrimitivesTag>
    {
        using memory_space = Kokkos::HostSpace;

        static std::size_t size(const GeometricSearch::CoordinatesClosestPoints<n_dim>& coordinates)
        {
            return coordinates.coordinates_->shape(0);
        }

        static auto get(
            const GeometricSearch::CoordinatesClosestPoints<n_dim>& coordinates, std::size_t i)
        {
            // Return an ArborX point representing the point from the numpy array.
            // TODO: Use a hyper point for arrays with other than 3 dimensions.
            return ArborX::Point{(float)coordinates.coordinates_->operator()(i, 0),
                (float)coordinates.coordinates_->operator()(i, 1),
                (float)coordinates.coordinates_->operator()(i, 2)};
        }
    };

    template <int n_dim>
    struct AccessTraits<GeometricSearch::CoordinatesClosestPoints<n_dim>, PredicatesTag>
    {
        using memory_space = Kokkos::HostSpace;

        static std::size_t size(const GeometricSearch::CoordinatesClosestPoints<n_dim>& coordinates)
        {
            return coordinates.coordinates_->shape(0);
        }

        static auto get(
            const GeometricSearch::CoordinatesClosestPoints<n_dim>& coordinates, std::size_t i)
        {
            // Create an ArborX sphere at the coordinates of the point here. The radius is the
            // tolerance for the closes point search.
            // TODO: Use a hyper sphere for arrays with other than 3 dimensions.
            return attach(
                intersects(ArborX::Sphere{{(float)coordinates.coordinates_->operator()(i, 0),
                                              (float)coordinates.coordinates_->operator()(i, 1),
                                              (float)coordinates.coordinates_->operator()(i, 2)},
                    (float)coordinates.tol_}),
                (int)i);
        }
    };
}  // namespace ArborX


namespace GeometricSearch
{
    struct ExcludeSelfCollision
    {
        template <class Predicate, class OutputFunctor>
        KOKKOS_FUNCTION void operator()(
            Predicate const& predicate, int i, OutputFunctor const& out) const
        {
            int const j = getData(predicate);
            if (i > j)
            {
                out(i);
            }
        }
    };

    template <int n_dim>
    std::tuple<pybind11::array_t<int>, pybind11::array_t<int>> find_close_points_template(
        pybind11::array_t<double> coordinates, double tol)
    {
        using memory_space = Kokkos::HostSpace;

        // Get the data from the numpy array and pack the raw point data and the search tolerance
        // into a pair for the ArborX access traits.
        const auto& coordinates_unchecked = coordinates.unchecked<2>();
        const CoordinatesClosestPoints<n_dim> coordinates_with_tol{&coordinates_unchecked, tol};

        // Build tree structure containing all points
        ArborX::BoundingVolumeHierarchy<memory_space> bounding_volume_hierarchy(
            Kokkos::DefaultExecutionSpace{}, coordinates_with_tol);

        // Perform the collision check
        Kokkos::View<int*, Kokkos::HostSpace> indices_arborx("indices_arborx", 0);
        Kokkos::View<int*, Kokkos::HostSpace> offset_arborx("offset_arborx", 0);
        bounding_volume_hierarchy.query(Kokkos::DefaultExecutionSpace{}, coordinates_with_tol,
            ExcludeSelfCollision{}, indices_arborx, offset_arborx);

        // Copy everything into numpy arrays
        auto copy_kokkos_array = [&](const Kokkos::View<int*, Kokkos::HostSpace>& vec)
        {
            auto np_array = pybind11::array_t<int>(vec.size());
            pybind11::buffer_info np_array_buffer = np_array.request();
            int* np_array_ptr = (int*)np_array_buffer.ptr;
            for (unsigned int i = 0; i < vec.size(); i++)
            {
                np_array_ptr[i] = vec[i];
            }
            return np_array;
        };

        // Return the search results
        return std::make_tuple(copy_kokkos_array(indices_arborx), copy_kokkos_array(offset_arborx));
    }


    std::tuple<pybind11::array_t<int>, pybind11::array_t<int>> find_close_points_factory(
        pybind11::array_t<double> coordinates, double tol)
    {
        switch (coordinates.shape(1))
        {
            case 3:
                return find_close_points_template<3>(coordinates, tol);
            default:
                throw std::out_of_range("Got unexpected number of spatial dimensions");
        }
    }


    std::tuple<pybind11::array_t<int>, int> find_close_points(
        pybind11::array_t<double> coordinates, double tol)
    {
        const auto& [indices, offsets] = find_close_points_factory(coordinates, tol);
        const auto& indices_unchecked = indices.unchecked<1>();
        const auto& offsets_unchecked = offsets.unchecked<1>();


        // Create empty data array.
        const auto n_points = coordinates.shape(0);
        auto clusters = pybind11::array_t<int>(n_points);
        pybind11::buffer_info clusters_buffer = clusters.request();
        int* clusters_ptr = (int*)clusters_buffer.ptr;
        for (unsigned int i_point = 0; i_point < n_points; i_point++)
        {
            clusters_ptr[i_point] = -1;
        }

        // Loop over ArborX data
        int n_clusters = 0;
        for (unsigned int i_point = 0; i_point < n_points; i_point++)
        {
            const auto start = offsets_unchecked(i_point);
            const auto end = offsets_unchecked(i_point + 1);

            const bool has_cluster = 0 < (end - start);
            const auto my_cluster_id = clusters_ptr[i_point];

            if (has_cluster && my_cluster_id == -1)
            {
                // Loop over matching points and set them to the ID of this cluster.
                for (int j = start; j < end; j++)
                {
                    int j_point = indices_unchecked(j);
                    if (clusters_ptr[j_point] == -1)
                    {
                        clusters_ptr[j_point] = n_clusters;
                    }
                    else
                    {
                        throw std::out_of_range("No point can be part of two clusters");
                    }
                }
                clusters_ptr[i_point] = n_clusters;
                n_clusters += 1;
            }
            else if (has_cluster)
            {
                // The point is part of a cluster and the cluster ID is already set. Check that all
                // partners match the ID.
                for (int j = start; j < end; j++)
                {
                    int j_point = indices_unchecked(j);
                    if (clusters_ptr[j_point] != my_cluster_id)
                    {
                        throw std::out_of_range("No point can be part of two clusters");
                    }
                }
            }
        }

        return std::make_tuple(clusters, n_clusters);
    }
}  // namespace GeometricSearch
