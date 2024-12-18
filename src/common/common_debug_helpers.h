/**
 * @file common_debug_helpers.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Common debugging helpers.
 * @version 0.1
 * @date 2024-12-07
 * 
 * @copyright Copyright (c) 2024 Matthew Bonanni
 * 
 */

#ifndef COMMON_DEBUG_HELPERS_H
#define COMMON_DEBUG_HELPERS_H

#include <iostream>
#include <vector>
#include <iomanip>

#include <Kokkos_Core.hpp>

// Recursive function to print the elements of a Kokkos view
template <typename T>
void print_view_recursive(const T* data, const std::vector<size_t>& extents, std::ostream& os,
                          std::vector<size_t>& indices, size_t dim, const std::vector<size_t>& strides) {
    if (dim == extents.size()) {
        // Base case: compute flat index and print element
        size_t flat_index = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            flat_index += indices[i] * strides[i];
        }
        os << std::setw(6) << data[flat_index]; // Align output for readability
    } else {
        // Recursive case: iterate through the current dimension
        for (size_t i = 0; i < extents[dim]; ++i) {
            indices[dim] = i;
            if (dim < extents.size() - 1) {
                for (size_t j = 0; j < dim; ++j) {
                    os << "     "; // Indent for subdimensions
                }
                os << std::setw(5) << i;
            }
            print_view_recursive(data, extents, os, indices, dim + 1, strides);
            if (dim < extents.size() - 1) {
                os << "\n";
            }
        }
    }
}

// Pretty-printing function for Kokkos views
template <typename ViewType>
void print_view(const ViewType& view, std::ostream& os = std::cout) {
    static_assert(Kokkos::is_view<ViewType>::value, "Input must be a Kokkos view");

    if (view.size() == 0) {
        os << "[] (Empty view)\n";
        return;
    }

    // Extract raw data pointer
    const auto* data = view.data();

    // Gather extents and strides for all dimensions
    std::vector<size_t> extents(view.rank());
    std::vector<size_t> strides(view.rank());
    view.stride(strides.data());
    for (size_t i = 0; i < view.rank(); ++i) {
        extents[i] = view.extent(i);
    }

    // Determine layout
    std::string layout = (std::is_same<typename ViewType::array_layout, Kokkos::LayoutRight>::value)
                             ? "Kokkos::LayoutRight"
                             : "Kokkos::LayoutLeft";

    // Print metadata
    os << "View Metadata:\n";
    os << "  Rank: " << view.rank() << "\n";
    os << "  Extents: [";
    for (size_t i = 0; i < extents.size(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << extents[i];
    }
    os << "]\n";
    os << "  Strides: [";
    for (size_t i = 0; i < strides.size(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << strides[i];
    }
    os << "]\n";
    os << "  Layout: " << layout << "\n";

    // Print the view
    os << "View Elements:\n";
    os << "Dim" << view.rank() - 1 << ":";
    if (view.rank() > 1) {
        for (size_t i = 0; i < view.rank() - 2; ++i) {
            os << "     ";
        }
    }
    for (size_t i = 0; i < extents[view.rank() - 1]; ++i) {
        os << std::setw(6) << i;
    }
    os << "\n";
    for (size_t i = 0; i < view.rank() - 1; ++i) {
        os << "Dim" << i << " ";
    }
    os << "\n";
    std::vector<size_t> indices(view.rank(), 0);
    print_view_recursive(data, extents, os, indices, 0, strides);
    os << "\n";
}

#endif // COMMON_DEBUG_HELPERS_H