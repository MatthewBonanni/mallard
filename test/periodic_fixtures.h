/**
 * @file periodic_fixtures.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Helpers shared by the 2D and 3D periodic-boundary tests.
 * @version 0.1
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef PERIODIC_FIXTURES_H
#define PERIODIC_FIXTURES_H

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <map>
#include <vector>

#include "mesh.h"
#include "solver.h"
#include "test_fixtures.h"

/**
 * @brief Set a solver's state to the cell averages of the conservatives
 *        U(x, y, z) (z = 0 in 2D).
 */
template <typename F>
void set_cell_averages(Solver & solver, F && U) {
    const Mesh & mesh = *solver.get_mesh();
    if constexpr (N_DIM == 2) {
        const auto avg = cell_averages(mesh, [&](double x, double y, double * u) { U(x, y, 0.0, u); });
        Kokkos::deep_copy(solver.h_conservatives, avg);
    } else {
        const auto avg = cell_averages_3d(mesh, [&](double x, double y, double z, double * u) { U(x, y, z, u); });
        Kokkos::deep_copy(solver.h_conservatives, avg);
    }
    solver.copy_host_to_device();
}

/** @brief Node coordinates as a 3-vector. */
inline std::array<double, 3> node_point(const Mesh & mesh, uint32_t n) {
    std::array<double, 3> x = {0.0, 0.0, 0.0};
    FOR_I_DIM x[i] = double(mesh.h_node_coords(n, i));
    return x;
}

/**
 * @brief Every interior face, moved by its shift, is one of its cell 1's own
 *        faces, and every vertex neighbor, moved by its shift, shares a node
 *        with its cell.
 */
inline void expect_shifts_join_cells(const Mesh & mesh) {
    double size = 0.0;
    for (uint32_t n = 0; n < mesh.n_nodes; n++) {
        FOR_I_DIM size = std::max(size, std::abs(double(mesh.h_node_coords(n, i))));
    }
    const double tol = 1e-12 * size;
    auto close = [&](const std::array<double, 3> & a, const std::array<double, 3> & b) {
        double d = 0.0;
        for (size_t i = 0; i < 3; i++) d = std::max(d, std::abs(a[i] - b[i]));
        return d < tol;
    };
    auto average = [&](const std::vector<uint32_t> & nodes, const double * shift) {
        std::array<double, 3> x = {0.0, 0.0, 0.0};
        for (uint32_t n : nodes) {
            const auto p = node_point(mesh, n);
            for (size_t i = 0; i < 3; i++) x[i] += p[i] / static_cast<double>(nodes.size());
        }
        FOR_I_DIM x[i] -= shift[i];
        return x;
    };
    const double zero[3] = {0.0, 0.0, 0.0};
    uint32_t n_shifted = 0;
    for (uint32_t f = 0; f < mesh.n_faces; f++) {
        const int32_t c1_index = mesh.h_cells_of_face(f, 1);
        if (c1_index < 0) continue;
        const uint32_t c1 = static_cast<uint32_t>(c1_index);
        double shift[3] = {0.0, 0.0, 0.0};
        FOR_I_DIM shift[i] = double(mesh.h_shifts(mesh.h_face_shift(f), i));
        n_shifted += mesh.h_face_shift(f) != 0;
        std::vector<uint32_t> face_nodes;
        for (uint32_t k = 0; k < mesh.h_n_nodes_of_face(f); k++) face_nodes.push_back(mesh.h_node_of_face(f, k));
        const auto seen_from_c1 = average(face_nodes, shift);
        bool found = false;
        const uint32_t n = mesh.h_n_nodes_of_cell(c1);
        if constexpr (N_DIM == 2) {
            for (uint32_t k = 0; k < n; k++) {
                found = found || close(seen_from_c1, average({mesh.h_node_of_cell(c1, k),
                                                              mesh.h_node_of_cell(c1, (k + 1) % n)}, zero));
            }
        } else {
            for (const auto & local : cell_local_faces(n)) {
                std::vector<uint32_t> own;
                for (uint8_t k : local) own.push_back(mesh.h_node_of_cell(c1, k));
                found = found || close(seen_from_c1, average(own, zero));
            }
        }
        EXPECT_TRUE(found) << "face " << f;
    }
    EXPECT_GT(n_shifted, 0u);
    for (uint32_t c = 0; c < mesh.n_cells; c++) {
        for (uint32_t k = mesh.h_offsets_cells_of_cell(c); k < mesh.h_offsets_cells_of_cell(c + 1); k++) {
            const uint32_t nb = mesh.h_cells_of_cell(k);
            bool touch = false;
            for (uint32_t a = 0; a < mesh.h_n_nodes_of_cell(c); a++) {
                for (uint32_t b = 0; b < mesh.h_n_nodes_of_cell(nb); b++) {
                    auto q = node_point(mesh, mesh.h_node_of_cell(nb, b));
                    FOR_I_DIM q[i] += double(mesh.h_shifts(mesh.h_cells_of_cell_shift(k), i));
                    touch = touch || close(node_point(mesh, mesh.h_node_of_cell(c, a)), q);
                }
            }
            EXPECT_TRUE(touch) << "cell " << c << ", neighbor " << nb;
        }
    }
}

/**
 * @brief For each cell of a periodic box mesh (axis-aligned translations),
 *        the cell whose centroid is its centroid translated by t, modulo the
 *        box. Throws if a translated cell is not a cell.
 */
inline std::vector<uint32_t> translated_cells(const Mesh & mesh, const std::array<double, 3> & t) {
    std::array<double, 3> L = {0.0, 0.0, 0.0};
    for (const auto & T : mesh.periodic_translations) FOR_I_DIM L[i] += double(T[i]);
    const double h = std::pow(double(mesh.h_cell_volume(0)), 1.0 / N_DIM);
    auto key = [&](std::array<double, 3> x) {
        std::array<long long, 3> k = {0, 0, 0};
        FOR_I_DIM {
            if (L[i] > 0.0) x[i] -= L[i] * std::floor(x[i] / L[i]);
            k[i] = std::llround(x[i] / (1e-3 * h));
        }
        return k;
    };
    std::map<std::array<long long, 3>, uint32_t> cell_at;
    for (uint32_t c = 0; c < mesh.n_cells; c++) {
        std::array<double, 3> x = {0.0, 0.0, 0.0};
        FOR_I_DIM x[i] = double(mesh.h_cell_coords(c, i));
        cell_at[key(x)] = c;
    }
    std::vector<uint32_t> map(mesh.n_cells);
    for (uint32_t c = 0; c < mesh.n_cells; c++) {
        std::array<double, 3> x = {0.0, 0.0, 0.0};
        FOR_I_DIM x[i] = double(mesh.h_cell_coords(c, i)) + t[i];
        map[c] = cell_at.at(key(x));
    }
    return map;
}

#endif // PERIODIC_FIXTURES_H
