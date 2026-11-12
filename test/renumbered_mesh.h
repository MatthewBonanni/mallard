/**
 * @file renumbered_mesh.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Generated test meshes written as Gmsh files, as generated or with
 *        their cells, nodes and boundary faces renumbered at random.
 * @version 0.1
 * @date 2026-10-23
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef RENUMBERED_MESH_H
#define RENUMBERED_MESH_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "mesh.h"
#include "mesh_block.h"

/**
 * @brief A generated box [0, L] (cartesian_2d_block or cartesian_3d_block of
 *        the given kind) whose interior nodes are displaced at random by up
 *        to amplitude times the spacing, so that no two distances tie.
 */
inline MeshBlock jittered_box(MeshType kind, const std::array<uint32_t, N_DIM> & n,
                              const std::array<double, N_DIM> & L, double amplitude = 0.15) {
    MeshBlock block;
#if N_DIM == 2
    block = cartesian_2d_block(n[0], n[1], static_cast<rtype>(L[0]), static_cast<rtype>(L[1]), kind, 0, 1);
#else
    const auto [cells, nodes] = cartesian_3d_size(n[0], n[1], n[2], kind);
    block = cartesian_3d_block(n[0], n[1], n[2], static_cast<rtype>(L[0]), static_cast<rtype>(L[1]),
                               static_cast<rtype>(L[2]), kind, 0, cells, 0, nodes);
#endif
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> jitter(-amplitude, amplitude);
    for (auto & x : block.node_coords) {
        bool interior = true;
        FOR_I_DIM interior = interior && x[i] > 1e-9 * L[i] && x[i] < (1.0 - 1e-9) * L[i];
        if (!interior) continue;
        FOR_I_DIM x[i] += jitter(rng) * L[i] / n[i];
    }
    return block;
}

/** @brief A mesh as a Gmsh file, and the original cell of each of its cells. */
struct RenumberedMesh {
    std::string gmsh;
    std::vector<uint32_t> original_cell;
};

/**
 * @brief The block as a Gmsh 2.2 file. A nonzero seed renumbers it at random:
 *        cells, nodes and boundary faces are shuffled, and each cell's and
 *        boundary face's node list is mapped by a random symmetry of its
 *        element, reflections included (2D: rotations and reversal; 3D:
 *        rotations of the element and a mirror), so faces come out with
 *        other owners, other first nodes and the opposite orientation.
 */
inline RenumberedMesh renumbered_gmsh(const MeshBlock & block, uint32_t seed = 0) {
    const uint32_t n_cells = static_cast<uint32_t>(block.n_cells());
    const uint32_t n_nodes = static_cast<uint32_t>(block.n_nodes());
    const uint32_t n_faces = static_cast<uint32_t>(block.n_faces());
    std::mt19937 rng(seed);
    auto order = [&](uint32_t n) {
        std::vector<uint32_t> p(n);
        std::iota(p.begin(), p.end(), 0u);
        if (seed != 0) std::shuffle(p.begin(), p.end(), rng);
        return p;
    };
    const std::vector<uint32_t> cell_order = order(n_cells), node_order = order(n_nodes), face_order = order(n_faces);
    std::vector<uint32_t> new_node(n_nodes);
    for (uint32_t k = 0; k < n_nodes; k++) new_node[node_order[k]] = k;

    // Generators of each element's symmetries, as maps new local node -> old
    auto generators = [](size_t n) -> std::vector<std::vector<uint8_t>> {
        if (N_DIM == 2 || n <= 3) {
            std::vector<uint8_t> rotate(n), reverse(n);
            for (size_t k = 0; k < n; k++) {
                rotate[k] = static_cast<uint8_t>((k + 1) % n);
                reverse[k] = static_cast<uint8_t>(n - 1 - k);
            }
            return {rotate, reverse};
        }
        switch (n) {
            case 4: return {{1, 2, 0, 3}, {0, 2, 3, 1}, {0, 2, 1, 3}};
            case 5: return {{1, 2, 3, 0, 4}, {0, 3, 2, 1, 4}};
            case 6: return {{1, 2, 0, 4, 5, 3}, {3, 4, 5, 0, 1, 2}, {0, 2, 1, 3, 5, 4}};
            case 8: return {{1, 2, 3, 0, 5, 6, 7, 4}, {3, 2, 6, 7, 0, 1, 5, 4}, {0, 3, 2, 1, 4, 7, 6, 5}};
            default: return {};
        }
    };
    // 3D boundary faces are polygons: rotations and reversal
    auto face_generators = [](size_t n) {
        std::vector<uint8_t> rotate(n), reverse(n);
        for (size_t k = 0; k < n; k++) {
            rotate[k] = static_cast<uint8_t>((k + 1) % n);
            reverse[k] = static_cast<uint8_t>(n - 1 - k);
        }
        return std::vector<std::vector<uint8_t>>{rotate, reverse};
    };
    auto shuffled = [&](std::vector<uint64_t> nodes, const std::vector<std::vector<uint8_t>> & gens) {
        if (seed == 0) return nodes;
        std::uniform_int_distribution<int> times(0, 3);
        for (const auto & g : gens) {
            for (int t = times(rng); t > 0; t--) {
                std::vector<uint64_t> next(nodes.size());
                for (size_t k = 0; k < nodes.size(); k++) next[k] = nodes[g[k]];
                nodes = next;
            }
        }
        return nodes;
    };
    auto element = [&](std::ostringstream & s, size_t id, int dim, int physical, const std::vector<uint64_t> & nodes) {
        int type = 0;
        if (dim == 1) type = 1;
        if (dim == 2) type = nodes.size() == 3 ? 2 : 3;
        if (dim == 3) type = nodes.size() == 4 ? 4 : nodes.size() == 5 ? 7 : nodes.size() == 6 ? 6 : 5;
        s << id << " " << type << " 2 " << physical << " 1";
        for (uint64_t g : nodes) s << " " << new_node[g - block.first_node] + 1;
        s << "\n";
    };

    std::ostringstream s;
    s << std::setprecision(17) << "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n" << block.zone_names.size()
      << "\n";
    for (size_t z = 0; z < block.zone_names.size(); z++) {
        s << N_DIM - 1 << " " << z + 1 << " \"" << block.zone_names[z] << "\"\n";
    }
    s << "$EndPhysicalNames\n$Nodes\n" << n_nodes << "\n";
    for (uint32_t k = 0; k < n_nodes; k++) {
        const auto & x = block.node_coords[node_order[k]];
        s << k + 1 << " " << x[0] << " " << x[1] << " " << (N_DIM == 3 ? x[N_DIM - 1] : 0.0) << "\n";
    }
    s << "$EndNodes\n$Elements\n" << n_faces + n_cells << "\n";
    size_t id = 1;
    for (uint32_t f : face_order) {
        std::vector<uint64_t> nodes(block.face_nodes.begin() + static_cast<std::ptrdiff_t>(block.face_offsets[f]),
                                    block.face_nodes.begin() + static_cast<std::ptrdiff_t>(block.face_offsets[f + 1]));
        element(s, id++, N_DIM - 1, static_cast<int>(block.face_zone[f]) + 1, shuffled(nodes, face_generators(nodes.size())));
    }
    RenumberedMesh out;
    for (uint32_t c : cell_order) {
        std::vector<uint64_t> nodes(block.cell_nodes.begin() + static_cast<std::ptrdiff_t>(block.cell_offsets[c]),
                                    block.cell_nodes.begin() + static_cast<std::ptrdiff_t>(block.cell_offsets[c + 1]));
        element(s, id++, N_DIM, 0, shuffled(nodes, generators(nodes.size())));
        out.original_cell.push_back(c);
    }
    s << "$EndElements\n";
    out.gmsh = s.str();
    return out;
}

#endif // RENUMBERED_MESH_H
