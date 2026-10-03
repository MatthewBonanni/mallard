/**
 * @file gmsh_fixtures.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Gmsh test meshes written on the fly.
 * @version 0.1
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef GMSH_FIXTURES_H
#define GMSH_FIXTURES_H

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "comm.h"

/** @brief Write content to a file in the temp directory; returns its path. */
inline std::string write_temp(const std::string & name, const std::string & content) {
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream(path) << content;
    return path.string();
}

/** @brief write_temp from rank 0 only, visible to every rank on return. */
inline std::string write_temp_shared(const std::string & name, const std::string & content) {
    if (comm::rank() == 0) write_temp(name, content);
    comm::barrier();
    return (std::filesystem::temp_directory_path() / name).string();
}

/**
 * @brief Gmsh 2.2 file of the unit square: columns alternate between quads and
 *        pairs of triangles, and interior nodes are randomly displaced.
 */
inline std::string jittered_mixed_mesh(uint32_t n) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> jitter(-0.2, 0.2);
    std::ostringstream s;
    s << "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n4\n"
      << "1 1 \"bottom\"\n1 2 \"right\"\n1 3 \"top\"\n1 4 \"left\"\n$EndPhysicalNames\n";
    auto id = [&](uint32_t i, uint32_t j) { return j * (n + 1) + i + 1; };
    s << "$Nodes\n" << (n + 1) * (n + 1) << "\n";
    for (uint32_t j = 0; j <= n; j++) {
        for (uint32_t i = 0; i <= n; i++) {
            const bool interior = i > 0 && i < n && j > 0 && j < n;
            const double x = (i + (interior ? jitter(rng) : 0.0)) / n;
            const double y = (j + (interior ? jitter(rng) : 0.0)) / n;
            s << id(i, j) << " " << x << " " << y << " 0\n";
        }
    }
    std::vector<std::string> elements;
    for (uint32_t i = 0; i < n; i++) {
        elements.push_back("1 2 1 1 " + std::to_string(id(i, 0)) + " " + std::to_string(id(i + 1, 0)));
        elements.push_back("1 2 3 3 " + std::to_string(id(i + 1, n)) + " " + std::to_string(id(i, n)));
        elements.push_back("1 2 4 4 " + std::to_string(id(0, i + 1)) + " " + std::to_string(id(0, i)));
        elements.push_back("1 2 2 2 " + std::to_string(id(n, i)) + " " + std::to_string(id(n, i + 1)));
    }
    for (uint32_t j = 0; j < n; j++) {
        for (uint32_t i = 0; i < n; i++) {
            const auto a = std::to_string(id(i, j)), b = std::to_string(id(i + 1, j));
            const auto c = std::to_string(id(i + 1, j + 1)), d = std::to_string(id(i, j + 1));
            if (i % 2 == 0) {
                elements.push_back("3 2 0 1 " + a + " " + b + " " + c + " " + d);
            } else {
                elements.push_back("2 2 0 1 " + a + " " + b + " " + c);
                elements.push_back("2 2 0 1 " + a + " " + c + " " + d);
            }
        }
    }
    s << "$EndNodes\n$Elements\n" << elements.size() << "\n";
    for (size_t k = 0; k < elements.size(); k++) s << k + 1 << " " << elements[k] << "\n";
    s << "$EndElements\n";
    return s.str();
}

/**
 * @brief Gmsh 2.2 file of the unit cube with zones left/right (x),
 *        bottom/top (y) and back/front (z): columns alternate between
 *        hexahedra and pairs of prisms. Every node moves by a random
 *        displacement that depends on its lattice indices modulo n and keeps
 *        it on its boundary planes, so the mesh is jittered everywhere, seams
 *        included, yet each zone is its opposite zone translated.
 */
inline std::string jittered_periodic_mesh_3d(uint32_t n, double amplitude = 0.15) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> jitter(-amplitude, amplitude);
    std::vector<std::array<double, 3>> displacement(n * n * n);
    for (auto & d : displacement) d = {jitter(rng), jitter(rng), jitter(rng)};
    auto id = [&](uint32_t i, uint32_t j, uint32_t k) { return std::to_string((k * (n + 1) + j) * (n + 1) + i + 1); };
    std::ostringstream s;
    s << "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n6\n"
      << "2 1 \"left\"\n2 2 \"right\"\n2 3 \"bottom\"\n2 4 \"top\"\n2 5 \"back\"\n2 6 \"front\"\n"
      << "$EndPhysicalNames\n$Nodes\n" << (n + 1) * (n + 1) * (n + 1) << "\n";
    s.precision(17);
    for (uint32_t k = 0; k <= n; k++) {
        for (uint32_t j = 0; j <= n; j++) {
            for (uint32_t i = 0; i <= n; i++) {
                const uint32_t idx[3] = {i, j, k};
                const auto & d = displacement[((k % n) * n + j % n) * n + i % n];
                s << id(i, j, k);
                for (int a = 0; a < 3; a++) s << " " << (idx[a] + (idx[a] % n ? d[a] : 0.0)) / n;
                s << "\n";
            }
        }
    }
    std::vector<std::string> elements;
    auto quad = [&](int tag, const std::string & a, const std::string & b, const std::string & c,
                    const std::string & d) {
        elements.push_back("3 2 " + std::to_string(tag) + " " + std::to_string(tag) + " " + a + " " + b + " " + c +
                           " " + d);
    };
    auto tri = [&](int tag, const std::string & a, const std::string & b, const std::string & c) {
        elements.push_back("2 2 " + std::to_string(tag) + " " + std::to_string(tag) + " " + a + " " + b + " " + c);
    };
    for (uint32_t p = 0; p < n; p++) {
        for (uint32_t q = 0; q < n; q++) {
            for (const uint32_t e : {0u, n}) {
                quad(e ? 2 : 1, id(e, p, q), id(e, p + 1, q), id(e, p + 1, q + 1), id(e, p, q + 1));
                quad(e ? 4 : 3, id(p, e, q), id(p + 1, e, q), id(p + 1, e, q + 1), id(p, e, q + 1));
                const int tag = e ? 6 : 5;
                if (p % 2 == 0) {
                    quad(tag, id(p, q, e), id(p + 1, q, e), id(p + 1, q + 1, e), id(p, q + 1, e));
                } else {
                    tri(tag, id(p, q, e), id(p + 1, q, e), id(p + 1, q + 1, e));
                    tri(tag, id(p, q, e), id(p + 1, q + 1, e), id(p, q + 1, e));
                }
            }
        }
    }
    for (uint32_t k = 0; k < n; k++) {
        for (uint32_t j = 0; j < n; j++) {
            for (uint32_t i = 0; i < n; i++) {
                std::string v[2][4];
                for (uint32_t l = 0; l < 2; l++) {
                    v[l][0] = id(i, j, k + l);
                    v[l][1] = id(i + 1, j, k + l);
                    v[l][2] = id(i + 1, j + 1, k + l);
                    v[l][3] = id(i, j + 1, k + l);
                }
                if (i % 2 == 0) {
                    elements.push_back("5 2 0 1 " + v[0][0] + " " + v[0][1] + " " + v[0][2] + " " + v[0][3] + " " +
                                       v[1][0] + " " + v[1][1] + " " + v[1][2] + " " + v[1][3]);
                } else {
                    for (const auto & [a, b, c] : {std::array<int, 3>{0, 1, 2}, std::array<int, 3>{0, 2, 3}}) {
                        elements.push_back("6 2 0 1 " + v[0][a] + " " + v[0][b] + " " + v[0][c] + " " + v[1][a] +
                                           " " + v[1][b] + " " + v[1][c]);
                    }
                }
            }
        }
    }
    s << "$EndNodes\n$Elements\n" << elements.size() << "\n";
    for (size_t k = 0; k < elements.size(); k++) s << k + 1 << " " << elements[k] << "\n";
    s << "$EndElements\n";
    return s.str();
}

#endif // GMSH_FIXTURES_H
