/**
 * @file probe_surface.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Probes on mesh faces: boundary zones.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "probe_surface.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <tuple>
#include <stdexcept>
#include <unordered_map>

#include "comm.h"

namespace probe {

namespace {

/** @brief A face as gathered on rank 0. */
struct Record {
    uint64_t key = 0;  // global cell * 64 + local face of the cell
    std::vector<uint64_t> nodes;
    Point centroid{}, normal{};
    double measure = 0.0;
    std::vector<Point> coords;
};

/**
 * @brief True area (2D: length) of a curved face, by a Gauss rule on its
 *        reference element: its area vector is that of its chord.
 */
double curved_measure(const curved::Geometry & geometry, uint32_t f, uint32_t n_nodes) {
    std::vector<std::array<double, 2>> ref;
    std::vector<double> weights;
    const double g[4] = {-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526};
    const double w[4] = {0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538};
    if (N_DIM == 2 || n_nodes == 4) {
        for (int i = 0; i < 4; i++) {
            if constexpr (N_DIM == 2) {
                ref.push_back({g[i], 0.0});
                weights.push_back(w[i]);
            } else {
                for (int j = 0; j < 4; j++) {
                    ref.push_back({g[i], g[j]});
                    weights.push_back(w[i] * w[j]);
                }
            }
        }
    } else {
        // Strang-Fix 6-point rule (degree 4), weights summing to 1 as face_rule takes them
        const double a = 0.445948490915965, b = 0.091576213509771;
        const double wa = 0.223381589678011, wb = 0.109951743655322;
        for (const auto & [u, v, wt] : {std::tuple{a, a, wa}, std::tuple{1 - 2 * a, a, wa}, std::tuple{a, 1 - 2 * a, wa},
                                        std::tuple{b, b, wb}, std::tuple{1 - 2 * b, b, wb}, std::tuple{b, 1 - 2 * b, wb}}) {
            ref.push_back({u, v});
            weights.push_back(wt);
        }
    }
    std::vector<curved::Vec3> x, n;
    std::vector<double> dA;
    geometry.face_rule(f, ref, weights, x, n, dA);
    double area = 0.0;
    for (double d : dA) area += d;
    return area;
}

} // namespace

FaceSet boundary_faces(const Mesh & mesh, const std::vector<std::string> & zones, bool distributed) {
    FaceSet set;
    const uint32_t n_owned = mesh.n_owned();
    std::vector<uint8_t> taken(mesh.n_faces, 0);
    for (const auto & zone : mesh.h_face_zones()) {
        if (std::find(zones.begin(), zones.end(), zone.get_name()) == zones.end()) continue;
        for (uint32_t i = 0; i < zone.n_faces(); i++) {
            const uint32_t f = zone.h_faces(i);
            const int32_t c = mesh.h_cells_of_face(f, 0);
            if (taken[f] || c < 0 || static_cast<uint32_t>(c) >= n_owned || mesh.h_cells_of_face(f, 1) >= 0) continue;
            taken[f] = 1;
            set.faces.push_back(f);
        }
    }

    // Pack: [key, n_nodes, node ids...] and [centroid, normal, measure, node coordinates...]
    std::vector<uint64_t> ids;
    std::vector<double> reals;
    for (uint32_t f : set.faces) {
        const uint32_t c = static_cast<uint32_t>(mesh.h_cells_of_face(f, 0));
        uint32_t k = 0;
        while (mesh.h_face_of_cell(c, k) != f) k++;
        const uint32_t n_nodes = mesh.h_n_nodes_of_face(f);
        ids.push_back(mesh.h_global_cell(c) * 64 + k);
        ids.push_back(n_nodes);
        double area = 0.0;
        for (size_t d = 0; d < N_DIM; d++) area += std::pow(static_cast<double>(mesh.h_face_normals(f, d)), 2);
        area = std::sqrt(area);
        for (size_t d = 0; d < N_DIM; d++) reals.push_back(static_cast<double>(mesh.h_face_coords(f, d)));
        for (size_t d = 0; d < N_DIM; d++) reals.push_back(static_cast<double>(mesh.h_face_normals(f, d)) / area);
        reals.push_back(mesh.curved_geometry && mesh.curved_geometry->face_is_curved(f)
                            ? curved_measure(*mesh.curved_geometry, f, n_nodes)
                            : static_cast<double>(mesh.h_face_measure(f)));
        for (uint32_t j = 0; j < n_nodes; j++) {
            const uint32_t node = mesh.h_node_of_face(f, j);
            ids.push_back(mesh.h_global_node_id.empty() ? node : mesh.h_global_node_id[node]);
            for (size_t d = 0; d < N_DIM; d++) reals.push_back(static_cast<double>(mesh.h_node_coords(node, d)));
        }
    }
    const uint64_t n_local = set.faces.size();
    set.n_global = distributed ? comm::allreduce(n_local, comm::Op::SUM) : n_local;
    const std::vector<uint64_t> all_ids = distributed ? comm::gatherv(ids) : ids;
    const std::vector<double> all_reals = distributed ? comm::gatherv(reals) : reals;
    if (distributed && !comm::is_root()) return set;

    std::vector<Record> records;
    size_t a = 0, b = 0;
    while (a < all_ids.size()) {
        Record r;
        r.key = all_ids[a++];
        const uint64_t n_nodes = all_ids[a++];
        for (size_t d = 0; d < N_DIM; d++) r.centroid[d] = all_reals[b++];
        for (size_t d = 0; d < N_DIM; d++) r.normal[d] = all_reals[b++];
        r.measure = all_reals[b++];
        for (uint64_t j = 0; j < n_nodes; j++) {
            r.nodes.push_back(all_ids[a++]);
            Point x;
            for (size_t d = 0; d < N_DIM; d++) x[d] = all_reals[b++];
            r.coords.push_back(x);
        }
        records.push_back(std::move(r));
    }

    Geometry & g = set.geometry;
    g.dim = N_DIM - 1;
    g.shape = "boundary";
    std::vector<uint64_t> order;  // record of each global sample
    if constexpr (N_DIM == 2) {
        // Each face from node 0 to node 1 with the fluid on its left: along t = (-n_y, n_x)
        for (auto & r : records) {
            const double along = (r.coords[1][0] - r.coords[0][0]) * -r.normal[1] +
                                 (r.coords[1][1] - r.coords[0][1]) * r.normal[0];
            if (along < 0.0) {
                std::swap(r.nodes[0], r.nodes[1]);
                std::swap(r.coords[0], r.coords[1]);
            }
        }
        std::multimap<uint64_t, uint64_t> from;  // start node -> record
        std::unordered_map<uint64_t, uint32_t> ends;
        for (uint64_t i = 0; i < records.size(); i++) {
            from.emplace(records[i].nodes[0], i);
            ends[records[i].nodes[1]]++;
        }
        std::vector<uint64_t> starts(records.size());
        std::iota(starts.begin(), starts.end(), 0);
        auto by_start = [&](uint64_t x, uint64_t y) {
            return records[x].nodes[0] != records[y].nodes[0] ? records[x].nodes[0] < records[y].nodes[0]
                                                               : records[x].key < records[y].key;
        };
        std::sort(starts.begin(), starts.end(), by_start);
        // Open chains first (from a node that ends no face), then closed loops
        std::stable_partition(starts.begin(), starts.end(),
                              [&](uint64_t i) { return ends.find(records[i].nodes[0]) == ends.end(); });
        std::vector<uint8_t> used(records.size(), 0);
        for (uint64_t first : starts) {
            if (used[first]) continue;
            std::vector<uint32_t> line;
            double s = 0.0;
            uint64_t i = first;
            while (true) {
                used[i] = 1;
                const Record & r = records[i];
                const double L = std::hypot(r.coords[1][0] - r.coords[0][0], r.coords[1][1] - r.coords[0][1]);
                line.push_back(static_cast<uint32_t>(order.size()));
                order.push_back(i);
                g.arc_length.push_back(s + 0.5 * L);
                s += L;
                int64_t next = -1;
                const auto [lo, hi] = from.equal_range(r.nodes[1]);
                for (auto it = lo; it != hi; ++it) {
                    if (!used[it->second] && (next < 0 || by_start(it->second, static_cast<uint64_t>(next)))) {
                        next = static_cast<int64_t>(it->second);
                    }
                }
                if (next < 0) break;
                i = static_cast<uint64_t>(next);
            }
            if (records[i].nodes[1] == records[first].nodes[0] && line.size() > 1) line.push_back(line.front());
            g.polylines.push_back(std::move(line));
        }
    } else {
        order.resize(records.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](uint64_t x, uint64_t y) { return records[x].key < records[y].key; });
        std::unordered_map<uint64_t, uint32_t> vertex;
        g.polygon_offsets.push_back(0);
        for (uint64_t i : order) {
            const Record & r = records[i];
            for (size_t j = 0; j < r.nodes.size(); j++) {
                const auto [it, added] = vertex.emplace(r.nodes[j], static_cast<uint32_t>(g.vertices.size()));
                if (added) g.vertices.push_back(r.coords[j]);
                g.polygon_vertices.push_back(it->second);
            }
            g.polygon_offsets.push_back(g.polygon_vertices.size());
        }
    }
    set.gathered_row.assign(records.size(), 0);
    for (uint64_t k = 0; k < order.size(); k++) {
        const Record & r = records[order[k]];
        set.gathered_row[k] = order[k];
        g.nodes.push_back(r.centroid);
        g.normals.push_back(r.normal);
        g.node_weights.push_back(r.measure);
    }
    return set;
}

} // namespace probe
