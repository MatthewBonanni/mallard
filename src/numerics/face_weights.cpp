/**
 * @file face_weights.cpp
 * @brief Per-face quadrature weights stored once per distinct row.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "face_weights.h"

#include <unordered_map>
#include <vector>

FaceWeights<> share_face_weights(const std::string & label, const uint32_t n_faces, const uint32_t n_q,
                                 const std::function<void(uint32_t, rtype *)> & row_of, const bool share) {
    std::vector<rtype> values;
    std::vector<uint32_t> rows(n_faces);
    std::unordered_map<std::string, uint32_t> row_of_bytes;
    std::vector<rtype> w(n_q);
    for (uint32_t f = 0; f < n_faces; f++) {
        row_of(f, w.data());
        if (!share) {
            values.insert(values.end(), w.begin(), w.end());
            continue;
        }
        std::string bytes(reinterpret_cast<const char *>(w.data()), n_q * sizeof(rtype));
        const auto [it, added] = row_of_bytes.emplace(std::move(bytes), static_cast<uint32_t>(row_of_bytes.size()));
        if (added) values.insert(values.end(), w.begin(), w.end());
        rows[f] = it->second;
    }
    const uint32_t n_rows = share ? static_cast<uint32_t>(row_of_bytes.size()) : n_faces;
    FaceWeights<> out;
    out.n_faces = n_faces;
    out.table = FaceWeights<>::V<rtype **>(Kokkos::view_alloc(Kokkos::WithoutInitializing, label), n_rows, n_q);
    auto h_table = Kokkos::create_mirror_view(out.table);
    for (uint32_t r = 0; r < n_rows; r++) {
        for (uint32_t q = 0; q < n_q; q++) h_table(r, q) = values[size_t(r) * n_q + q];
    }
    Kokkos::deep_copy(out.table, h_table);
    if (n_rows < n_faces) {
        out.rows = FaceWeights<>::V<uint32_t *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, label + "_rows"), n_faces);
        const Kokkos::View<uint32_t *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_rows(rows.data(),
                                                                                                         n_faces);
        Kokkos::deep_copy(out.rows, h_rows);
    }
    return out;
}
