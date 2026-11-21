/**
 * @file face_weights.cpp
 * @brief Per-face quadrature weights stored once per distinct row.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "face_weights.h"

#include <unordered_map>
#include <vector>

std::vector<FaceRows<>> share_face_rows(const std::string & rows_label, const std::vector<std::string> & labels,
                                        const std::vector<uint32_t> & widths, const uint32_t n_faces,
                                        const std::function<void(uint32_t, rtype *)> & row_of, const bool share) {
    uint32_t width = 0;
    for (const uint32_t w : widths) width += w;
    std::vector<rtype> values;
    std::vector<uint32_t> rows(n_faces);
    std::unordered_map<std::string, uint32_t> row_of_bytes;
    std::vector<rtype> v(width);
    for (uint32_t f = 0; f < n_faces; f++) {
        row_of(f, v.data());
        if (!share) {
            values.insert(values.end(), v.begin(), v.end());
            continue;
        }
        std::string bytes(reinterpret_cast<const char *>(v.data()), width * sizeof(rtype));
        const auto [it, added] = row_of_bytes.emplace(std::move(bytes), static_cast<uint32_t>(row_of_bytes.size()));
        if (added) values.insert(values.end(), v.begin(), v.end());
        rows[f] = it->second;
    }
    const uint32_t n_rows = share ? static_cast<uint32_t>(row_of_bytes.size()) : n_faces;
    FaceRows<>::V<uint32_t *> d_rows;
    if (n_rows < n_faces) {
        d_rows = FaceRows<>::V<uint32_t *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, rows_label), n_faces);
        const Kokkos::View<uint32_t *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_rows(rows.data(),
                                                                                                         n_faces);
        Kokkos::deep_copy(d_rows, h_rows);
    }
    std::vector<FaceRows<>> out;
    uint32_t first = 0;
    for (size_t t = 0; t < widths.size(); t++) {
        FaceRows<> table;
        table.n_faces = n_faces;
        table.rows = d_rows;
        table.table = FaceRows<>::V<rtype **>(Kokkos::view_alloc(Kokkos::WithoutInitializing, labels[t]), n_rows,
                                              widths[t]);
        auto h_table = Kokkos::create_mirror_view(table.table);
        for (uint32_t r = 0; r < n_rows; r++) {
            for (uint32_t k = 0; k < widths[t]; k++) h_table(r, k) = values[size_t(r) * width + first + k];
        }
        Kokkos::deep_copy(table.table, h_table);
        out.push_back(table);
        first += widths[t];
    }
    return out;
}

FaceWeights<> share_face_weights(const std::string & label, const uint32_t n_faces, const uint32_t n_q,
                                 const std::function<void(uint32_t, rtype *)> & row_of, const bool share) {
    return share_face_rows(label + "_rows", {label}, {n_q}, n_faces, row_of, share)[0];
}
