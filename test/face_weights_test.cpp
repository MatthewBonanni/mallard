/**
 * @file face_weights_test.cpp
 * @brief Face quadrature weights stored once per distinct row.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstring>
#include <vector>

#include "face_weights.h"

namespace {

using Rows = std::vector<std::array<rtype, 2>>;

FaceWeights<> share(const Rows & faces, const bool on = true) {
    return share_face_weights("face_quad_weights", static_cast<uint32_t>(faces.size()), 2,
                              [&](const uint32_t f, rtype * w) {
                                  w[0] = faces[f][0];
                                  w[1] = faces[f][1];
                              },
                              on);
}

void expect_values(const FaceWeights<> & weights, const Rows & faces) {
    const auto h = weights.host();
    ASSERT_EQ(h.extent(0), faces.size());
    ASSERT_EQ(h.extent(1), 2u);
    for (uint32_t f = 0; f < faces.size(); f++) {
        for (uint32_t q = 0; q < 2; q++) {
            const rtype w = h(f, q);
            EXPECT_EQ(std::memcmp(&w, &faces[f][q], sizeof(rtype)), 0) << "face " << f << ", point " << q;
        }
    }
}

} // namespace

TEST(FaceWeights, FacesShareOnlyBitwiseEqualRowsAndReadTheirOwnValues) {
    const rtype a = rtype(0.5), b = std::nextafter(a, rtype(1));
    const Rows faces = {{a, 1}, {a, 1}, {b, 1}, {rtype(0), 1}, {-rtype(0), 1}, {a, 1}, {a, 2}};
    const FaceWeights<> weights = share(faces);
    // a, the next number after it, +0, -0, and a again with another second weight
    EXPECT_EQ(weights.table.extent(0), 5u);
    ASSERT_TRUE(weights.rows.is_allocated());
    const auto rows = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), weights.rows);
    const std::vector<uint32_t> expected = {0, 0, 1, 2, 3, 0, 4};
    for (uint32_t f = 0; f < faces.size(); f++) EXPECT_EQ(rows(f), expected[f]) << "face " << f;
    expect_values(weights, faces);
}

TEST(FaceWeights, FacesWithoutEqualRowsOrWithoutSharingKeepARowEach) {
    const Rows distinct = {{1, 2}, {2, 1}, {1, 3}};
    const FaceWeights<> own = share(distinct);
    EXPECT_FALSE(own.rows.is_allocated());
    EXPECT_EQ(own.table.extent(0), distinct.size());
    expect_values(own, distinct);

    const Rows equal = {{1, 2}, {1, 2}, {1, 2}};
    const FaceWeights<> unshared = share(equal, false);
    EXPECT_FALSE(unshared.rows.is_allocated());
    EXPECT_EQ(unshared.table.extent(0), equal.size());
    expect_values(unshared, equal);
}
