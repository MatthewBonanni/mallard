/**
 * @file region_sampler_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief The team samplers of a cell's simplices and cones, run by host
 *        teams, against the samplers of one thread.
 * @version 0.1
 * @date 2026-11-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "region_sampler.h"

namespace {

using region::sampler::Shape;

/** @brief A ball's signed distance, or with bounded false an implicit function crossing it twice. */
struct Ball {
    double center[3], radius;
    bool implicit = false;

    KOKKOS_INLINE_FUNCTION double operator()(const double * x) const {
        double r2 = 0.0;
        for (int d = 0; d < N_DIM; d++) r2 += (x[d] - center[d]) * (x[d] - center[d]);
        return implicit ? r2 - radius * radius + 0.05 * Kokkos::sin(9.0 * x[0]) : Kokkos::sqrt(r2) - radius;
    }
};

/** @brief A cell's nodes: a quadrilateral (hexahedron) with one node pulled out of place. */
struct Cell {
    static constexpr uint32_t n = N_DIM == 2 ? 4 : 8;

    KOKKOS_INLINE_FUNCTION void operator()(const uint32_t k, double * x) const {
        x[0] = (k & 1) ^ ((k >> 1) & 1) ? 0.6 : 0.2;
        x[1] = (k >> 1) & 1 ? 0.7 : 0.3;
        x[2] = N_DIM == 2 ? 0.0 : k >= 4 ? 0.65 : 0.25;
        if (k == 2) x[0] += 0.07;
    }
};

/** @brief A curved cone: box coordinates (s, t) or (s, u, t) bent onto an arc, with a volume per unit p varying along it. */
struct Bent {
    KOKKOS_INLINE_FUNCTION double operator()(uint32_t, const double * p, double * x) const {
        const double s = p[0], t = p[N_DIM - 1];
        x[0] = 0.4 + 0.3 * t * Kokkos::cos(1.2 * s);
        x[1] = 0.5 + 0.3 * t * Kokkos::sin(1.2 * s);
        x[2] = N_DIM == 3 ? 0.2 + 0.5 * p[1] : 0.0;
        return 0.09 * t * (1.0 + 0.2 * s);
    }
};

struct Sums {
    double inside = 0.0, total = 0.0;
};

/**
 * @brief f(member, scratch, inside, total) from sums start, by host teams of
 *        the largest size up to 4, as rank 0 of each saw it.
 */
template <typename F>
Sums by_teams(const F & f, const Sums start = {}) {
    using Policy = Kokkos::TeamPolicy<Kokkos::DefaultHostExecutionSpace>;
    const int size = std::min(4, Policy(1, 1).team_size_max([](const Policy::member_type &) {}, Kokkos::ParallelForTag()));
    Policy policy(2, size);
    policy.set_scratch_size(0, Kokkos::PerTeam(region::sampler::team_bytes(size)));
    Kokkos::View<double *[2], Kokkos::HostSpace> out("sampler_sums", 2 * size);
    Kokkos::parallel_for("region_sampler_test", policy, [=](const Policy::member_type & member) {
        const auto scratch = region::sampler::team_scratch(member);
        double inside = start.inside, total = start.total;
        f(member, scratch, inside, total);
        const int i = member.league_rank() * member.team_size() + member.team_rank();
        out(i, 0) = inside;
        out(i, 1) = total;
    });
    Kokkos::fence();
    for (size_t i = 1; i < out.extent(0); i++) {
        EXPECT_EQ(out(i, 0), out(0, 0)) << "thread " << i;
        EXPECT_EQ(out(i, 1), out(0, 1)) << "thread " << i;
    }
    return {out(0, 0), out(0, 1)};
}

} // namespace

TEST(RegionSamplerTest, AdditionsInABinadeAtOnceAreTheAdditionsOneAtATime) {
    using region::sampler::detail::add_in_binade;
    std::mt19937_64 rng(1101);
    std::uniform_real_distribution<double> unit(1.0, 2.0);
    std::uniform_int_distribution<uint32_t> count(0, 600);
    int added = 0, refused = 0;
    for (int trial = 0; trial < 20000; trial++) {
        // A running sum and the measures of 3 levels of simplices, as a sharp sum adds them
        const double measure0 = unit(rng) * std::ldexp(1.0, -30), r0 = unit(rng) * std::ldexp(1.0, -20 - trial % 8);
        const double m[3] = {measure0 / 8.0, measure0 / 64.0, measure0 / 512.0};
        uint32_t n[3] = {count(rng), count(rng), count(rng)};
        if (trial % 7 == 0) n[trial % 3] = 0;
        double r = r0;
        const bool at_once = add_in_binade(r, m, n, 3);
        (at_once ? added : refused)++;
        if (!at_once) {
            EXPECT_EQ(r, r0);
            continue;
        }
        // Interleaved: the order makes no difference within a binade
        double one_at_a_time = r0;
        uint32_t left[3] = {n[0], n[1], n[2]};
        for (uint64_t i = 0; left[0] + left[1] + left[2] > 0; i++) {
            const int l = static_cast<int>((i * 2654435761u) % 3);
            if (left[l] == 0) continue;
            one_at_a_time += m[l];
            left[l]--;
        }
        ASSERT_EQ(r, one_at_a_time) << "trial " << trial;
    }
    EXPECT_GT(added, 5000);
    EXPECT_GT(refused, 1000);

    // Ties: m one and a half ulps of r, rounded to even, so alternately 2 ulps and 1
    const double r0 = 1.0, half_ulp = std::ldexp(1.0, -53);
    const double m[1] = {3.0 * half_ulp};
    const uint32_t n[1] = {4};
    double r = r0;
    EXPECT_FALSE(add_in_binade(r, m, n, 1));
    EXPECT_EQ(r, r0);
    // Nothing to add, and sums that are not positive and normal
    const uint32_t none[1] = {0};
    r = 0.0;
    EXPECT_TRUE(add_in_binade(r, m, none, 1));
    EXPECT_FALSE(add_in_binade(r, m, n, 1));
    r = -1.0;
    EXPECT_FALSE(add_in_binade(r, m, n, 1));

    // Up to the last value of a binade, and one addition past it
    const double ulp = std::ldexp(1.0, -53), below = 1.0 - 8.0 * ulp;
    const double one_ulp[1] = {ulp};
    for (const uint32_t count : {7u, 8u}) {
        const uint32_t times[1] = {count};
        double one_at_a_time = below;
        for (uint32_t i = 0; i < count; i++) one_at_a_time += ulp;
        r = below;
        EXPECT_EQ(add_in_binade(r, one_ulp, times, 1), count == 7) << count;
        EXPECT_EQ(r, count == 7 ? one_at_a_time : below) << count;
    }
}

TEST(RegionSamplerTest, TeamsSumWhatOneThreadSumsAcrossBinades) {
    // Sums starting just below powers of two that the cell's simplices carry them past
    const Shape<Ball> shape{{{0.43, 0.51, 0.47}, 0.17}, 0.0, true};
    const Sums start{std::ldexp(1.0 - std::ldexp(1.0, -30), -7), std::ldexp(1.0 - std::ldexp(1.0, -30), -5)};
    Sums cell = start;
    region::sampler::straight_cell(shape, false, Cell::n, Cell{}, cell.inside, cell.total);
    EXPECT_GT(cell.inside, 2.0 * std::ldexp(1.0, -7));
    EXPECT_GT(cell.total, 2.0 * std::ldexp(1.0, -5));
    const Sums team_cell = by_teams([=](const auto & member, const auto & scratch, double & inside, double & total) {
        region::sampler::team_straight_cell(member, scratch, shape, false, Cell::n, Cell{}, inside, total);
    }, start);
    EXPECT_EQ(team_cell.inside, cell.inside);
    EXPECT_EQ(team_cell.total, cell.total);
}

TEST(RegionSamplerTest, TeamsSumWhatOneThreadSums) {
    const Ball ball{{0.43, 0.51, 0.47}, 0.17};
    struct Case {
        std::string name;
        Shape<Ball> shape;
        bool axisymmetric;
    };
    Ball implicit = ball;
    implicit.implicit = true;
    std::vector<Case> cases = {{"sharp", {ball, 0.0, true}, false},
                               {"smooth", {ball, 0.06, true}, false},
                               {"implicit", {implicit, 0.0, false}, false}};
    if (N_DIM == 2) cases.push_back({"axisymmetric", {ball, 0.0, true}, true});
    for (const Case & c : cases) {
        const Shape<Ball> shape = c.shape;
        const bool axi = c.axisymmetric;
        Sums cell;
        region::sampler::straight_cell(shape, axi, Cell::n, Cell{}, cell.inside, cell.total);
        const Sums team_cell = by_teams([=](const auto & member, const auto & scratch, double & inside, double & total) {
            region::sampler::team_straight_cell(member, scratch, shape, axi, Cell::n, Cell{}, inside, total);
        });
        EXPECT_GT(cell.inside, 0.0) << c.name;
        EXPECT_LT(cell.inside, cell.total) << c.name;
        EXPECT_EQ(team_cell.inside, cell.inside) << c.name;
        EXPECT_EQ(team_cell.total, cell.total) << c.name;

        const double lo[3] = {0.0, 0.0, 0.0}, hi[3] = {1.0, 1.0, 1.0};
        Sums cone;
        region::sampler::cone(shape, axi, Bent{}, 0, lo, hi, cone.inside, cone.total);
        const Sums team_cone = by_teams([=](const auto & member, const auto & scratch, double & inside, double & total) {
            region::sampler::team_cone(member, scratch, shape, axi, Bent{}, 0, lo, hi, inside, total);
        });
        EXPECT_GT(cone.inside, 0.0) << c.name;
        EXPECT_LT(cone.inside, cone.total) << c.name;
        EXPECT_EQ(team_cone.inside, cone.inside) << c.name;
        EXPECT_EQ(team_cone.total, cone.total) << c.name;
    }
}
