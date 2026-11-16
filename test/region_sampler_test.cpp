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

/** @brief f(member, scratch, inside, total) by host teams of the largest size up to 4, as rank 0 of each saw it. */
template <typename F>
Sums by_teams(const F & f) {
    using Policy = Kokkos::TeamPolicy<Kokkos::DefaultHostExecutionSpace>;
    const int size = std::min(4, Policy(1, 1).team_size_max([](const Policy::member_type &) {}, Kokkos::ParallelForTag()));
    Policy policy(2, size);
    policy.set_scratch_size(0, Kokkos::PerTeam(region::sampler::team_bytes(size)));
    Kokkos::View<double *[2], Kokkos::HostSpace> out("sampler_sums", 2 * size);
    Kokkos::parallel_for("region_sampler_test", policy, [=](const Policy::member_type & member) {
        const auto scratch = region::sampler::team_scratch(member);
        double inside = 0.0, total = 0.0;
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
