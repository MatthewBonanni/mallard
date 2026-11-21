/**
 * @file gated_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Kernels launched through gated.h do nothing while their flag is 0, and reductions then leave their
 *        destinations as they were, as a step graph's launches after a stop must.
 * @version 0.1
 * @date 2026-11-14
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <stdexcept>
#include <vector>

#include "gated.h"

namespace {

using Memory = Kokkos::DefaultExecutionSpace::memory_space;

/** @brief Sums three columns and adds the sums to a total in its final step (a reduction with its own final). */
struct ColumnSums {
    using value_type = double[];
    const uint32_t value_count = 3;
    Kokkos::View<double *> total;

    KOKKOS_INLINE_FUNCTION void operator()(const uint32_t i, double * sum) const {
        for (uint32_t k = 0; k < value_count; k++) sum[k] += double(i + k);
    }
    KOKKOS_INLINE_FUNCTION void init(double * sum) const {
        for (uint32_t k = 0; k < value_count; k++) sum[k] = 0.0;
    }
    KOKKOS_INLINE_FUNCTION void join(double * dst, const double * src) const {
        for (uint32_t k = 0; k < value_count; k++) dst[k] += src[k];
    }
    KOKKOS_INLINE_FUNCTION void final(double * sum) const {
        for (uint32_t k = 0; k < value_count; k++) total(k) += sum[k];
    }
};

struct Views {
    Kokkos::View<uint32_t> on{"on"};
    Kokkos::View<double *> x{"x", 4}, max{"max", 1}, sum{"sum", 1}, columns{"columns", 3}, total{"total", 3};
    Kokkos::View<int64_t *> scanned{"scanned", 4};
};

void launch(const Views & v, const uint32_t on) {
    Kokkos::deep_copy(v.on, on);
    const gated::Scope scope(v.on.data());
    const Kokkos::View<double *> x = v.x;
    const Kokkos::View<int64_t *> scanned = v.scanned;
    gated::parallel_for("gated_add", 4, KOKKOS_LAMBDA(const int i) { x(i) += 1.0 + i; });
    gated::parallel_reduce("gated_max", 4, KOKKOS_LAMBDA(const int i, double & m) { m = Kokkos::fmax(m, x(i)); },
                           Kokkos::Max<double, Memory>(Kokkos::View<double, Memory>(v.max.data())));
    gated::parallel_reduce("gated_sum", 4, KOKKOS_LAMBDA(const int i, double & s) { s += x(i); },
                           Kokkos::View<double, Memory>(v.sum.data()));
    gated::parallel_reduce("gated_columns", 5u, ColumnSums{3, v.total}, v.columns);
    gated::parallel_scan("gated_scan", 4, KOKKOS_LAMBDA(const int i, int64_t & s, const bool final) {
        if (final) scanned(i) = s;
        s += 1 + i;
    });
    Kokkos::fence("gated_test");
}

void launch_with_level_one_scratch() {
    using Policy = Kokkos::TeamPolicy<>;
    gated::parallel_for("gated_scratch", Policy(1, 1).set_scratch_size(1, Kokkos::PerTeam(64)),
                        KOKKOS_LAMBDA(const Policy::member_type &) {});
}

std::vector<double> host(const Kokkos::View<double *> & a) {
    const auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a);
    return std::vector<double>(h.data(), h.data() + h.extent(0));
}

} // namespace

TEST(GatedTest, KernelsDoNothingAndReductionsKeepTheirDestinationsWhileTheFlagIsCleared) {
    const Views v;
    launch(v, 1);
    EXPECT_EQ(host(v.x), (std::vector<double>{1.0, 2.0, 3.0, 4.0}));
    EXPECT_EQ(host(v.max), std::vector<double>{4.0});
    EXPECT_EQ(host(v.sum), std::vector<double>{10.0});
    EXPECT_EQ(host(v.columns), (std::vector<double>{10.0, 15.0, 20.0}));
    EXPECT_EQ(host(v.total), (std::vector<double>{10.0, 15.0, 20.0}));
    const auto scanned = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v.scanned);
    EXPECT_EQ(scanned(3), 6);

    // Cleared: every destination keeps the first launch's values (an ungated reduction would write its identity)
    Kokkos::deep_copy(v.scanned, -1);
    launch(v, 0);
    EXPECT_EQ(host(v.x), (std::vector<double>{1.0, 2.0, 3.0, 4.0}));
    EXPECT_EQ(host(v.max), std::vector<double>{4.0});
    EXPECT_EQ(host(v.sum), std::vector<double>{10.0});
    EXPECT_EQ(host(v.columns), (std::vector<double>{10.0, 15.0, 20.0}));
    EXPECT_EQ(host(v.total), (std::vector<double>{10.0, 15.0, 20.0}));
    const auto untouched = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v.scanned);
    for (int i = 0; i < 4; i++) EXPECT_EQ(untouched(i), -1);
}

TEST(GatedTest, KernelsWithLevelOneScratchAreRefusedUnderAScope) {
    // A thread that does nothing exits, which would keep the level-1 scratch it holds
    const Kokkos::View<uint32_t> on("on");
    const gated::Scope scope(on.data());
    EXPECT_THROW(launch_with_level_one_scratch(), std::logic_error);
}
