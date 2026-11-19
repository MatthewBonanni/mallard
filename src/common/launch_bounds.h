/**
 * @file launch_bounds.h
 * @brief Launch bounds of kernels that need many registers per thread.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef LAUNCH_BOUNDS_H
#define LAUNCH_BOUNDS_H

#include <Kokkos_Core.hpp>

/**
 * @brief HIP compiles a kernel without launch bounds for blocks of 1024
 *        threads, which caps it at 128 registers per thread and spills the
 *        rest to scratch memory; blocks of 256 threads allow 512. Other
 *        backends: no bounds.
 */
#if defined(KOKKOS_ENABLE_HIP)
using HeavyBounds = Kokkos::LaunchBounds<256, 1>;
#else
using HeavyBounds = Kokkos::LaunchBounds<>;
#endif

/**
 * @brief CUDA: four blocks of 128 threads per SM, which caps a kernel at 128
 *        registers per thread, for kernels whose spills cost less than the
 *        threads gained (the 2D TENO setup). Other backends: HeavyBounds.
 */
#if defined(KOKKOS_ENABLE_CUDA)
using OccupancyBounds = Kokkos::LaunchBounds<128, 4>;
#else
using OccupancyBounds = HeavyBounds;
#endif

/** @brief Range policy for register-heavy kernels. */
template <typename... Traits>
using HeavyRange = Kokkos::RangePolicy<HeavyBounds, Traits...>;

#endif // LAUNCH_BOUNDS_H
