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

/** @brief Range policy for register-heavy kernels. */
template <typename... Traits>
using HeavyRange = Kokkos::RangePolicy<HeavyBounds, Traits...>;

#endif // LAUNCH_BOUNDS_H
