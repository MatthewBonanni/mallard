/**
 * @file lock_arrays.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Copies of the device atomics' lock arrays that every translation
 *        unit holds in CUDA builds, made before a stream capture.
 * @version 0.1
 * @date 2026-11-12
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef LOCK_ARRAYS_H
#define LOCK_ARRAYS_H

#include <Kokkos_Macros.hpp>

#ifdef KOKKOS_ENABLE_CUDA
#include <desul/atomics/Lock_Array.hpp>
#endif

/**
 * Without relocatable device code each translation unit has its own device
 * copy of desul's lock-array pointers, which Kokkos sets with a synchronous
 * copy on the legacy stream at the unit's first kernel launch, once: made
 * inside a stream capture, that copy fails and invalidates the capture without
 * an error until cudaStreamEndCapture, and the unit's pointers are never set.
 * Each unit that includes this header registers its own copy, which
 * copy_all() makes, checked, before a capture.
 */
namespace lock_arrays {

#if defined(KOKKOS_ENABLE_CUDA) && !defined(DESUL_ATOMICS_ENABLE_CUDA_SEPARABLE_COMPILATION)
using UnitCopy = cudaError_t (*)();

bool register_unit(UnitCopy copy);

namespace {
cudaError_t copy_unit() {
    cudaError_t err = cudaMemcpyToSymbol(desul::Impl::CUDA_SPACE_ATOMIC_LOCKS_DEVICE,
                                         &desul::Impl::CUDA_SPACE_ATOMIC_LOCKS_DEVICE_h, sizeof(int32_t *));
    if (err == cudaSuccess) {
        err = cudaMemcpyToSymbol(desul::Impl::CUDA_SPACE_ATOMIC_LOCKS_NODE, &desul::Impl::CUDA_SPACE_ATOMIC_LOCKS_NODE_h,
                                 sizeof(int32_t *));
    }
    return err;
}
[[maybe_unused]] const bool unit_registered = register_unit(&copy_unit);
} // namespace
#endif

/** @brief Sets every registered unit's lock-array pointers; outside a stream capture. Throws on a CUDA error. */
void copy_all();

} // namespace lock_arrays

#endif // LOCK_ARRAYS_H
