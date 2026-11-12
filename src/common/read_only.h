/**
 * @file read_only.h
 * @brief Loads through the GPU's read-only data path.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef READ_ONLY_H
#define READ_ONLY_H

#include <Kokkos_Core.hpp>

/**
 * @brief *p, on NVIDIA GPUs through the read-only (non-coherent) data path
 *        (ld.global.nc), as `const __restrict__` pointers allow in CUDA. Only
 *        for data that no thread writes during the kernel. Kokkos's
 *        RandomAccess memory trait would do this, but its View
 *        implementation (mdspan-based since Kokkos 5) ignores that trait.
 */
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION T load_read_only(const T * p) {
#ifdef __CUDA_ARCH__
    return __ldg(p);
#else
    return *p;
#endif
}

#endif // READ_ONLY_H
