/**
 * @file lock_arrays.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Copies of the device atomics' lock arrays that every translation
 *        unit holds in CUDA builds, made before a stream capture.
 * @version 0.1
 * @date 2026-11-12
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "lock_arrays.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace lock_arrays {

#if defined(KOKKOS_ENABLE_CUDA) && !defined(DESUL_ATOMICS_ENABLE_CUDA_SEPARABLE_COMPILATION)
namespace {
std::vector<UnitCopy> & units() {
    static std::vector<UnitCopy> copies;
    return copies;
}
} // namespace

bool register_unit(const UnitCopy copy) {
    units().push_back(copy);
    return true;
}

void copy_all() {
    for (const UnitCopy copy : units()) {
        const cudaError_t err = copy();
        if (err != cudaSuccess) throw std::runtime_error(std::string("lock arrays: ") + cudaGetErrorString(err));
    }
}
#else
void copy_all() {}
#endif

} // namespace lock_arrays
