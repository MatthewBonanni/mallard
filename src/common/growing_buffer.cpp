/**
 * @file growing_buffer.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Device memory that grows in place (CUDA virtual memory management).
 * @version 0.1
 * @date 2026-10-05
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "growing_buffer.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#include <Kokkos_Core.hpp>

#ifdef KOKKOS_ENABLE_CUDA
#include <cuda.h>
#include <cudaTypedefs.h>
#endif

#ifdef KOKKOS_ENABLE_CUDA

namespace {

// The driver API through the runtime's entry points, so that nothing links libcuda
struct Driver {
    PFN_cuMemAddressReserve address_reserve = nullptr;
    PFN_cuMemAddressFree address_free = nullptr;
    PFN_cuMemCreate create = nullptr;
    PFN_cuMemRelease release = nullptr;
    PFN_cuMemMap map = nullptr;
    PFN_cuMemUnmap unmap = nullptr;
    PFN_cuMemSetAccess set_access = nullptr;
    PFN_cuMemGetAllocationGranularity granularity = nullptr;
};

template <typename F>
void entry_point(const char * name, F & f) {
    void * p = nullptr;
    cudaDriverEntryPointQueryResult status;
#if CUDART_VERSION >= 12050
    const cudaError_t err = cudaGetDriverEntryPointByVersion(name, &p, 12000, cudaEnableDefault, &status);
#else
    const cudaError_t err = cudaGetDriverEntryPoint(name, &p, cudaEnableDefault, &status);
#endif
    if (err != cudaSuccess || status != cudaDriverEntryPointSuccess || p == nullptr) {
        throw std::runtime_error(std::string("GrowingBuffer: no driver entry point ") + name);
    }
    f = reinterpret_cast<F>(p);
}

const Driver & driver() {
    static const Driver d = [] {
        Driver d;
        entry_point("cuMemAddressReserve", d.address_reserve);
        entry_point("cuMemAddressFree", d.address_free);
        entry_point("cuMemCreate", d.create);
        entry_point("cuMemRelease", d.release);
        entry_point("cuMemMap", d.map);
        entry_point("cuMemUnmap", d.unmap);
        entry_point("cuMemSetAccess", d.set_access);
        entry_point("cuMemGetAllocationGranularity", d.granularity);
        return d;
    }();
    return d;
}

void check(const CUresult r, const char * what) {
    if (r != CUDA_SUCCESS) throw std::runtime_error(std::string("GrowingBuffer: ") + what + " failed (" + std::to_string(r) + ")");
}

CUmemAllocationProp properties(const int device) {
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    return prop;
}

} // namespace

bool GrowingBuffer::supported() { return true; }

GrowingBuffer::GrowingBuffer(const size_t max_bytes) {
    if (cudaGetDevice(&device) != cudaSuccess) throw std::runtime_error("GrowingBuffer: cudaGetDevice failed");
    const CUmemAllocationProp prop = properties(device);
    check(driver().granularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED), "cuMemGetAllocationGranularity");
    reserved = (max_bytes + granularity - 1) / granularity * granularity;
    CUdeviceptr ptr = 0;
    check(driver().address_reserve(&ptr, reserved, 0, 0, 0), "cuMemAddressReserve");
    base = ptr;
}

GrowingBuffer::~GrowingBuffer() {
    // Freed while the device is idle, as cudaFree would
    Kokkos::fence("GrowingBuffer");
    if (mapped > 0) driver().unmap(base, mapped);
    for (const auto h : handles) driver().release(h);
    if (base != 0) driver().address_free(base, reserved);
}

void GrowingBuffer::ensure(const size_t bytes) {
    if (bytes <= mapped) return;
    if (bytes > reserved) throw std::length_error("GrowingBuffer: more than the reserved size");
    // Grow by at least a quarter, in whole granules, so that a table filled in
    // many pieces maps a few large blocks
    size_t size = std::max(bytes - mapped, mapped / 4);
    size = std::min((size + granularity - 1) / granularity * granularity, reserved - mapped);
    const CUmemAllocationProp prop = properties(device);
    CUmemGenericAllocationHandle handle;
    check(driver().create(&handle, size, &prop, 0), "cuMemCreate");
    handles.push_back(handle);
    check(driver().map(base + mapped, size, 0, handle, 0), "cuMemMap");
    CUmemAccessDesc access = {};
    access.location = prop.location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    check(driver().set_access(base + mapped, size, &access, 1), "cuMemSetAccess");
    mapped += size;
}

#else

bool GrowingBuffer::supported() { return false; }

GrowingBuffer::GrowingBuffer(size_t) { throw std::logic_error("GrowingBuffer: CUDA builds only"); }

GrowingBuffer::~GrowingBuffer() {}

void GrowingBuffer::ensure(size_t) {}

#endif
