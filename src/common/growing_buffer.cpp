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

// The driver API through the runtime's entry points, so that nothing links libcuda. The versioned
// typedefs: CUDA 13's cudaTypedefs.h no longer defines the unversioned names
struct Driver {
    PFN_cuMemAddressReserve_v10020 address_reserve = nullptr;
    PFN_cuMemAddressFree_v10020 address_free = nullptr;
    PFN_cuMemCreate_v10020 create = nullptr;
    PFN_cuMemRelease_v10020 release = nullptr;
    PFN_cuMemMap_v10020 map = nullptr;
    PFN_cuMemUnmap_v10020 unmap = nullptr;
    PFN_cuMemSetAccess_v10020 set_access = nullptr;
    PFN_cuMemGetAllocationGranularity_v10020 granularity = nullptr;
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
    static const Driver entries = [] {
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
    return entries;
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
    size_t offset = 0;
    for (size_t k = 0; k < handles.size(); k++) {
        driver().unmap(base + offset, sizes[k]);
        driver().release(handles[k]);
        offset += sizes[k];
    }
    if (base != 0) driver().address_free(base, reserved);
}

void GrowingBuffer::ensure(const size_t bytes) {
    if (bytes <= mapped) return;
    if (bytes > reserved) throw std::length_error("GrowingBuffer: more than the reserved size");
    const size_t size = std::min((bytes - mapped + granularity - 1) / granularity * granularity, reserved - mapped);
    const CUmemAllocationProp prop = properties(device);
    CUmemGenericAllocationHandle handle;
    check(driver().create(&handle, size, &prop, 0), "cuMemCreate");
    handles.push_back(handle);
    sizes.push_back(size);
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
