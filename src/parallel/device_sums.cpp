/**
 * @file device_sums.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Exact sums of per-cell terms on the device.
 * @version 0.1
 * @date 2026-11-10
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "device_sums.h"

#include <cmath>
#include <span>
#include <stdexcept>
#include <type_traits>

#include "comm.h"
#include "device_comm.h"

DeviceSums::DeviceSums(const uint32_t n_columns_, const uint32_t n_owned_, const bool distributed_) :
    n_columns(n_columns_), n_owned(n_owned_), distributed(distributed_),
    largest("device_sums_largest", n_columns_), sums("device_sums", n_columns_),
    m_totals("device_sums_totals", n_columns_) {
    const uint64_t n_global =
        distributed ? comm::allreduce(static_cast<uint64_t>(n_owned), comm::Op::SUM) : static_cast<uint64_t>(n_owned);
    const int bits = 62 - static_cast<int>(std::ceil(std::log2(static_cast<double>(n_global) + 1.0)));
    unit = std::ldexp(1.0, bits);
}

template <typename T>
void DeviceSums::reduce(const Kokkos::View<T *> & values, const bool max, const bool nccl) const {
    if (!distributed) return;
    if (nccl) {
#ifdef Mallard_HAS_NCCL
        const ncclDataType_t type = std::is_same_v<T, double> ? ncclDouble : ncclInt64;
        if (ncclAllReduce(values.data(), values.data(), values.extent(0), type, max ? ncclMax : ncclSum, comm::nccl(),
                          Kokkos::DefaultExecutionSpace().cuda_stream()) != ncclSuccess) {
            throw std::runtime_error("NCCL: exact sum failed");
        }
        return;
#else
        throw std::logic_error("DeviceSums: NCCL sums need an NCCL build");
#endif
    }
    const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
    comm::allreduce(std::span<T>(host.data(), host.extent(0)), max ? comm::Op::MAX : comm::Op::SUM);
    Kokkos::deep_copy(values, host);
}

void DeviceSums::sum(const Kokkos::View<const double **, Kokkos::LayoutLeft> & terms, const bool nccl) const {
    using DeviceMemory = Kokkos::DefaultExecutionSpace::memory_space;
    for (uint32_t k = 0; k < n_columns; k++) {
        Kokkos::parallel_reduce("device_sums_largest", n_owned, KOKKOS_LAMBDA(const uint32_t c, double & m) {
            m = Kokkos::fmax(m, Kokkos::fabs(terms(c, k)));
        }, Kokkos::Max<double, DeviceMemory>(Kokkos::View<double, DeviceMemory>(largest.data() + k)));
    }
    reduce(largest, true, nccl);
    const Kokkos::View<double *> m = largest;
    const double u = unit;
    for (uint32_t k = 0; k < n_columns; k++) {
        Kokkos::parallel_reduce("device_sums", n_owned, KOKKOS_LAMBDA(const uint32_t c, int64_t & sum) {
            if (m(k) > 0.0) sum += static_cast<int64_t>(Kokkos::round(terms(c, k) * (u / m(k))));
        }, Kokkos::Sum<int64_t, DeviceMemory>(Kokkos::View<int64_t, DeviceMemory>(sums.data() + k)));
    }
    reduce(sums, false, nccl);
    const Kokkos::View<int64_t *> s = sums;
    const Kokkos::View<double *> totals = m_totals;
    Kokkos::parallel_for("device_sums_totals", n_columns, KOKKOS_LAMBDA(const uint32_t k) {
        totals(k) = m(k) > 0.0 ? static_cast<double>(s(k)) / (u / m(k)) : 0.0;
    });
}
