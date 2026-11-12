/**
 * @file device_sums.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief The exact sums of exact_sums.h kept on the device: enqueued without
 *        host work, so a CUDA graph can capture them (across ranks with NCCL).
 * @version 0.1
 * @date 2026-11-10
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef DEVICE_SUMS_H
#define DEVICE_SUMS_H

#include <cstdint>

#include <Kokkos_Core.hpp>

/**
 * @brief Sums over all ranks of columns of per-cell terms, equal bit for bit
 *        to exact_sums: each term rounded to 2^-b of the column's largest,
 *        summed in 64-bit integers.
 */
class DeviceSums {
    public:
        /** @brief For n_columns columns over n_owned cells per rank. Collective if distributed. */
        DeviceSums(uint32_t n_columns, uint32_t n_owned, bool distributed);

        /**
         * @brief totals() = the column sums of terms (cell, column). Across
         *        ranks with NCCL on the default stream if nccl (which needs
         *        an NCCL build), otherwise with MPI through the host.
         */
        void sum(const Kokkos::View<const double **, Kokkos::LayoutLeft> & terms, bool nccl) const;

        const Kokkos::View<double *> & totals() const { return m_totals; }

    private:
        uint32_t n_columns = 0, n_owned = 0;
        bool distributed = false;
        double unit = 1.0;
        Kokkos::View<double *> largest;
        Kokkos::View<int64_t *> sums;
        Kokkos::View<double *> m_totals;

        template <typename T>
        void reduce(const Kokkos::View<T *> & values, bool max, bool nccl) const;
};

#endif // DEVICE_SUMS_H
