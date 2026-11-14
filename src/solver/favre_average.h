/**
 * @file favre_average.h
 * @brief Favre (density-weighted) time averages per cell, for mallard post average.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef FAVRE_AVERAGE_H
#define FAVRE_AVERAGE_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "common_typedef.h"
#include "data.h"
#include "statistics.h"

/**
 * @brief Favre means a~ = <rho a> / <rho> and covariances <rho a'' b''> / <rho>
 *        per owned cell, over samples weighted by w (the time since the
 *        previous sample, as Statistics weighs them) times the cell's density:
 *        the update of Statistics with a weight per cell, so exact, the triple
 *        correlations <rho' a' b'> included.
 */
class FavreAverage {
    public:
        /** @brief Fields by name (as [statistics] fields, derived ones among the columns of derived), and products as pairs of indices into them. */
        FavreAverage(const std::vector<std::string> & fields, const std::vector<std::pair<uint32_t, uint32_t>> & products,
                     const std::vector<std::string> & species_names, const derived::Selection & derived,
                     uint32_t n_cells);

        /** @brief Add a sample of weight w > 0 of cells [0, n_owned). */
        void sample(rtype w, const CellSampler & sampler, uint32_t n_owned);
        void copy_device_to_host();

        /** @brief FAVRE_MEAN_<field>..., FAVRE_COV_<a>_<b>... */
        std::vector<std::string> variables() const;
        void register_data(std::vector<Data> & data) const;

        /** @brief Host copies (see copy_device_to_host): (cell, field) means, (cell, product) covariances. */
        const auto & host_means() const { return h_mean; }
        const auto & host_covariances() const { return h_cov; }

    private:
        std::vector<std::string> fields;
        std::vector<std::pair<uint32_t, uint32_t>> products;
        Kokkos::View<int32_t *> codes;
        Kokkos::View<uint32_t *[2]> pairs;
        Kokkos::View<rtype *> weight;  // per cell: the sum of w rho
        Kokkos::View<rtype **, Kokkos::LayoutLeft> mean;
        Kokkos::View<rtype **, Kokkos::LayoutLeft> cov;
        Kokkos::View<rtype **, Kokkos::LayoutLeft>::host_mirror_type h_mean;
        Kokkos::View<rtype **, Kokkos::LayoutLeft>::host_mirror_type h_cov;
};

#endif // FAVRE_AVERAGE_H
