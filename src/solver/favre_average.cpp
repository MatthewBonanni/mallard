/**
 * @file favre_average.cpp
 * @brief Favre (density-weighted) time averages per cell, for mallard post average.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "favre_average.h"

FavreAverage::FavreAverage(const std::vector<std::string> & fields_in,
                           const std::vector<std::pair<uint32_t, uint32_t>> & products_in,
                           const std::vector<std::string> & species_names, const derived::Selection & derived,
                           uint32_t n_cells)
    : fields(fields_in), products(products_in) {
    codes = Kokkos::View<int32_t *>("favre_codes", fields.size());
    auto h_codes = Kokkos::create_mirror_view(codes);
    for (size_t i = 0; i < fields.size(); i++) h_codes(i) = CellSampler::code(fields[i], species_names, "--fields", &derived);
    Kokkos::deep_copy(codes, h_codes);
    pairs = Kokkos::View<uint32_t *[2]>("favre_pairs", products.size());
    auto h_pairs = Kokkos::create_mirror_view(pairs);
    for (size_t p = 0; p < products.size(); p++) {
        h_pairs(p, 0) = products[p].first;
        h_pairs(p, 1) = products[p].second;
    }
    Kokkos::deep_copy(pairs, h_pairs);
    weight = Kokkos::View<rtype *>("favre_weight", n_cells);
    mean = Kokkos::View<rtype **, Kokkos::LayoutLeft>("favre_mean", n_cells, fields.size());
    cov = Kokkos::View<rtype **, Kokkos::LayoutLeft>("favre_cov", n_cells, products.size());
    h_mean = Kokkos::create_mirror_view(mean);
    h_cov = Kokkos::create_mirror_view(cov);
}

void FavreAverage::sample(rtype w, const CellSampler & sampler, uint32_t n_owned) {
    const auto W = weight;
    const auto m = mean;
    const auto C = cov;
    const auto code = codes;
    const auto pair = pairs;
    const uint32_t n_fields = static_cast<uint32_t>(fields.size()), n_products = static_cast<uint32_t>(products.size());
    Kokkos::parallel_for("favre_sample", n_owned, KOKKOS_LAMBDA(const uint32_t c) {
        const rtype w_c = w * sampler(c, CellSampler::RHO);
        W(c) += w_c;
        const rtype a = w_c / W(c);
        for (uint32_t p = 0; p < n_products; p++) {
            const uint32_t i = pair(p, 0), j = pair(p, 1);
            const rtype d_i = sampler(c, code(i)) - m(c, i);
            const rtype d_j = sampler(c, code(j)) - m(c, j);
            const rtype mean_j = m(c, j) + a * d_j;
            C(c, p) += a * (d_i * (sampler(c, code(j)) - mean_j) - C(c, p));
        }
        for (uint32_t i = 0; i < n_fields; i++) m(c, i) += a * (sampler(c, code(i)) - m(c, i));
    });
}

void FavreAverage::copy_device_to_host() {
    Kokkos::deep_copy(h_mean, mean);
    Kokkos::deep_copy(h_cov, cov);
}

std::vector<std::string> FavreAverage::variables() const {
    std::vector<std::string> names;
    for (const auto & f : fields) names.push_back("FAVRE_MEAN_" + f);
    for (const auto & [i, j] : products) names.push_back("FAVRE_COV_" + fields[i] + "_" + fields[j]);
    return names;
}

void FavreAverage::register_data(std::vector<Data> & data) const {
    const std::vector<std::string> names = variables();
    for (size_t i = 0; i < fields.size(); i++) data.emplace_back(names[i], Kokkos::subview(h_mean, Kokkos::ALL(), i));
    for (size_t p = 0; p < products.size(); p++) {
        data.emplace_back(names[fields.size() + p], Kokkos::subview(h_cov, Kokkos::ALL(), p));
    }
}
