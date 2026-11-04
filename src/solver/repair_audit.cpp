/**
 * @file repair_audit.cpp
 * @brief Repair audit implementation.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "repair_audit.h"

#include <array>

#include "comm.h"

size_t RepairAudit::add_site(const std::string & name, const std::vector<std::string> & components,
                             const uint32_t n_cells) {
    Site site;
    site.name = name;
    site.components = components;
    site.recorder.change = Kokkos::View<double *>("repair_" + name, components.size());
    site.recorder.activations = Kokkos::View<uint64_t[1]>("repair_" + name + "_activations");
    site.recorder.touched = Kokkos::View<uint8_t *>("repair_" + name + "_cells", n_cells);
    site.last_change.assign(components.size(), 0.0);
    sites.push_back(std::move(site));
    return sites.size() - 1;
}

void RepairAudit::touch(const size_t site, const Kokkos::View<uint8_t *> & flags) const {
    const Kokkos::View<uint8_t *> touched = sites[site].recorder.touched;
    Kokkos::parallel_for("repair_touch", flags.extent(0), KOKKOS_LAMBDA(const uint32_t c) {
        if (flags(c)) touched(c) = 3;
    });
}

std::vector<double> RepairAudit::run_change(const size_t site) {
    const auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sites[site].recorder.change);
    std::vector<double> v(h.data(), h.data() + h.extent(0));
    comm::allreduce(std::span<double>(v), comm::Op::SUM);
    return v;
}

void RepairAudit::report(const uint32_t n_owned, std::vector<health::Repair> & interval,
                         std::vector<health::Repair> & run) {
    interval.clear();
    run.clear();
    for (Site & site : sites) {
        const std::vector<double> total = run_change(static_cast<size_t>(&site - sites.data()));
        uint64_t activations = 0;
        Kokkos::deep_copy(activations, Kokkos::subview(site.recorder.activations, 0));
        const Kokkos::View<uint8_t *> touched = site.recorder.touched;
        uint64_t now = 0, ever = 0;
        Kokkos::parallel_reduce("repair_cells", n_owned, KOKKOS_LAMBDA(const uint32_t c, uint64_t & a, uint64_t & b) {
            a += touched(c) & 1u;
            b += (touched(c) >> 1) & 1u;
        }, now, ever);
        Kokkos::parallel_for("repair_cells_reset", n_owned, KOKKOS_LAMBDA(const uint32_t c) {
            touched(c) = static_cast<uint8_t>(touched(c) & 2u);
        });
        const auto counts = comm::allreduce(std::array<uint64_t, 3>{activations, now, ever}, comm::Op::SUM);
        health::Repair i, r;
        i.site = r.site = site.name;
        i.activations = counts[0] - site.last_activations;
        r.activations = counts[0];
        i.cells = counts[1];
        r.cells = counts[2];
        std::vector<double> delta(total.size());
        for (size_t k = 0; k < total.size(); k++) {
            delta[k] = total[k] - site.last_change[k];
            i.change.emplace_back(site.components[k], delta[k]);
            r.change.emplace_back(site.components[k], total[k]);
        }
        if (site.derive) {
            for (const auto & c : site.derive(delta)) i.change.push_back(c);
            for (const auto & c : site.derive(total)) r.change.push_back(c);
        }
        site.last_change = total;
        site.last_activations = counts[0];
        interval.push_back(std::move(i));
        run.push_back(std::move(r));
    }
}
