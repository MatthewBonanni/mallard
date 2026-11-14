/**
 * @file repair_audit.cpp
 * @brief Repair audit implementation.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "repair_audit.h"

#include <array>
#include <span>

#include "comm.h"

size_t RepairAudit::add_site(const std::string & name, const std::vector<std::string> & components,
                             const uint32_t n_cells) {
    Site site;
    site.name = name;
    site.components = components;
    site.recorder.change = Kokkos::View<double *>("repair_" + name, components.size());
    site.recorder.activations = Kokkos::View<uint64_t[1]>("repair_" + name + "_activations");
    site.recorder.touched = Kokkos::View<uint8_t *>("repair_" + name + "_cells", n_cells);
    site.cells = Kokkos::View<RepairCells>("repair_" + name + "_cell_counts");
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

namespace {

struct CellCountFunctor {
    using value_type = RepairCells;
    Kokkos::View<uint8_t *> touched;

    KOKKOS_INLINE_FUNCTION void init(value_type & v) const { v.n[0] = v.n[1] = 0; }
    KOKKOS_INLINE_FUNCTION void join(value_type & dst, const value_type & src) const {
        dst.n[0] += src.n[0];
        dst.n[1] += src.n[1];
    }
    KOKKOS_INLINE_FUNCTION void operator()(const uint32_t c, value_type & v) const {
        v.n[0] += touched(c) & 1u;
        v.n[1] += (touched(c) >> 1) & 1u;
    }
};

} // namespace

uint32_t RepairAudit::n_values() const {
    size_t n = 0;
    for (const Site & site : sites) n += 3 + site.components.size();
    return static_cast<uint32_t>(n);
}

void RepairAudit::local_values(const uint32_t n_owned, const Kokkos::View<double *> & out) const {
    size_t offset = 0;
    for (const Site & site : sites) {
        const Kokkos::View<uint8_t *> touched = site.recorder.touched;
        const Kokkos::View<RepairCells> cells = site.cells;
        Kokkos::parallel_reduce("repair_cells", n_owned, CellCountFunctor{touched}, cells);
        Kokkos::parallel_for("repair_cells_reset", n_owned, KOKKOS_LAMBDA(const uint32_t c) {
            touched(c) = static_cast<uint8_t>(touched(c) & 2u);
        });
        const Kokkos::View<uint64_t[1]> activations = site.recorder.activations;
        const Kokkos::View<double *> change = site.recorder.change;
        const size_t first = offset;
        Kokkos::parallel_for("repair_values", 1, KOKKOS_LAMBDA(const int) {
            out(first) = static_cast<double>(activations(0));
            out(first + 1) = static_cast<double>(cells().n[0]);
            out(first + 2) = static_cast<double>(cells().n[1]);
            for (size_t k = 0; k < change.extent(0); k++) out(first + 3 + k) = change(k);
        });
        offset += 3 + site.components.size();
    }
}

void RepairAudit::report(const double * local, std::vector<health::Repair> & interval,
                         std::vector<health::Repair> & run) {
    interval.clear();
    run.clear();
    std::vector<double> values(local, local + n_values());
    comm::allreduce(std::span<double>(values), comm::Op::SUM);
    const double * v = values.data();
    for (Site & site : sites) {
        const std::array<uint64_t, 3> counts = {static_cast<uint64_t>(v[0]), static_cast<uint64_t>(v[1]),
                                                static_cast<uint64_t>(v[2])};
        const std::vector<double> total(v + 3, v + 3 + site.components.size());
        v += 3 + site.components.size();
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
