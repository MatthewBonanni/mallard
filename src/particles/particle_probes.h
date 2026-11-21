/**
 * @file particle_probes.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Particle probes ([[particles.probes]]): statistics of particle
 *        expressions over the particles inside a region.
 * @version 0.1
 * @date 2026-11-20
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef PARTICLE_PROBES_H
#define PARTICLE_PROBES_H

#include <cstdint>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "particle_quantities.h"

namespace particles {

/** @brief Statistics of one probe row: the count, then per field its mean, standard deviation, min and max. */
struct ProbeRow {
    uint64_t count = 0;
    std::vector<double> mean, std, min, max;
};

/**
 * @brief The probes of [[particles.probes]]: each samples its fields, particle
 *        expressions, over the particles inside its region every interval
 *        steps, and writes a row of statistics to its CSV file.
 */
class ParticleProbes {
    public:
        void init(const Input & table, region::Regions & regions, Quantities & quantities);

        bool empty() const { return probes.empty(); }

        /** @brief Whether some probe samples after this step. */
        bool due(uint64_t step) const;

        /** @brief Lower step_limit to the next step after step that some probe samples. */
        void due_limits(uint64_t step, uint64_t & step_limit) const;

        /**
         * @brief The statistics of the probes due at step from the particle
         *        variables: values(s, j) is field j (over every probe's fields)
         *        of slot s, NaN outside its probe or in an empty slot (collective).
         */
        std::vector<ProbeRow> reduce(uint64_t step, const Kokkos::View<double **, Kokkos::LayoutRight> & values,
                                     bool distributed) const;

        /** @brief Write the rows reduce() gave for this step, from rank 0. */
        void write(uint64_t step, double t, const std::vector<ProbeRow> & rows);

        /** @brief Fields over every probe, and their programs. */
        uint32_t n_fields() const { return static_cast<uint32_t>(field_probe.size()); }
        const kexpr::ProgramTable<> & table() const { return m_table; }

        /** @brief For each field, its probe; for each probe, its region (-1: everywhere). */
        const Kokkos::View<const uint32_t *> & fields_of() const { return m_field_probe; }
        const Kokkos::View<const int32_t *> & regions_of() const { return m_probe_region; }
        const std::vector<region::RegionView<>> & regions() const { return m_regions; }

        std::vector<std::string> files() const;
        std::vector<std::string> summary() const;

    private:
        struct Probe {
            std::string name, file;
            uint64_t interval = 1;
            std::vector<std::string> fields;
            uint32_t first = 0;  // its first field over every probe
            std::string region;
            bool started = false;  // its file has its header
        };
        std::vector<Probe> probes;
        std::vector<uint32_t> field_probe;
        std::vector<kexpr::Program> programs;
        std::vector<region::RegionView<>> m_regions;
        kexpr::ProgramTable<> m_table;
        Kokkos::View<const uint32_t *> m_field_probe;
        Kokkos::View<const int32_t *> m_probe_region;
};

} // namespace particles

#endif // PARTICLE_PROBES_H
