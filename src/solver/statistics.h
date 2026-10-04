/**
 * @file statistics.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief In-situ running time averages and probes.
 * @version 0.1
 * @date 2026-10-04
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef STATISTICS_H
#define STATISTICS_H

#include <array>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>
#include <toml.hpp>

#include "common_typedef.h"
#include "data.h"
#include "data_writer.h"
#include "mesh.h"
#include "state.h"

/**
 * @brief Cell values that statistics and probes sample, by code: 0 is RHO,
 *        1 + i the primitive PRIMITIVE_NAMES[i], 1 + N_PRIMITIVE + k the mass
 *        fraction Y_k of species k. The primitives must be current.
 */
struct CellSampler {
    StateView U;
    SpeciesView rhoY;
    Kokkos::View<rtype *[N_PRIMITIVE]> P;

    static constexpr int32_t RHO = 0;
    static constexpr int32_t FIRST_PRIMITIVE = 1;
    static constexpr int32_t FIRST_SPECIES = 1 + N_PRIMITIVE;

    KOKKOS_INLINE_FUNCTION
    rtype operator()(const uint32_t c, const int32_t code) const {
        if (code == RHO) return U(c, 0);
        if (code < FIRST_SPECIES) return P(c, code - FIRST_PRIMITIVE);
        return rhoY(c, code - FIRST_SPECIES) / U(c, 0);
    }

    /** @brief Code of a variable name (RHO, a primitive, Y_<species>); throws InputError naming key. */
    static int32_t code(const std::string & name, const std::vector<std::string> & species_names,
                        const std::string & key);
};

/**
 * @brief Running time averages per owned cell: means of fields and
 *        covariances <a'b'> of pairs of them, updated on the device every
 *        `interval` steps once t > t_start. A sample at time t has weight
 *        t - t_prev (the previous sample's time, or t_start), and with
 *        a = w / W over the total weight W the update is (West 1979)
 *            d_a = a_new - mean_a,  mean_a += a d_a,
 *            cov_ab += a (d_a (b_new - mean_b,new) - cov_ab),
 *        so each output field is the current average itself. Restart files
 *        carry the averages and the weights, so a restarted run continues them
 *        bitwise.
 */
class Statistics {
    public:
        /** @brief Read [statistics] (if any) and allocate n_cells. */
        void init(const toml::value & input, const std::vector<std::string> & species_names, uint32_t n_cells);

        bool enabled() const { return !fields.empty(); }
        bool due(uint64_t step, rtype t) const { return enabled() && step % interval == 0 && t > t_start; }

        /** @brief Add the sample of time t of cells [0, n_owned). */
        void sample(rtype t, const CellSampler & sampler, uint32_t n_owned);

        /** @brief Output variables: MEAN_<field>..., COV_<a>_<b>... */
        std::vector<std::string> variables() const;
        void register_data(std::vector<Data> & data) const;
        void copy_device_to_host();

        /**
         * @brief Continue the averages of a restart file, or start afresh at
         *        time t if it holds none (or with reset); throws if it holds
         *        only some of them.
         */
        void restore(const RestartData & restart, const std::string & file, rtype t);
        /** @brief Start afresh at time t (first sample weighted from max(t, t_start)). */
        void start(rtype t);
        RestartAttributes attributes() const;

        std::string summary() const;
        uint64_t samples() const { return n_samples; }
        /** @brief Host copies (see copy_device_to_host): (cell, field) means, (cell, product) covariances. */
        const auto & host_means() const { return h_mean; }
        const auto & host_covariances() const { return h_cov; }

    private:
        std::vector<std::string> fields;
        std::vector<std::pair<uint32_t, uint32_t>> products;  // indices into fields
        uint64_t interval = 1;
        rtype t_start = 0.0;
        bool reset = false;

        Kokkos::View<int32_t *> codes;
        Kokkos::View<uint32_t *[2]> pairs;
        Kokkos::View<rtype **, Kokkos::LayoutLeft> mean;
        Kokkos::View<rtype **, Kokkos::LayoutLeft> cov;
        Kokkos::View<rtype **, Kokkos::LayoutLeft>::host_mirror_type h_mean;
        Kokkos::View<rtype **, Kokkos::LayoutLeft>::host_mirror_type h_cov;

        rtype weight = 0.0;
        rtype t_last = 0.0;
        uint64_t n_samples = 0;
};

/**
 * @brief Point and line probes: the values of the cells holding a set of
 *        points, written every `interval` steps by rank 0 as rows
 *        step,t,point,x,y(,z),variables... Each point takes the cell average
 *        of the cell containing it (the lowest global id on a shared face),
 *        or of the nearest cell centroid if no cell contains it.
 */
class Probes {
    public:
        /** @brief Read [[probes]] and locate the points; resume appends to existing files. */
        void init(const toml::value & input, const Mesh & mesh, const std::vector<std::string> & species_names,
                  bool resume);

        bool due(uint64_t step) const;
        void write(uint64_t step, rtype t, const CellSampler & sampler);

        std::vector<std::pair<std::string, std::string>> summary() const;
        std::vector<std::string> files() const;

    private:
        struct Set {
            std::string name;
            std::string file;
            uint64_t interval = 1;
            std::vector<std::string> variables;
            std::vector<std::array<double, N_DIM>> points;
            std::vector<uint32_t> local_points;  // points held by owned cells of this rank
            Kokkos::View<uint32_t *> cells;      // their cells
            Kokkos::View<int32_t *> codes;
            Kokkos::View<rtype **, Kokkos::LayoutRight> values;  // (local point, variable)
            std::shared_ptr<std::ofstream> out;
        };
        std::vector<Set> sets;
};

/**
 * @brief Owned cell of each point on this rank, or -1 where another rank (or
 *        none) holds it: the cell containing the point, the lowest global id
 *        among several, else the nearest cell centroid over all ranks.
 */
std::vector<int32_t> locate_points(const Mesh & mesh, const std::vector<std::array<double, N_DIM>> & points);

#endif // STATISTICS_H
