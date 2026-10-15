/**
 * @file remap.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Conservative, high-order remap of cell averages between meshes:
 *        each source cell's least-squares polynomial integrated exactly over
 *        its intersections with the target cells (see docs/design/remap.md).
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef REMAP_H
#define REMAP_H

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <toml.hpp>

#include "box_tree.h"
#include "data_writer.h"
#include "overlap.h"

class Mesh;

namespace remap {

constexpr uint8_t MAX_DEGREE = 5;

struct Options {
    uint8_t degree = 2;         // of the source polynomials
    bool limit = true;          // keep them within the averages of their vertex neighbors
    bool axisymmetric = false;  // averages weighted by r = y, as cell_measure
};

/** @brief What a remap did, summed over the ranks' owned cells. */
struct Report {
    uint64_t n_source = 0, n_target = 0;
    uint64_t n_partial = 0;      // target cells the source mesh covers only in part
    uint64_t n_nearest = 0;      // target cells it misses, which take the nearest source cell's average
    double max_uncovered = 0.0;  // largest uncovered fraction of a target cell
    uint64_t n_reduced = 0;      // source cells whose stencils allowed only a lower degree
    uint64_t n_limited = 0;      // source cells whose polynomials the limiter scaled down
    std::vector<double> source_total, target_total;  // integral of each variable
};

/** @brief Halo layers a source mesh needs for polynomials of this degree. */
int halo_layers(uint8_t degree);

/**
 * @brief The source mesh described by the [mesh] table and [[periodic]]
 *        entries of input, partitioned over the ranks, with halo_layers(degree)
 *        layers (collective).
 */
std::shared_ptr<Mesh> source_mesh(const toml::value & input, uint8_t degree);

/**
 * @brief Cell averages on a source mesh (owned cells and halo), as
 *        polynomials of the cells the rank owns.
 */
class Source {
    public:
        /**
         * @param values [variable][local cell] averages on every local cell.
         * @param groups Per variable, a group id: variables of the same group
         *        (>= 0) share one limiter factor, so that linear relations
         *        between them (e.g. sum of rho Y_k = rho) survive; -1 alone.
         */
        Source(std::shared_ptr<const Mesh> mesh, const std::vector<std::vector<double>> & values,
               const std::vector<int> & groups, const Options & options);

        /**
         * @brief Averages [variable][cell] of the first n cells of target
         *        (collective): the integral of the source polynomials over
         *        each target cell's intersection with the source cells, over
         *        the measure of that intersection. Target cells outside the
         *        source mesh take the average of the source cell with the
         *        nearest centroid.
         */
        std::vector<std::vector<double>> to(const Mesh & target, uint32_t n, Report & report) const;

        /** @brief Value of variable v of owned source cell c's polynomial at x. */
        double evaluate(uint32_t c, uint32_t v, const overlap::Point & x) const;

    private:
        /** @brief Weighted measure and integrals of every variable over simplices inside owned cell c. */
        void integrate(uint32_t c, const std::vector<overlap::Simplex> & simplices, double * out) const;
        void fit(uint32_t c, const std::vector<std::vector<double>> & values);
        void limit(uint32_t c, const std::vector<std::vector<double>> & values, const std::vector<int> & groups);
        void monomials(uint32_t c, const overlap::Point & x, double * phi) const;

        std::shared_ptr<const Mesh> mesh;
        Options options;
        uint32_t n_owned = 0, n_vars = 0, n_max = 0;  // n_max: monomials of options.degree, constant included
        std::vector<std::array<uint8_t, 3>> exponents;  // monomial l, constant first, by total degree
        std::vector<overlap::Point> center;   // [local cell] centroid (weighted)
        std::vector<double> size;             // [local cell] measure^(1/N_DIM)
        std::vector<double> measure;          // [local cell] weighted measure
        std::vector<double> moments;          // [local cell][n_max] means of ((x - center) / size)^l
        std::vector<uint8_t> degree;          // [owned cell]
        std::vector<double> mean;             // [owned cell][var]
        std::vector<double> coefficients;     // [owned cell][var][n_max], constant slot unused
        BoxTree tree;
        std::vector<overlap::Region> regions;  // [owned cell]
        uint64_t n_reduced = 0, n_limited = 0;
};

/**
 * @brief The restart file named by [initialize] file, on the source mesh of
 *        [initialize.mesh] (and [[initialize.periodic]], else the run's
 *        [[periodic]]), remapped to every local cell of target: variables,
 *        step, time and attributes; no face records (collective).
 */
RestartData remap_restart(const toml::value & input, const Mesh & target, bool axisymmetric, Report & report);

/** @brief Text of a report for the run log. */
std::string describe(const Report & report, const std::vector<std::string> & names);

} // namespace remap

#endif // REMAP_H
