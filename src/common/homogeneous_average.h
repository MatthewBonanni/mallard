/**
 * @file homogeneous_average.h
 * @brief Averages of cell fields over homogeneous directions, as functions of the other
 *        directions: the plane (or line) averages of mallard post average.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef HOMOGENEOUS_AVERAGE_H
#define HOMOGENEOUS_AVERAGE_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "common_typedef.h"

namespace homogeneous {

/** @brief A field of the owned cells to average. */
struct Column {
    std::string name;
    std::vector<double> values;
    /** @brief Weighted by the cell's mean density as well as its weight: a Favre mean or covariance. */
    bool favre = false;
    /**
     * @brief A covariance <a'b'>: the indices of the columns of its two means (of its weighting),
     *        or -1 to average it as it is. With them it becomes the covariance over time and the
     *        averaged directions together, avg(cov + a b) - avg(a) avg(b): the time covariance plus
     *        the dispersive part, the spatial variation of the time means.
     */
    int32_t mean_a = -1, mean_b = -1;
};

/** @brief The averages: one row per position in the kept directions, in increasing order. */
struct Profile {
    std::vector<std::string> axes;  ///< the kept directions: "x", "y", "z"
    std::vector<double> coords;     ///< per row, its weighted mean cell center in each kept direction
    std::vector<uint64_t> n_cells;  ///< per row, the cells averaged
    std::vector<std::string> names;
    std::vector<double> values;  ///< row-major, names.size() per row

    size_t rows() const { return n_cells.size(); }
    /** @brief The columns as CSV: the kept directions, n_cells, then each name; full precision. */
    std::string csv() const;
};

/**
 * @brief Averages columns over the directions of averaged ("xz": over x and z) as functions of
 *        the others. Cells whose centers agree in every kept direction, to rtol of that
 *        direction's extent, form a row; each weighs weights[c] (its volume, times its fraction
 *        of a region), and a Favre column weights[c] density[c]. Collective; the profile is
 *        complete on rank 0 only. Throws std::runtime_error naming the option for a direction
 *        this build does not have, or none left.
 */
Profile average(const std::vector<std::array<double, N_DIM>> & centers, const std::vector<double> & weights,
                const std::vector<double> & density, const std::vector<Column> & columns,
                const std::string & averaged, double rtol = 1e-6);

} // namespace homogeneous

#endif // HOMOGENEOUS_AVERAGE_H
