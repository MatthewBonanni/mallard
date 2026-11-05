/**
 * @file mesh_quality.h
 * @brief Cell quality metrics of a finite volume mesh.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef MESH_QUALITY_H
#define MESH_QUALITY_H

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mesh.h"

/**
 * @brief Per owned cell, the worst over its faces of:
 *        - non-orthogonality: angle in degrees between the face normal and the
 *          line joining the centroids of the two cells (cell to face centroid
 *          at boundaries);
 *        - skewness: distance from the face centroid to where that line
 *          crosses the face's plane, over the centroid distance (0 where the
 *          line points away from the face: non-orthogonality > 90 says so);
 *        - volume ratio: larger over smaller volume of the two cells;
 *        and the aspect ratio: largest over smallest distance from the cell
 *        centroid to its face centroids.
 */
struct MeshQuality {
    static constexpr int N_METRICS = 4;
    static constexpr std::array<const char *, N_METRICS> NAMES = {"non-orthogonality", "skewness", "aspect ratio",
                                                                  "volume ratio"};
    static constexpr std::array<const char *, N_METRICS> FIELDS = {"NON_ORTHOGONALITY", "SKEWNESS", "ASPECT_RATIO",
                                                                   "VOLUME_RATIO"};
    // Values above which a cell is questionable
    static constexpr std::array<double, N_METRICS> LIMITS = {70.0, 0.5, 1000.0, 10.0};
    // Values worth listing the worst cells for
    static constexpr std::array<double, N_METRICS> NOTABLE = {10.0, 0.05, 10.0, 2.0};

    std::array<std::vector<double>, N_METRICS> cells;  // [metric][owned cell]
};

MeshQuality mesh_quality(const Mesh & mesh);

struct QualityCell {
    double value = 0.0;
    uint64_t cell = 0;  // global id
    std::array<double, N_DIM> x{};
};

struct QualitySummary {
    double max = 0.0;
    double mean = 0.0;
    uint64_t n_over_limit = 0;
    std::vector<QualityCell> worst;  // largest first
};

/** @brief Over all ranks: largest, mean, cells over the limit and the n_worst worst cells of each metric. */
std::array<QualitySummary, MeshQuality::N_METRICS> summarize(const Mesh & mesh, const MeshQuality & quality,
                                                              size_t n_worst);

/** @brief A point as the quality report prints it: "(0.5, 1.25)". */
std::string quality_point(const std::array<double, N_DIM> & x);

/**
 * @brief The quality report of mallard check and mallard info: per metric its
 *        maximum, mean and cells above the limit, and the worst cells where it
 *        is notable; key-value lines of the current section.
 */
void print_quality(const std::array<QualitySummary, MeshQuality::N_METRICS> & summary);

/**
 * @brief Writes the quality fields, then the extra (name, per-owned-cell values)
 *        fields, as a snapshot PREFIX_000000 in the default format (collective).
 * @return The file written.
 */
std::string write_quality(std::shared_ptr<Mesh> mesh, const MeshQuality & quality, const std::string & prefix,
                          const std::vector<std::pair<std::string, std::vector<double>>> & extra = {});

#endif // MESH_QUALITY_H
