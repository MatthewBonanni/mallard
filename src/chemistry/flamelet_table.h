/**
 * @file flamelet_table.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tabulated chemistry: Mallard's table format (.mtab, written by
 *        tools/setup/flamelet_table.py), multilinear lookup on the host and in
 *        device kernels, and the flamelet/progress-variable (FPV) map from the
 *        transported mean mixture fraction, its variance and the mean progress
 *        variable to a table point (docs/design/flamelet.md).
 * @version 0.1
 * @date 2026-11-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef FLAMELET_TABLE_H
#define FLAMELET_TABLE_H

#include <cstdint>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

namespace flamelet {

/** @brief Axes of a table, at most (Z, its variance, the progress variable and an enthalpy defect). */
constexpr uint32_t MAX_AXES = 4;

/** @brief A table coordinate: per axis, the lower node of its cell and the fraction t in [0, 1] across it. */
struct Point {
    uint32_t lo[MAX_AXES] = {0, 0, 0, 0};
    double t[MAX_AXES] = {0.0, 0.0, 0.0, 0.0};
};

/**
 * @brief A table in one memory space, copyable by value into kernels.
 *        Values are node-major (last axis fastest) with a node's fields
 *        contiguous, so one lookup of several fields reads 2^axes short runs.
 *        Results depend only on the query, never on the backend or the order
 *        of evaluation, so lookups are bitwise reproducible across ranks.
 */
template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
struct TableView {
    uint32_t n_axes = 0;
    uint32_t n_fields = 0;
    uint32_t n_nodes[MAX_AXES] = {1, 1, 1, 1};
    uint32_t axis_first[MAX_AXES] = {0, 0, 0, 0};  // axis a's nodes at axes(axis_first[a] ...)
    Kokkos::View<const double *, MemorySpace> axes;
    Kokkos::View<const double *, MemorySpace> values;  // indexed in 64 bits: tables may exceed 2^31 values

    /** @brief The cell of axis a holding x, and the fraction across it; clamped to the axis' ends. */
    KOKKOS_INLINE_FUNCTION void locate_axis(const uint32_t a, const double x, uint32_t & lo, double & t) const {
        const double * v = axes.data() + axis_first[a];
        const uint32_t n = n_nodes[a];
        if (!(x > v[0])) {  // also NaN, which then reads the first node rather than garbage
            lo = 0;
            t = 0.0;
            return;
        }
        if (x >= v[n - 1]) {
            lo = n - 2;
            t = 1.0;
            return;
        }
        uint32_t l = 0, h = n - 1;  // v[l] <= x < v[h]
        while (h - l > 1) {
            const uint32_t m = l + (h - l) / 2;
            if (v[m] <= x) {
                l = m;
            } else {
                h = m;
            }
        }
        lo = l;
        t = (x - v[l]) / (v[l + 1] - v[l]);
    }

    /** @brief The table point of the coordinates q[0, n_axes). */
    KOKKOS_INLINE_FUNCTION Point locate(const double * q) const {
        Point p;
        for (uint32_t a = 0; a < n_axes; a++) locate_axis(a, q[a], p.lo[a], p.t[a]);
        return p;
    }

    /** @brief Field f at point p, interpolated multilinearly over the 2^n_axes nodes of its cell. */
    KOKKOS_INLINE_FUNCTION double operator()(const Point & p, const uint32_t f) const {
        double v = 0.0;
        evaluate(p, &f, 1, &v);
        return v;
    }

    /** @brief Fields fields[0, n) at point p into out[0, n), the corner weights computed once. */
    KOKKOS_INLINE_FUNCTION void evaluate(const Point & p, const uint32_t * fields, const uint32_t n, double * out) const {
        for (uint32_t i = 0; i < n; i++) out[i] = 0.0;
        const uint32_t corners = 1u << n_axes;
        for (uint32_t k = 0; k < corners; k++) {
            uint64_t node = 0;
            double w = 1.0;
            for (uint32_t a = 0; a < n_axes; a++) {
                const uint32_t up = (k >> (n_axes - 1 - a)) & 1u;
                node = node * n_nodes[a] + p.lo[a] + up;
                w *= up ? p.t[a] : 1.0 - p.t[a];
            }
            if (w == 0.0) continue;
            const double * row = values.data() + node * n_fields;
            for (uint32_t i = 0; i < n; i++) out[i] += w * row[fields[i]];
        }
    }
};

/**
 * @brief The FPV map (axes Z, S = Zvar / (Z (1 - Z)) and c, the mean progress
 *        variable normalized between the table's ends at that (Z, S)), on
 *        a TableView. Plain aggregate, captured by value in kernels.
 */
template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
struct FPV {
    TableView<MemorySpace> table;
    uint32_t field_C = 0;

    /** @brief Normalized variance of Z, clamped to [0, 1] (0 where Z (1 - Z) vanishes). */
    KOKKOS_INLINE_FUNCTION static double normalized_variance(const double Z, const double Zvar) {
        const double m = Z * (1.0 - Z);
        if (!(m > 1e-12)) return 0.0;
        const double s = Zvar / m;
        return s < 0.0 ? 0.0 : (s > 1.0 ? 1.0 : s);
    }

    /**
     * @brief The table point of a mean mixture fraction Z, its variance Zvar
     *        and the mean progress variable C. C between its table bounds
     *        at (Z, S) is reproduced exactly by the C field; outside, it is
     *        clamped to them.
     */
    KOKKOS_INLINE_FUNCTION Point locate(const double Z, const double Zvar, const double C) const {
        Point p;
        table.locate_axis(0, Z, p.lo[0], p.t[0]);
        table.locate_axis(1, normalized_variance(Z, Zvar), p.lo[1], p.t[1]);
        const uint32_t n_c = table.n_nodes[2];
        p.lo[2] = 0;
        p.t[2] = 0.0;
        const double C_min = table(p, field_C);
        p.lo[2] = n_c - 2;
        p.t[2] = 1.0;
        const double C_max = table(p, field_C);
        const double span = C_max - C_min;
        const double c = span > 0.0 ? (C - C_min) / span : 0.0;
        table.locate_axis(2, c, p.lo[2], p.t[2]);
        return p;
    }
};

/**
 * @brief A table read from a .mtab file: one line "mallard-table 1", one line
 *        of JSON (axes with their nodes, fields with their units, the model
 *        and provenance), then the values as little-endian float64.
 */
class Table {
    public:
        /** @brief An axis: its name and nodes (strictly increasing). */
        struct Axis {
            std::string name;
            std::vector<double> nodes;
        };

        /** @brief A field: its name and units (UDUNITS syntax). */
        struct Field {
            std::string name;
            std::string units;
        };

        /**
         * @brief Read and check a table file.
         * @throws std::runtime_error naming the file and what is wrong with it
         */
        static Table read(const std::string & file);

        /** @brief A table from its parts (tests and converters); checked as read() checks a file. */
        Table(std::string model, std::vector<Axis> axes, std::vector<Field> fields, std::vector<double> values,
              const std::string & where = "table");

        const std::string & model() const { return m_model; }
        const std::vector<Axis> & axes() const { return m_axes; }
        const std::vector<Field> & fields() const { return m_fields; }
        const std::vector<double> & values() const { return m_values; }
        /** @brief The header's JSON text, provenance included, as read. */
        const std::string & header() const { return m_header; }

        /** @brief Index of the named field, or -1. */
        int32_t field(const std::string & name) const;

        /** @brief The table in host memory (views of this object's vectors). */
        TableView<Kokkos::HostSpace> host() const;
        /** @brief The table copied to the default device's memory. */
        TableView<> device() const;

        /**
         * @brief The FPV map of a table of model "fpv" with axes Z, S and c and a field C.
         * @throws std::runtime_error if it is not one
         */
        FPV<> fpv() const;
        /** @brief As fpv(), in host memory. */
        FPV<Kokkos::HostSpace> fpv_host() const;

    private:
        Table() = default;
        void finish(const std::string & where);  // checks, and gathers the axes' nodes
        uint32_t fpv_field(const std::string & where) const;

        std::string m_model;
        std::vector<Axis> m_axes;
        std::vector<Field> m_fields;
        std::vector<double> m_values;
        std::vector<double> m_axis_nodes;  // every axis' nodes, in order
        std::string m_header;
};

} // namespace flamelet

#endif // FLAMELET_TABLE_H
