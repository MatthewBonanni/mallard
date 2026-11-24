/**
 * @file flamelet_table.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tabulated chemistry: Mallard's table format (HDF5, written by an
 *        external table generator; specified in docs/design/table_format.md),
 *        multilinear lookup on the host and in device kernels,
 *        and the flamelet/progress-variable (FPV) map from the
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
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

namespace flamelet {

/** @brief A table coordinate in D dimensions: per axis, the lower node of its cell and the fraction t in [0, 1] across it. */
template <uint32_t D>
struct Point {
    uint32_t lo[D] = {};
    double t[D] = {};
};

/**
 * @brief A table of D axes in one memory space, copyable by value into kernels.
 *        Values (float64 or float32; lookups sum in double either way) are
 *        node-major (last axis fastest) with a node's fields
 *        contiguous, so one lookup of several fields reads 2^D short runs.
 *        D is a compile-time constant so that the corner loop unrolls and a
 *        Point stays in registers on GPUs. Results depend only on the query,
 *        never on the backend or the order of evaluation, so lookups are
 *        bitwise reproducible across ranks.
 */
template <uint32_t D, typename Value = double, typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
struct TableView {
    static_assert(D >= 1 && D <= 16, "a lookup reads 2^D nodes");
    static_assert(std::is_same_v<Value, double> || std::is_same_v<Value, float>, "values are float64 or float32");

    uint32_t n_fields = 0;
    uint32_t n_nodes[D] = {};
    uint32_t axis_first[D] = {};  // axis a's nodes at axes(axis_first[a] ...)
    uint64_t stride[D] = {};      // values between neighboring nodes of axis a
    Kokkos::View<const double *, MemorySpace> axes;
    Kokkos::View<const Value *, MemorySpace> values;  // indexed in 64 bits: tables may exceed 2^31 values

    /**
     * @brief The cell of axis a holding x, and the fraction across it; clamped to the axis' ends
     *        (NaN to the first node). Bisection with a trip count set by the axis alone and no
     *        branch on x, so neighboring GPU threads do not diverge.
     */
    KOKKOS_INLINE_FUNCTION void locate_axis(const uint32_t a, const double x, uint32_t & lo, double & t) const {
        const double * v = axes.data() + axis_first[a];
        const uint32_t n = n_nodes[a];
        const double xc = Kokkos::fmin(Kokkos::fmax(x, v[0]), v[n - 1]);
        uint32_t l = 0, len = n - 1;  // the cell, the last l in [0, n - 2] with v[l] <= xc, is in [l, l + len)
        while (len > 1) {
            const uint32_t half = len / 2;
            l = v[l + half] <= xc ? l + half : l;
            len -= half;
        }
        lo = l;
        t = (xc - v[l]) / (v[l + 1] - v[l]);
    }

    /** @brief The table point of the coordinates q[0, D). */
    KOKKOS_INLINE_FUNCTION Point<D> locate(const double * q) const {
        Point<D> p;
        for (uint32_t a = 0; a < D; a++) locate_axis(a, q[a], p.lo[a], p.t[a]);
        return p;
    }

    /** @brief Field f at point p, interpolated multilinearly over the 2^D nodes of its cell. */
    KOKKOS_INLINE_FUNCTION double operator()(const Point<D> & p, const uint32_t f) const {
        double v = 0.0;
        evaluate(p, &f, 1, &v);
        return v;
    }

    /** @brief Fields fields[0, n) at point p into out[0, n), the corner weights computed once. */
    KOKKOS_INLINE_FUNCTION void evaluate(const Point<D> & p, const uint32_t * fields, const uint32_t n, double * out) const {
        corners<D>(p, fields, n, out, 0);
    }

    /**
     * @brief Field f at point p with the last axis held at its node k: interpolated over the
     *        first D - 1 axes alone, 2^(D - 1) nodes.
     */
    KOKKOS_INLINE_FUNCTION double at_last_node(const Point<D> & p, const uint32_t k, const uint32_t f) const {
        double v = 0.0;
        corners<D - 1>(p, &f, 1, &v, k * stride[D - 1]);
        return v;
    }

    private:
        // Sums the 2^M corners of p's cell over its first M axes, from the node offset base along the
        // others, in a fixed order (axis 0 slowest)
        template <uint32_t M>
        KOKKOS_INLINE_FUNCTION void corners(const Point<D> & p, const uint32_t * fields, const uint32_t n, double * out,
                                            uint64_t base) const {
            for (uint32_t i = 0; i < n; i++) out[i] = 0.0;
            for (uint32_t a = 0; a < M; a++) base += p.lo[a] * stride[a];
            constexpr uint32_t n_corners = 1u << M;
            for (uint32_t k = 0; k < n_corners; k++) {
                uint64_t node = base;
                double w = 1.0;
                for (uint32_t a = 0; a < M; a++) {
                    const uint32_t up = (k >> (M - 1 - a)) & 1u;
                    node += up * stride[a];
                    w *= up ? p.t[a] : 1.0 - p.t[a];
                }
                const Value * row = values.data() + node;
                for (uint32_t i = 0; i < n; i++) out[i] += w * static_cast<double>(row[fields[i]]);
            }
        }
};

/**
 * @brief The FPV map (axes Z, Sz = Zvar / (Z (1 - Z)) and Lambda, the mean progress
 *        variable normalized between the table's ends at that (Z, Sz)), on
 *        a TableView. Plain aggregate, captured by value in kernels.
 */
template <typename Value = double, typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
struct FPV {
    TableView<3, Value, MemorySpace> table;
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
     *        at (Z, Sz) is reproduced exactly by the C field (to the rounding
     *        of float32 values in a float32 table); outside, it is clamped to them.
     */
    KOKKOS_INLINE_FUNCTION Point<3> locate(const double Z, const double Zvar, const double C) const {
        Point<3> p;
        table.locate_axis(0, Z, p.lo[0], p.t[0]);
        table.locate_axis(1, normalized_variance(Z, Zvar), p.lo[1], p.t[1]);
        const double C_min = table.at_last_node(p, 0, field_C);
        const double C_max = table.at_last_node(p, table.n_nodes[2] - 1, field_C);
        const double span = C_max - C_min;
        const double Lambda = span > 0.0 ? (C - C_min) / span : 0.0;
        table.locate_axis(2, Lambda, p.lo[2], p.t[2]);
        return p;
    }
};

/**
 * @brief A table read from an HDF5 file (docs/design/table_format.md): the
 *        model and thermodynamics tags, species and pressure as attributes,
 *        the axes' nodes, the values (float64 or float32, node-major) with the
 *        names and units of their fields, model data and provenance.
 */
class Table {
    public:
        /** @brief An axis: a control variable's name, its nodes (strictly increasing) and units. */
        struct Axis {
            std::string name;
            std::vector<double> nodes;
            std::string units = "1";
        };

        /** @brief A stored quantity: its name and units. */
        struct Field {
            std::string name;
            std::string units;
        };

        /**
         * @brief The definitions of an fpv table's control variables, from /model_data: the progress variable
         *        C = sum_k w_k Y_k, the mixture fraction's definition ("bilger") and the mass fractions of its
         *        fuel and oxidizer streams, each by species.
         */
        struct Definitions {
            std::vector<std::pair<std::string, double>> progress_variable;
            std::string mixture_fraction;
            std::vector<std::pair<std::string, double>> fuel, oxidizer;
        };

        /**
         * @brief Read a table file and check it against the format and the contract of its
         *        model and thermodynamics tags. Builds for a GPU map the values from the file
         *        when its layout allows (contiguous, unfiltered, native byte order, aligned),
         *        since they only stage them for the copy to the device; other builds, and files
         *        that cannot be mapped, read them into memory.
         * @throws InputError in a build without HDF5
         * @throws std::runtime_error naming the file and what is wrong with it
         */
        static Table read(const std::string & file);

        /**
         * @brief A table from its parts, for tests: checked as a container (axes, fields, values),
         *        not against its tags' contract until fpv() is called.
         */
        Table(std::string model, std::string thermodynamics, std::vector<Axis> axes, std::vector<Field> fields,
              std::vector<double> values, const std::string & where = "table");

        const std::string & model() const { return m_model; }
        const std::string & thermodynamics() const { return m_thermodynamics; }
        const std::vector<Axis> & axes() const { return m_axes; }
        const std::vector<Field> & fields() const { return m_fields; }
        /** @brief The species of the species block; species k is the field "Y_k". */
        const std::vector<std::string> & species() const { return m_species; }
        /** @brief The pressure the table was generated at (Pa); 0 for a table built from parts. */
        double pressure() const { return m_pressure; }
        /** @brief "<name> <version>" of the generator that wrote the table, for the run log. */
        const std::string & generator() const { return m_generator; }
        /** @brief "<name> sha256:<hash>" of the mechanism the table was generated with, for the run log. */
        const std::string & mechanism() const { return m_mechanism; }
        /** @brief The definitions of an fpv table's control variables (empty for a table from parts). */
        const Definitions & definitions() const { return m_definitions; }
        /** @brief Whether the values are float32 (else float64). */
        bool single() const { return m_single; }
        /** @brief Whether the values are mapped from the file (GPU builds), else read into memory. */
        bool mapped() const { return m_mapped; }

        /** @brief Index of the named field, or -1. */
        int32_t field(const std::string & name) const;

        /**
         * @brief The table in host memory: views of this object's storage, valid while it lives.
         * @throws std::runtime_error unless the table has D axes and values of type Value
         */
        template <uint32_t D, typename Value = double>
        TableView<D, Value, Kokkos::HostSpace> host() const {
            require_value(std::is_same_v<Value, float>);
            using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
            return view<D, Value, Kokkos::HostSpace>(
                Kokkos::View<const double *, Kokkos::HostSpace, Unmanaged>(m_axis_nodes.data(), m_axis_nodes.size()),
                Kokkos::View<const Value *, Kokkos::HostSpace, Unmanaged>(static_cast<const Value *>(m_values), m_n_values));
        }

        /**
         * @brief The table in the default device's memory: a copy, or on host backends the
         *        storage itself, valid while this object lives.
         * @throws std::runtime_error unless the table has D axes and values of type Value
         */
        template <uint32_t D, typename Value = double>
        TableView<D, Value> device() const {
            using Space = Kokkos::DefaultExecutionSpace::memory_space;
            const TableView<D, Value, Kokkos::HostSpace> h = host<D, Value>();
            if constexpr (std::is_same_v<Space, Kokkos::HostSpace>) {
                return h;
            } else {
                Kokkos::View<double *, Space> axes("flamelet_axes", h.axes.size());
                Kokkos::View<Value *, Space> values("flamelet_values", h.values.size());
                Kokkos::deep_copy(axes, h.axes);
                Kokkos::deep_copy(values, h.values);
                return view<D, Value, Space>(axes, values);
            }
        }

        /**
         * @brief The FPV map of a table that meets the contract of model "fpv".
         * @throws std::runtime_error naming what the table lacks, or unless its values are of type Value
         */
        template <typename Value = double>
        FPV<Value> fpv() const {
            const uint32_t C = fpv_field();
            return FPV<Value>{device<3, Value>(), C};
        }

        /** @brief As fpv(), in host memory. */
        template <typename Value = double>
        FPV<Value, Kokkos::HostSpace> fpv_host() const {
            const uint32_t C = fpv_field();
            return FPV<Value, Kokkos::HostSpace>{host<3, Value>(), C};
        }

    private:
        Table() = default;
        void finish();                // container checks, and gathers the axes' nodes
        void check_contract() const;  // the model's and thermodynamics' axes, fields and values
        void check_definitions() const;  // fpv: the weights name species of the block, and C = sum_k w_k Y_k
        uint32_t fpv_field() const;
        void require_axes(uint32_t d) const;
        void require_value(bool single) const;
        template <typename F>
        decltype(auto) with_values(F && f) const {  // f(pointer to the values, float or double)
            return m_single ? f(static_cast<const float *>(m_values)) : f(static_cast<const double *>(m_values));
        }

        template <uint32_t D, typename Value, typename MemorySpace>
        TableView<D, Value, MemorySpace> view(Kokkos::View<const double *, MemorySpace> axis_nodes,
                                              Kokkos::View<const Value *, MemorySpace> values) const {
            require_axes(D);
            TableView<D, Value, MemorySpace> t;
            t.n_fields = static_cast<uint32_t>(m_fields.size());
            uint32_t first = 0;
            for (uint32_t a = 0; a < D; a++) {
                t.n_nodes[a] = static_cast<uint32_t>(m_axes[a].nodes.size());
                t.axis_first[a] = first;
                first += t.n_nodes[a];
            }
            uint64_t stride = t.n_fields;
            for (uint32_t a = D; a-- > 0;) {
                t.stride[a] = stride;
                stride *= t.n_nodes[a];
            }
            t.axes = axis_nodes;
            t.values = values;
            return t;
        }

        std::string m_where = "table";  // the file, in errors
        std::string m_model;
        std::string m_thermodynamics;
        std::vector<Axis> m_axes;
        std::vector<Field> m_fields;
        std::vector<std::string> m_species;
        std::vector<double> m_axis_nodes;  // every axis' nodes, in order
        std::shared_ptr<const void> m_storage;  // owns the values: a vector, or the file's mapping
        const void * m_values = nullptr;
        uint64_t m_n_values = 0;
        bool m_single = false;
        bool m_mapped = false;
        std::string m_generator;
        std::string m_mechanism;
        double m_pressure = 0.0;
        Definitions m_definitions;
};

} // namespace flamelet

#endif // FLAMELET_TABLE_H
