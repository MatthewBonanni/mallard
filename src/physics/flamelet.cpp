/**
 * @file flamelet.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tabulated chemistry in the solver: the flamelet/progress-variable gas.
 * @version 0.1
 * @date 2026-11-20
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "flamelet.h"

#include <algorithm>
#include <stdexcept>
#include <type_traits>

#include "common.h"
#include "log.h"

namespace flamelet {

namespace {

/**
 * @brief The table's lookup in memory space Space with float64 values: the
 *        table's own (copied to the device on GPU backends) when they are
 *        float64; widened on the copy when they are float32, since the
 *        solver's lookups read float64.
 */
template <typename Space>
TableView<3, double, Space> table_view(const Table & table) {
    if (!table.single()) {
        if constexpr (std::is_same_v<Space, Kokkos::HostSpace>) {
            return table.host<3, double>();
        } else {
            return table.device<3, double>();
        }
    }
    const TableView<3, float, Kokkos::HostSpace> h = table.host<3, float>();
    TableView<3, double, Space> t;
    t.n_fields = h.n_fields;
    for (uint32_t a = 0; a < 3; a++) {
        t.n_nodes[a] = h.n_nodes[a];
        t.axis_first[a] = h.axis_first[a];
        t.stride[a] = h.stride[a];
    }
    Kokkos::View<double *, Space> axes("flamelet_axes", h.axes.size());
    Kokkos::View<double *, Space> values("flamelet_values", h.values.size());
    auto h_values = Kokkos::create_mirror_view(values);
    for (size_t i = 0; i < h.values.size(); i++) h_values(i) = static_cast<double>(h.values(i));
    Kokkos::deep_copy(axes, h.axes);
    Kokkos::deep_copy(values, h_values);
    t.axes = axes;
    t.values = values;
    return t;
}

template <typename Space>
Gas<Space> make_gas(const Table & table) {
    // Checks the fpv contract, which the solver's fields are part of
    if (table.single()) {
        (void)table.fpv_host<float>();
    } else {
        (void)table.fpv_host<double>();
    }
    Gas<Space> g;
    g.fpv.table = table_view<Space>(table);
    g.fpv.field_C = static_cast<uint32_t>(table.field("C"));
    const char * thermo[5] = {"T0", "e0", "R", "gamma0", "a_gamma"};
    const char * transport[3] = {"mu", "kappa", "cp"};
    const char * source[2] = {"omega_C", "rho0"};
    for (int i = 0; i < 5; i++) g.thermo_fields[i] = static_cast<uint32_t>(table.field(thermo[i]));
    for (int i = 0; i < 3; i++) g.transport_fields[i] = static_cast<uint32_t>(table.field(transport[i]));
    for (int i = 0; i < 2; i++) g.source_fields[i] = static_cast<uint32_t>(table.field(source[i]));
    return g;
}

} // namespace

FlameletModel::FlameletModel(std::shared_ptr<const Table> table, const bool viscous) :
    m_table(std::move(table)), gas(make_gas<Kokkos::DefaultExecutionSpace::memory_space>(*m_table)),
    host_gas(make_gas<Kokkos::HostSpace>(*m_table)), m_viscous(viscous) {
    m_output_names.push_back("T_TABLE");
    const int32_t hrr = m_table->field("HRR");
    m_heat_release = hrr >= 0;
    if (m_heat_release) {
        m_output_names.push_back("HRR");
        m_output_fields.push_back(static_cast<uint32_t>(hrr));
    }
    const std::vector<std::string> scalars = scalar_names();
    for (const std::string & s : m_table->species()) {
        if (std::find(scalars.begin(), scalars.end(), s) != scalars.end()) continue;
        m_output_names.push_back("Y_" + s);
        m_output_fields.push_back(static_cast<uint32_t>(m_table->field("Y_" + s)));
    }
}

FlameletModel FlameletModel::from_input(const Input & input) {
    const Input & physics = input.at("physics");
    if (!physics.contains("table")) {
        throw missing_key(physics, "physics.table",
                          "Add table = \"...\": a flamelet table in Mallard's format (docs/design/table_format.md), "
                          "written by a table generator");
    }
    const std::string file = find_value<std::string>(physics, "table");
    const std::string type = find_value_or<std::string>(physics, "type", "euler");
    if (type != "euler" && type != "navier_stokes") {
        throw not_accepted(physics, "physics.type", type, {"euler", "navier_stokes"});
    }
    std::shared_ptr<const Table> table;
    try {
        table = std::make_shared<const Table>(Table::read(file));
        if (table->model() != "fpv") throw std::runtime_error(file + ": its model is \"" + table->model() + "\", not \"fpv\"");
    } catch (const InputError &) {
        throw;
    } catch (const std::runtime_error & e) {
        throw InputError({.key = "physics.table",
                          .line = line_of(physics, "table"),
                          .message = e.what(),
                          .docs = "docs/guide/flamelet.md"});
    }
    FlameletModel model(table, type == "navier_stokes");
    model.m_file = file;
    const bool les = input.contains("les");
    for (const char * key : {"variance", "C_Z", "C_chi"}) {
        if (!les && physics.contains(key)) {
            throw InputError({.key = std::string("physics.") + key,
                              .line = line_of(physics, key),
                              .message = "closes the subgrid variance of Z, which exists only in LES",
                              .fix = "Remove it, or add [les]",
                              .docs = "docs/guide/flamelet.md"});
        }
    }
    if (les) {
        const std::string variance = find_value_or<std::string>(physics, "variance", "algebraic");
        if (variance != "algebraic" && variance != "transported") {
            throw not_accepted(physics, "physics.variance", variance, {"algebraic", "transported"});
        }
        const bool transported = variance == "transported";
        const char * other = transported ? "C_Z" : "C_chi";
        if (physics.contains(other)) {
            throw InputError({.key = std::string("physics.") + other,
                              .line = line_of(physics, other),
                              .message = std::string("is a constant of variance = \"") +
                                         (transported ? "algebraic" : "transported") + "\", not of \"" + variance +
                                         "\"",
                              .fix = std::string("Remove ") + other,
                              .docs = "docs/guide/flamelet.md"});
        }
        model.m_variance = transported ? Variance::TRANSPORTED : Variance::ALGEBRAIC;
        model.gas.second_moment = model.host_gas.second_moment = transported;
        const char * key = transported ? "C_chi" : "C_Z";
        double & constant = transported ? model.m_C_chi : model.m_C_Z;
        constant = static_cast<double>(find_real_or(physics, key, dim::none, static_cast<rtype>(constant)));
        if (!(transported ? constant > 0.0 : constant >= 0.0)) {
            throw InputError({.key = std::string("physics.") + key,
                              .line = line_of(physics, key),
                              .message = transported ? "must be positive" : "must not be negative",
                              .docs = "docs/guide/flamelet.md"});
        }
    }
    return model;
}

double FlameletModel::burnt_progress(const double Z) const {
    return host_gas.burnt(Z);
}

Thermo FlameletModel::thermo(const double Z, const double C, const double Zvar) const {
    return host_gas.thermo(host_gas.locate(Z, Zvar, C));
}

void FlameletModel::conservatives(const double p, const double T, const double * u, const double Z, const double C,
                                  double * U, double * rhoY) const {
    const Thermo th = thermo(Z, C);
    const double rho = p / (th.R * T);
    double u2 = 0.0;
    for (int i = 0; i < N_DIM; i++) u2 += u[i] * u[i];
    U[0] = rho;
    for (int i = 0; i < N_DIM; i++) U[1 + i] = rho * u[i];
    U[N_DIM + 1] = rho * (th.energy(T) + 0.5 * u2);
    rhoY[Gas<>::Z] = rho * Z;
    rhoY[Gas<>::C] = rho * C;
    if (gas.second_moment) rhoY[Gas<>::Z2] = rho * Z * Z;
}

void FlameletModel::surrogates(const double T, const double Z, const double C, double & gamma, double & e0) const {
    const Thermo th = thermo(Z, C);
    gamma = th.gamma(T);
    e0 = th.energy(T) - th.cv(T) * T;
}

void FlameletModel::outputs(const double rho, const double Z, const double C, const double Zvar, double * out) const {
    const Point<3> p = host_gas.locate(Z, Zvar, C);
    out[0] = host_gas.thermo(p).T0;
    for (size_t i = 0; i < m_output_fields.size(); i++) out[1 + i] = host_gas.field(p, m_output_fields[i]);
    if (m_heat_release) {
        const double r = rho / host_gas.field(p, host_gas.source_fields[1]);
        out[1] *= r * r;
    }
}

report::Summary FlameletModel::summary() const {
    report::Summary s;
    s.setting("Gas", "flamelet/progress variable (" + m_table->model() + ", " + m_table->thermodynamics() + ")");
    s.setting("Table", m_file + " (" + m_table->generator() + ", " + std::to_string(m_table->axes()[0].nodes.size()) +
                           " x " + std::to_string(m_table->axes()[1].nodes.size()) + " x " +
                           std::to_string(m_table->axes()[2].nodes.size()) + " nodes" +
                           (m_table->single() ? ", float32" : "") + ")");
    s.setting("Mechanism", m_table->mechanism());
    if (m_viscous) s.setting("Transport", "from the table, unity Lewis number");
    if (m_variance == Variance::ALGEBRAIC) {
        s.setting("Subgrid variance", "algebraic, C_Z Delta^2 |grad Z|^2 with C_Z = " + logging::real(m_C_Z));
    } else if (m_variance == Variance::TRANSPORTED) {
        s.setting("Subgrid variance", "transported second moment Z2, chi_sgs = C_chi (D + D_t) Zvar / Delta^2 with "
                                      "C_chi = " + logging::real(m_C_chi));
    }
    return s;
}

} // namespace flamelet
