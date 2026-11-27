/**
 * @file particle_snapshots.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Particle snapshots of [[write_data]] (particle_fields): the particle
 *        variables and quantities of the particles in a region, staged on the
 *        device and gathered by id.
 * @version 0.1
 * @date 2026-11-21
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <algorithm>

#include "particles.h"
#include "gated.h"

using particles::EMPTY;

namespace {

const std::string DOCS = "docs/guide/output.md#particle-snapshots";

} // namespace

Particles::Resolved Particles::resolve_fields(const std::vector<std::string> & names, const std::string & key,
                                              const size_t line, const std::string & docs) const {
    std::vector<std::string> variables = particles::base_variables();
    for (const auto & name : quantity_names()) variables.push_back(name);
    auto index = [&](const std::string & name) -> int64_t {
        const auto it = std::find(variables.begin(), variables.end(), name);
        return it == variables.end() ? -1 : it - variables.begin();
    };
    std::vector<std::string> accepted;
    for (const auto & v : variables) {
        if (index(v) > static_cast<int64_t>(N_DIM)) accepted.push_back(v);
    }
    for (const char * vector : {"U", "V"}) accepted.push_back(vector);
    std::sort(accepted.begin(), accepted.end());
    Resolved s;
    std::vector<uint32_t> & columns = s.columns;
    for (const auto & name : names) {
        auto error = [&](const std::string & message, const std::string & fix) {
            return InputError(Diagnostic{.key = key, .line = line, .message = message, .valid = accepted,
                                         .suggestion = suggest_value(name, accepted), .fix = fix, .docs = docs});
        };
        for (const auto & f : s.fields) {
            if (f.name == name) throw error("lists \"" + name + "\" twice", "Remove one");
        }
        std::vector<int64_t> found;
        if (const int64_t scalar = index(name); scalar >= 0) {
            found.push_back(scalar);
        } else {
            const char * suffixes[3] = {"_X", "_Y", "_Z"};
            FOR_I_DIM {
                if (const int64_t c = index(name + suffixes[i]); c >= 0) found.push_back(c);
            }
            if (found.size() != N_DIM) found.clear();
        }
        if (found.empty()) {
            throw error("\"" + name + "\" is no particle variable or quantity",
                        "List variables of particle expressions (RHO, U, P, T, V, T_P) or names of "
                        "[[particles.quantities]]; a vector goes by its name without _X");
        }
        if (found[0] <= static_cast<int64_t>(particles::VAR_T)) {
            throw error("\"" + name + "\": every record holds the positions, time and ids",
                        "Remove it");
        }
        const bool temperature = found[0] == particles::VAR_TEMPERATURE || (found[0] == particles::VAR_TP && !heated);
        if (temperature && R <= 0.0) {
            throw error("\"" + name + "\" is known for a single gas only", "Remove it");
        }
        s.fields.push_back({name, static_cast<uint32_t>(columns.size()), static_cast<uint32_t>(found.size())});
        for (const int64_t c : found) {
            const auto v = static_cast<uint32_t>(c);
            columns.push_back(v);
            const bool gas = v < particles::N_BASE_VARIABLES &&
                             (v < particles::VAR_V || (v < particles::VAR_TP && !is_solid) ||
                              (v == particles::VAR_TP && !heated));
            s.gas = s.gas || gas;
        }
    }
    return s;
}

uint32_t Particles::add_snapshot(const std::vector<std::string> & names, const region::RegionView<> * region,
                                 const std::string & key, const size_t line) {
    return add_staging(resolve_fields(names, key, line, DOCS), region, NO_MEMBERSHIP);
}

uint32_t Particles::add_staging(const Resolved & resolved, const region::RegionView<> * region, const uint32_t bit) {
    const std::vector<uint32_t> & columns = resolved.columns;
    Snapshot s;
    s.bit = bit;
    s.fields = resolved.fields;
    s.gas = resolved.gas;
    s.n_columns = static_cast<uint32_t>(columns.size());
    s.filtered = region != nullptr;
    if (region) s.region = *region;
    s.variable = Kokkos::View<uint32_t *>("particle_snapshot_variables", columns.size());
    Kokkos::deep_copy(s.variable, Kokkos::View<const uint32_t *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
                                      columns.data(), columns.size()));
    s.id = Kokkos::View<uint64_t *>("particle_snapshot_id", capacity);
    s.values = Kokkos::View<double **, Kokkos::LayoutRight>("particle_snapshot_values", capacity, N_DIM + s.n_columns);
    s.h_id = mirrors(s.id);
    s.h_values = mirrors(s.values);
    Kokkos::deep_copy(s.h_id, EMPTY);
    snapshots.push_back(std::move(s));
    return static_cast<uint32_t>(snapshots.size() - 1);
}

void Particles::stage_snapshot(const uint32_t k) {
    Snapshot & snap = snapshots[k];
    if (capacity == 0) return;
    const auto ids = id;
    const auto xs = x;
    const auto a = aux;
    const auto Wp = W_point;
    const auto out_id = snap.id;
    const auto out = snap.values;
    const auto variable = snap.variable;
    const uint32_t w = width, n_columns = snap.n_columns, n_aux = quantities.n_aux();
    const double gas_R = R;
    const bool inertia = is_solid, heat = heated, filtered = snap.filtered, gas = snap.gas;
    const particles::Solid sphere = solid;
    const particles::QuantityFunctor q = quantities.functor();
    const region::RegionView<> where = snap.region;
    const auto members = trajectory_members;
    const uint32_t bit = snap.bit;
    gated::parallel_for("particle_snapshot", capacity, KOKKOS_LAMBDA(const uint32_t s) {
        const rtype * values = xs.data() + size_t(w) * s;
        const double p[3] = {static_cast<double>(values[0]), static_cast<double>(values[1]),
                             N_DIM == 3 ? static_cast<double>(values[N_DIM - 1]) : 0.0};
        const bool member = bit == NO_MEMBERSHIP || (ids(s) < members.extent(0) && ((members(ids(s)) >> bit) & 1u));
        const bool inside = ids(s) != EMPTY && member && (!filtered || where.distance(p) < 0.0);
        out_id(s) = inside ? ids(s) : EMPTY;
        if (!inside) return;
        rtype Ws[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE Ws[i] = gas ? Wp(s, i) : 0.0_r;
        double vars[particles::MAX_VARIABLES];
        const rtype Tp = heat ? sphere.temperature(values + N_DIM) : 0.0_r;
        particles::fill_base(vars, values, 0.0, Ws, inertia ? values + N_DIM : Ws + 1, heat ? &Tp : nullptr, gas_R);
        q.fill(vars, values, a.data() + size_t(n_aux) * s, nullptr);
        FOR_I_DIM out(s, i) = vars[i];
        for (uint32_t j = 0; j < n_columns; j++) out(s, N_DIM + j) = vars[variable(j)];
    });
    copy_to_pinned(mirrors, snap.h_id, snap.id);
    copy_to_pinned(mirrors, snap.h_values, snap.values);
}

ParticleCloud Particles::snapshot(const uint32_t k) const {
    const Snapshot & snap = snapshots[k];
    const size_t n_columns = snap.n_columns, stride = N_DIM + n_columns;
    ParticleCloud cloud;
    for (uint32_t s = 0; s < capacity; s++) {
        if (snap.h_id(s) == EMPTY) continue;
        cloud.id.push_back(snap.h_id(s));
        for (size_t d = 0; d < 3; d++) cloud.x.push_back(d < N_DIM ? snap.h_values(s, d) : 0.0);
        for (size_t j = N_DIM; j < stride; j++) cloud.values.push_back(snap.h_values(s, j));
    }
    return cloud;
}
