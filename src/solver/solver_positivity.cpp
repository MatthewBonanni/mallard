/**
 * @file solver_positivity.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Positivity preservation: the face-state limiter and the
 *        a-posteriori order fallback of each stage (docs/numerics/positivity.md).
 * @version 0.9
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "solver.h"

#include <Kokkos_Core.hpp>

#include "input.h"
#include "launch_bounds.h"
#include "positivity.h"

void Solver::init_positivity() {
    positivity = toml::find_or<bool>(input, "numerics", "positivity", false);
    const toml::value mood = toml::find_or(input, "numerics", "mood", toml::value(toml::table{}));
    if (!mood.is_table()) throw InputError("[numerics.mood] must be a table.");
    if (!positivity && !mood.as_table().empty()) throw InputError("[numerics.mood] needs numerics.positivity = true.");
    for (const auto & [key, value] : mood.as_table()) {
        if (key != "dmp" && key != "dmp_floor" && key != "dmp_range") {
            throw InputError("numerics.mood: unknown key " + key + " (keys: dmp, dmp_floor, dmp_range).");
        }
    }
    mood_dmp = toml::find_or<bool>(mood, "dmp", false);
    mood_dmp_floor = find_real_or(mood, "dmp_floor", 1.0e-4_r);
    mood_dmp_range = find_real_or(mood, "dmp_range", 1.0e-3_r);
    if (!(mood_dmp_floor >= 0.0_r) || !(mood_dmp_range >= 0.0_r)) {
        throw InputError("numerics.mood: dmp_floor and dmp_range must be non-negative.");
    }
    if (!positivity) return;
    if (simpler) {
        throw InputError("numerics.positivity is not available with chemistry.coupling = \"simpler\" (its stages "
                         "are not forward-Euler steps of the RHS that the fallback checks).");
    }
    face_reconstruction->raw_face_states = true;
    n_limited_cells = Kokkos::View<uint64_t>("n_limited_cells");
    cell_level = Kokkos::View<uint8_t *>("cell_level", mesh->n_cells);
    face_level = Kokkos::View<uint8_t *>("face_level", mesh->n_faces);
    face_changed = Kokkos::View<uint8_t *>("face_changed", mesh->n_faces);
}

void Solver::limit_face_states() {
    if (face_reconstruction->get_type() == FaceReconstructionType::FIRST_ORDER) return;
    const positivity::FaceLimiterFunctor functor{
        mesh->offsets_faces_of_cell,
        mesh->faces_of_cell,
        mesh->cells_of_face,
        face_solution,
        face_reconstruction->face_quad_weights,
        W_cells,
        is_mixture() ? cell_scalars : ScalarView(),
        physics.gamma,
        mixture.n_species,
        static_cast<uint8_t>(face_solution.extent(1)),
        mesh->n_owned(),
        n_limited_cells};
    Kokkos::parallel_for("positivity_limiter", HeavyRange<>(0, mesh->n_reconstructed()), functor);
}

void Solver::order_fallback(const State & state, const State & rhs_state,
                            const std::function<void(uint8_t)> & recompute, const std::function<void()> & assemble) {
    const uint32_t n_owned = mesh->n_owned();
    if (levels_set) {
        Kokkos::deep_copy(cell_level, uint8_t(0));
        Kokkos::deep_copy(face_level, uint8_t(0));
        levels_set = false;
    }
    if (halo.active() && level_halo.flow.extent(0) == 0) {
        level_halo = State("level_halo", mesh->n_cells, mixture.n_species);
    }
    const positivity::CandidateFunctor candidate{mesh->offsets_faces_of_cell,
                                                 mesh->faces_of_cell,
                                                 mesh->cells_of_face,
                                                 state.flow,
                                                 rhs_state.flow,
                                                 state.species,
                                                 rhs_state.species,
                                                 W_cells,
                                                 T_seed,
                                                 cell_level,
                                                 mixture,
                                                 is_mixture(),
                                                 physics.gamma,
                                                 stage_dt,
                                                 mood_dmp,
                                                 mood_dmp_floor,
                                                 mood_dmp_range};
    const positivity::FallbackFacesFunctor faces{mesh->cells_of_face,
                                                 cell_level,
                                                 face_level,
                                                 face_changed,
                                                 face_solution,
                                                 is_mixture() ? face_thermo : Kokkos::View<rtype **[2][2]>(),
                                                 W_cells,
                                                 cell_scalars,
                                                 n_owned,
                                                 mixture.n_species};
    bool any = false;
    while (true) {
        uint32_t n_changed = 0;
        Kokkos::parallel_reduce("fallback_detect", n_owned, candidate, n_changed);
        if (comm::allreduce(n_changed, comm::Op::SUM) == 0) break;
        any = true;
        n_fallback_rounds++;
        if (halo.active()) {
            const positivity::LevelPackFunctor pack{cell_level, level_halo.flow};
            Kokkos::parallel_for("fallback_pack", Kokkos::RangePolicy<positivity::LevelPackFunctor::Pack>(0, n_owned),
                                 pack);
            halo.exchange(level_halo);
            Kokkos::parallel_for("fallback_unpack",
                                 Kokkos::RangePolicy<positivity::LevelPackFunctor::Unpack>(n_owned, mesh->n_cells), pack);
        }
        Kokkos::parallel_for("fallback_faces", mesh->n_faces, faces);
        for (uint8_t level = 1; level <= positivity::MAX_LEVEL; level++) recompute(level);
        assemble();
    }
    if (!any) return;
    positivity::CandidateFunctor failures = candidate;
    failures.count_failures = true;
    uint32_t n_failed = 0;
    Kokkos::parallel_reduce("fallback_failures", n_owned, failures, n_failed);
    n_fallback_failed += comm::allreduce(n_failed, comm::Op::SUM);
    levels_set = true;
    n_fallback_stages++;
    positivity::LevelCountFunctor::value_type counts;
    Kokkos::parallel_reduce("fallback_count", n_owned, positivity::LevelCountFunctor{cell_level}, counts);
    const std::array<uint64_t, 2> total = comm::allreduce(std::array<uint64_t, 2>{counts.n[0], counts.n[1]}, comm::Op::SUM);
    n_fallback_cells[0] += total[0];
    n_fallback_cells[1] += total[1];
}
