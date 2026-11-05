/**
 * @file derived_fields.cpp
 * @brief Derived output fields of a flow state.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "derived_fields.h"

namespace derived {

void compute(const Selection & selection, const Inputs & inputs, const uint32_t n_cells, Kokkos::View<rtype **> out) {
    if (!selection.any()) return;
    const Selection sel = selection;
    const Kokkos::View<rtype *[N_CONSERVATIVE]> W = inputs.W;
    const Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> g = inputs.gradients;
    const rtype gamma = inputs.gamma;
    const Kokkos::View<rtype *, Kokkos::LayoutStride> cell_gamma = inputs.cell_gamma;
    const Kokkos::View<rtype *[N_DIM]> radius_coords = inputs.radius_coords;
    Kokkos::parallel_for("derived_fields", n_cells, KOKKOS_LAMBDA(const uint32_t c) {
        const auto set = [&](const Field f, const rtype value) { out(c, static_cast<uint32_t>(sel.column[f])) = value; };
        if (sel.column[MACH] >= 0) {
            const rtype rho = W(c, 0), p = W(c, N_DIM + 1);
            const rtype gamma_c = cell_gamma.extent(0) > 0 ? cell_gamma(c) : gamma;
            rtype u2 = 0.0;
            FOR_I_DIM u2 += W(c, 1 + i) * W(c, 1 + i);
            set(MACH, Kokkos::sqrt(u2 * rho / (gamma_c * p)));
        }
        if (sel.column[Q_CRITERION] >= 0 || sel.column[LAMBDA2] >= 0) {
            const Tensor a = velocity_gradient(g, W, radius_coords, c);
            if (sel.column[Q_CRITERION] >= 0) set(Q_CRITERION, q_criterion(a));
            if (sel.column[LAMBDA2] >= 0) set(LAMBDA2, lambda2(a));
        }
        if (sel.column[SCHLIEREN] >= 0) {
            rtype g2 = 0.0;
            FOR_I_DIM g2 += g(c, 0, i) * g(c, 0, i);
            set(SCHLIEREN, Kokkos::sqrt(g2));
        }
    });
}

} // namespace derived
