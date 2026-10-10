/**
 * @file solver_les.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Large-eddy simulation parts of the Solver: filter width, eddy
 *        viscosity and the kinetic-energy budget (docs/design/les.md).
 * @version 0.1
 * @date 2026-10-05
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "solver.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <Kokkos_Core.hpp>

#include "input.h"
#include "launch_bounds.h"
#include "mixture_viscous_flux.h"
#include "tfles.h"

namespace {

/** @brief The 3x3 velocity gradient g[i][j] = d u_i / d x_j of a cell from grad(i, j) over N_DIM components. */
template <typename Gradient>
KOKKOS_INLINE_FUNCTION void velocity_gradient(const Gradient & grad, double g[3][3]) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) g[i][j] = 0.0;
    }
    FOR_I_DIM {
        for (uint8_t j = 0; j < N_DIM; j++) g[i][j] = static_cast<double>(grad(i, j));
    }
}

/** @brief SGS coefficients of a calorically perfect gas: mu_t and cp mu_t / Pr_t. */
struct EddyViscosityFunctor {
    LES les;
    rtype cp;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients;
    Kokkos::View<rtype *> delta;
    Kokkos::View<rtype *[3]> sgs;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        double g[3][3];
        velocity_gradient([&](int i, int j) { return gradients(c, 1 + i, j); }, g);
        const rtype mu_t = W(c, 0) * les.nu_t(g, delta(c));
        sgs(c, 0) = mu_t;
        sgs(c, 1) = cp * mu_t / les.Pr_t;
        sgs(c, 2) = 0.0_r;
    }
};

/**
 * @brief SGS coefficients of a gas mixture, [mu_t, cp mu_t / Pr_t,
 *        mu_t / (Sc_t W)], from the transport gradients; the time-step
 *        diffusivity NU_EFF grows by max(4/3 nu_t, lambda_t / (rho cv),
 *        nu_t / Sc_t), a bound on the growth of the largest diffusivity.
 */
struct MixtureEddyViscosityFunctor {
    LES les;
    Mixture gas;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    ScalarView scalars;
    Kokkos::View<rtype **, Kokkos::LayoutRight> values;
    Kokkos::View<rtype ***, Kokkos::LayoutRight> gradients;
    Kokkos::View<rtype *> delta;
    Kokkos::View<rtype *[3]> coefficients;
    Kokkos::View<rtype *[3]> sgs;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        double g[3][3];
        velocity_gradient([&](int i, int j) { return gradients(c, i, j); }, g);
        const double rho = static_cast<double>(W(c, 0));
        const double nu_t = static_cast<double>(les.nu_t(g, delta(c)));
        const ScalarMassFractions y{&scalars(c, 0)};
        const double R = gas.thermo.gas_constant(y);
        const double cp = gas.thermo.cp_mass(static_cast<double>(values(c, N_DIM)), y);
        const double mu_t = rho * nu_t;
        const double lambda_t = cp * mu_t / static_cast<double>(les.Pr_t);
        const double Sc_t = static_cast<double>(les.Sc_t);
        sgs(c, 0) = static_cast<rtype>(mu_t);
        sgs(c, 1) = static_cast<rtype>(lambda_t);
        sgs(c, 2) = static_cast<rtype>(mu_t / Sc_t * R / chemistry::GAS_CONSTANT);
        const double growth = Kokkos::fmax(Kokkos::fmax(4.0 / 3.0 * nu_t, lambda_t / (rho * (cp - R))), nu_t / Sc_t);
        coefficients(c, NU_EFF) += static_cast<rtype>(growth);
    }
};

/** @brief u . R_m - |u|^2 / 2 R_rho per owned cell, R the face fluxes summed over the cell. */
struct KineticEnergyRateFunctor {
    Kokkos::View<uint32_t *> offsets_faces_of_cell;
    Kokkos::View<uint32_t *> faces_of_cell;
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<rtype *[N_CONSERVATIVE]> face_flux;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c, rtype & sum) const {
        rtype R[N_DIM + 1] = {};
        for (uint32_t k = offsets_faces_of_cell(c); k < offsets_faces_of_cell(c + 1); k++) {
            const uint32_t f = faces_of_cell(k);
            const rtype sign = (cells_of_face(f, 0) == static_cast<int32_t>(c)) ? 1.0_r : -1.0_r;
            for (uint8_t i = 0; i <= N_DIM; i++) R[i] += sign * face_flux(f, i);
        }
        rtype u2 = 0.0_r, work = 0.0_r;
        FOR_I_DIM {
            u2 += W(c, 1 + i) * W(c, 1 + i);
            work += W(c, 1 + i) * R[1 + i];
        }
        sum += work - 0.5_r * u2 * R[0];
    }
};

/**
 * @brief Pressure work of the convective operator per owned cell: interior faces by the cell values'
 *        mean(p) n A, boundary faces by their face_flux (their exact share).
 */
struct PressureWorkFunctor {
    Kokkos::View<uint32_t *> offsets_faces_of_cell;
    Kokkos::View<uint32_t *> faces_of_cell;
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<rtype *[N_DIM]> face_normals;
    Kokkos::View<rtype *[N_CONSERVATIVE]> face_flux;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c, rtype & sum) const {
        rtype u2 = 0.0_r;
        FOR_I_DIM u2 += W(c, 1 + i) * W(c, 1 + i);
        for (uint32_t k = offsets_faces_of_cell(c); k < offsets_faces_of_cell(c + 1); k++) {
            const uint32_t f = faces_of_cell(k);
            const bool owner = cells_of_face(f, 0) == static_cast<int32_t>(c);
            const rtype sign = owner ? 1.0_r : -1.0_r;
            const int32_t other = cells_of_face(f, owner ? 1 : 0);
            if (other < 0) {
                rtype work = 0.0_r;
                FOR_I_DIM work += W(c, 1 + i) * face_flux(f, 1 + i);
                sum += sign * (work - 0.5_r * u2 * face_flux(f, 0));
            } else {
                rtype u_n = 0.0_r;
                FOR_I_DIM u_n += W(c, 1 + i) * face_normals(f, i);
                sum -= sign * 0.5_r * (W(c, N_DIM + 1) + W(other, N_DIM + 1)) * u_n;
            }
        }
    }
};

/** @brief Vorticity of each cell from the mixture transport gradients (2D: its z component only). */
struct VorticityFunctor {
    Kokkos::View<rtype ***, Kokkos::LayoutRight> gradients;  // (cell, [u, T, X], dimension)
    Kokkos::View<rtype *[3]> vorticity;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        if constexpr (N_DIM == 2) {
            vorticity(c, 0) = vorticity(c, 1) = 0.0_r;
            vorticity(c, 2) = gradients(c, 1, 0) - gradients(c, 0, 1);
        } else {
            for (uint8_t k = 0; k < 3; k++) {
                const uint8_t a = static_cast<uint8_t>((k + 1) % 3), b = static_cast<uint8_t>((k + 2) % 3);
                vorticity(c, k) = gradients(c, b, a) - gradients(c, a, b);
            }
        }
    }
};

/** @brief Least-squares gradient of the vorticity, on the stencil of the viscous gradients (zero-gradient ghosts). */
struct VorticityGradientFunctor {
    LSQVertexGradientFunctor stencil;
    Kokkos::View<rtype *[3]> vorticity;
    Kokkos::View<rtype *[3][N_DIM]> gradients;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        rtype q_c[3], g[3][N_DIM] = {};
        for (uint8_t v = 0; v < 3; v++) q_c[v] = vorticity(c, v);
        for (uint32_t k = stencil.offsets_cells_of_cell(c); k < stencil.offsets_cells_of_cell(c + 1); k++) {
            const uint32_t j = stencil.cells_of_cell(k);
            rtype w[N_DIM];
            FOR_I_DIM w[i] = stencil.weights.cells(k, i);
            for (uint8_t v = 0; v < 3; v++) {
                const rtype dq = vorticity(j, v) - q_c[v];
                FOR_I_DIM g[v][i] += w[i] * dq;
            }
        }
        for (uint8_t v = 0; v < 3; v++) FOR_I_DIM gradients(c, v, i) = g[v][i];
    }
};

/**
 * @brief Thickened flame fields of an owned cell: [F, E, Omega] and the
 *        chemistry time scale E / F. The SGS velocity is Colin et al.'s
 *        u' = 2 Delta^3 |lap(curl u)|, with the Laplacian of the vorticity the
 *        divergence of its face gradients (cell means corrected along the
 *        line of centroids, as the viscous fluxes); boundary faces carry none.
 */
struct ThickenedFlameFunctor {
    ThickenedFlame tf;
    Kokkos::View<uint32_t *> offsets_faces_of_cell;
    Kokkos::View<uint32_t *> faces_of_cell;
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<rtype *[N_DIM]> normals;
    Kokkos::View<rtype *> face_area;
    Kokkos::View<rtype *> cell_volume;
    Kokkos::View<rtype *[N_DIM]> cell_coords;
    Kokkos::View<rtype *[N_DIM]> shifts;
    Kokkos::View<uint8_t *> face_shift;
    Kokkos::View<rtype *[3]> vorticity;
    Kokkos::View<rtype *[3][N_DIM]> gradients;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    Kokkos::View<rtype **, Kokkos::LayoutRight> values;  // (cell, [u, T, X])
    Kokkos::View<rtype *[3]> transport;
    Kokkos::View<rtype *> delta;
    Kokkos::View<rtype *[3]> fields;
    Kokkos::View<rtype *> time_scale;
    Kokkos::View<rtype *[3]> sgs;  // [mu_t, ...]

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        double lap[3] = {};
        for (uint32_t k = offsets_faces_of_cell(c); k < offsets_faces_of_cell(c + 1); k++) {
            const uint32_t f = faces_of_cell(k);
            const int32_t c0 = cells_of_face(f, 0), c1 = cells_of_face(f, 1);
            if (c1 < 0) continue;
            rtype n[N_DIM], n_vec[N_DIM], d[N_DIM];
            FOR_I_DIM n_vec[i] = normals(f, i);
            unit<N_DIM>(n_vec, n);
            FOR_I_DIM d[i] = (cell_coords(c1, i) + shifts(face_shift(f), i)) - cell_coords(c0, i);
            const rtype d_n = dot<N_DIM>(d, n);
            const rtype sign = c0 == static_cast<int32_t>(c) ? 1.0_r : -1.0_r;
            for (uint8_t v = 0; v < 3; v++) {
                rtype g[N_DIM];
                FOR_I_DIM g[i] = 0.5_r * (gradients(c0, v, i) + gradients(c1, v, i));
                const rtype dq = vorticity(c1, v) - vorticity(c0, v);
                const rtype g_n = dot<N_DIM>(g, n) + (dq - dot<N_DIM>(g, d)) / d_n;
                lap[v] += static_cast<double>(sign * face_area(f) * g_n);
            }
        }
        const double V = static_cast<double>(cell_volume(c)), D = static_cast<double>(delta(c));
        const double lap_norm = Kokkos::sqrt(lap[0] * lap[0] + lap[1] * lap[1] + lap[2] * lap[2]) / V;
        const double rho = static_cast<double>(W(c, 0));
        const double u_prime = tf.eddy_viscosity_velocity ? tf.C_u * static_cast<double>(sgs(c, 0)) / (rho * D)
                                                          : 2.0 * D * D * D * lap_norm;
        const double omega = tf.sensor(static_cast<double>(values(c, N_DIM)));
        const double F = tf.thickening(D, omega);
        const double nu = static_cast<double>(transport(c, MU)) / rho;
        const double E = tf.wrinkling(F, u_prime, nu);
        fields(c, 0) = static_cast<rtype>(F);
        fields(c, 1) = static_cast<rtype>(E);
        fields(c, 2) = static_cast<rtype>(omega);
        time_scale(c) = static_cast<rtype>(E / F);
    }
};

/**
 * @brief Thickened transport of each cell: conductivity and species
 *        diffusivities times E F; the SGS heat and species fluxes times
 *        1 - Omega (the thickened flame is resolved); the time step's
 *        diffusivity times E F.
 */
struct ThickenFunctor {
    Kokkos::View<rtype *[3]> fields;
    Kokkos::View<rtype *[3]> coefficients;
    Kokkos::View<double **, Kokkos::LayoutRight> diffusion;
    Kokkos::View<rtype *[3]> sgs;
    uint32_t n_species;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        const rtype EF = fields(c, 0) * fields(c, 1), resolved = 1.0_r - fields(c, 2);
        coefficients(c, LAMBDA) *= EF;
        coefficients(c, NU_EFF) *= EF;
        // Species and thermal diffusion coefficients
        for (uint32_t k = 0; k < diffusion.extent(1); k++) diffusion(c, k) *= static_cast<double>(EF);
        if (sgs.extent(0) > 0) {
            sgs(c, 1) *= resolved;
            sgs(c, 2) *= resolved;
        }
    }
};

/** @brief The 3x3 velocity gradient d u_i / d x_j of a cell packed at 3 i + j (zero beyond N_DIM). */
KOKKOS_INLINE_FUNCTION
void packed_gradient(const SpeciesView & gradients, const uint32_t n, double g[3][3]) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) g[i][j] = 0.0;
    }
    for (uint8_t i = 0; i < N_DIM; i++) {
        for (uint8_t j = 0; j < N_DIM; j++) g[i][j] = static_cast<double>(gradients(n, 3 * i + j));
    }
}

KOKKOS_INLINE_FUNCTION
void deviatoric_strain(const double g[3][3], double s[3][3]) {
    const double third_trace = (g[0][0] + g[1][1] + g[2][2]) / 3.0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) s[i][j] = 0.5 * (g[i][j] + g[j][i]) - (i == j ? third_trace : 0.0);
    }
}

/** @brief rho Delta^2 D(g) S^d of every cell (owned and halo), filtered by DynamicProcedureFunctor. */
struct DynamicModelStressFunctor {
    SGSModel model;
    StateView flow;          // [rho, u, ...]
    SpeciesView gradients;   // d u_i / d x_j at 3 i + j
    Kokkos::View<rtype *> delta;
    Kokkos::View<double *[6]> model_stresses;  // xx, yy, zz, xy, xz, yz

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t n) const {
        double g[3][3], s[3][3];
        packed_gradient(gradients, n, g);
        deviatoric_strain(g, s);
        const double d = static_cast<double>(delta(n));
        const double q = static_cast<double>(flow(n, 0)) * d * d * LES::operator_of(model, g);
        model_stresses(n, 0) = q * s[0][0];
        model_stresses(n, 1) = q * s[1][1];
        model_stresses(n, 2) = q * s[2][2];
        model_stresses(n, 3) = q * s[0][1];
        model_stresses(n, 4) = q * s[0][2];
        model_stresses(n, 5) = q * s[1][2];
    }
};

/**
 * @brief Germano-identity terms [L^d : M, M : M] V of one owned cell for the global dynamic procedure
 *        (Germano et al. 1991, Lilly 1992; Favre-weighted as Moin et al. 1991), with the model's
 *        tau^d = -2 C^2 rho Delta^2 D(g) S^d. The test filter is the volume-weighted average over the cell
 *        and its vertex neighbors, of width Delta_hat^2 = Delta^2 + 12 / d tr(cov), cov the stencil's
 *        volume-weighted covariance of the centroids (Delta_hat = 3 Delta on uniform hexahedra);
 *        filtered gradients stand for the gradients of the filtered field.
 */
struct DynamicProcedureFunctor {
    SGSModel model;
    Kokkos::View<uint32_t *> offsets;
    Kokkos::View<uint32_t *> neighbors;
    Kokkos::View<uint8_t *> neighbor_shift;
    Kokkos::View<rtype *[N_DIM]> shifts;
    Kokkos::View<rtype *[N_DIM]> cell_coords;
    Kokkos::View<rtype *> volume;
    Kokkos::View<rtype *> delta;
    StateView flow;
    SpeciesView gradients;
    Kokkos::View<double *[6]> model_stresses;
    Kokkos::View<double *[2]> terms;

    static constexpr uint32_t MAX_STENCIL = 160;  // cell + vertex neighbors (tetrahedra: up to about 100)

    /** @brief Component i of the position of stencil entry k (the cell itself for k == end) relative to cell c. */
    KOKKOS_INLINE_FUNCTION
    double relative(const uint32_t c, const uint32_t k, const uint32_t end, const uint8_t i) const {
        if (k == end) return 0.0;
        return static_cast<double>(cell_coords(neighbors(k), i) - cell_coords(c, i)) +
               static_cast<double>(shifts(neighbor_shift(k), i));
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        double w_sum = 0.0, rho = 0.0, m[3] = {}, rho_uu[3][3] = {}, g[3][3] = {}, model_stress[3][3] = {};
        double x1[3] = {}, x2[3] = {};
        const uint32_t begin = offsets(c), end = offsets(c + 1);
        // The stencil in a canonical order (by relative position), so the sums do not depend on the local
        // numbering of the neighbors and the constant is bitwise independent of the rank count
        uint32_t order[MAX_STENCIL];
        const uint32_t size = Kokkos::min(end - begin + 1, static_cast<uint32_t>(MAX_STENCIL));
        for (uint32_t i = 0; i < size; i++) order[i] = begin + i;
        auto before = [&](const uint32_t ka, const uint32_t kb) {
            for (uint8_t i = 0; i < N_DIM; i++) {
                const double a = relative(c, ka, end, i), b = relative(c, kb, end, i);
                if (a != b) return a < b;
            }
            return false;
        };
        for (uint32_t i = 1; i < size; i++) {
            const uint32_t key = order[i];
            uint32_t j = i;
            while (j > 0 && before(key, order[j - 1])) {
                order[j] = order[j - 1];
                j--;
            }
            order[j] = key;
        }
        for (uint32_t o = 0; o < size; o++) {
            const uint32_t k = order[o];
            const uint32_t n = k < end ? neighbors(k) : c;
            const double w = static_cast<double>(volume(n));
            double u[3] = {}, gn[3][3], dx[3] = {};
            packed_gradient(gradients, n, gn);
            FOR_I_DIM {
                u[i] = static_cast<double>(flow(n, 1 + i));
                dx[i] = static_cast<double>(cell_coords(n, i) - cell_coords(c, i)) +
                        (k < end ? static_cast<double>(shifts(neighbor_shift(k), i)) : 0.0);
            }
            const double r = static_cast<double>(flow(n, 0));
            const double q[3][3] = {{model_stresses(n, 0), model_stresses(n, 3), model_stresses(n, 4)},
                                    {model_stresses(n, 3), model_stresses(n, 1), model_stresses(n, 5)},
                                    {model_stresses(n, 4), model_stresses(n, 5), model_stresses(n, 2)}};
            w_sum += w;
            rho += w * r;
            for (int i = 0; i < 3; i++) {
                m[i] += w * r * u[i];
                x1[i] += w * dx[i];
                x2[i] += w * dx[i] * dx[i];
                for (int j = 0; j < 3; j++) {
                    rho_uu[i][j] += w * r * u[i] * u[j];
                    g[i][j] += w * gn[i][j];
                    model_stress[i][j] += w * q[i][j];
                }
            }
        }
        double variance = 0.0;
        for (int i = 0; i < 3; i++) {
            x1[i] /= w_sum;
            variance += x2[i] / w_sum - x1[i] * x1[i];
        }
        const double d = static_cast<double>(delta(c));
        const double delta_hat2 = d * d + 12.0 / N_DIM * variance;
        rho /= w_sum;
        for (int i = 0; i < 3; i++) {
            m[i] /= w_sum;
            for (int j = 0; j < 3; j++) {
                rho_uu[i][j] /= w_sum;
                g[i][j] /= w_sum;
                model_stress[i][j] /= w_sum;
            }
        }
        double s_hat[3][3];
        deviatoric_strain(g, s_hat);
        const double nu_hat = delta_hat2 * LES::operator_of(model, g);
        double L[3][3], M[3][3], trace = 0.0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                L[i][j] = rho_uu[i][j] - m[i] * m[j] / rho;
                M[i][j] = 2.0 * (model_stress[i][j] - rho * nu_hat * s_hat[i][j]);
            }
            trace += L[i][i];
        }
        double lm = 0.0, mm = 0.0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                lm += (L[i][j] - (i == j ? trace / 3.0 : 0.0)) * M[i][j];
                mm += M[i][j] * M[i][j];
            }
        }
        const double V = static_cast<double>(volume(c));
        terms(c, 0) = lm * V;
        terms(c, 1) = mm * V;
    }
};

/**
 * @brief Sums over all ranks of the owned cells' terms, exact in fixed point (each term rounded to 2^-b of
 *        the largest, b leaving room for every cell in 63 bits), so independent of the rank count and order.
 */
std::array<double, 2> exact_sums(const Kokkos::View<double *[2]> & terms, const uint32_t n_owned, const bool distributed) {
    std::array<double, 2> largest = {0.0, 0.0};
    for (size_t k = 0; k < 2; k++) {
        double local = 0.0;
        Kokkos::parallel_reduce(
            "dynamic_largest", n_owned,
            KOKKOS_LAMBDA(const uint32_t c, double & m) { m = Kokkos::fmax(m, Kokkos::fabs(terms(c, k))); },
            Kokkos::Max<double>(local));
        largest[k] = local;
    }
    if (distributed) largest = comm::allreduce(largest, comm::Op::MAX);
    const uint64_t n_global =
        distributed ? comm::allreduce(static_cast<uint64_t>(n_owned), comm::Op::SUM) : static_cast<uint64_t>(n_owned);
    const int bits = 62 - static_cast<int>(std::ceil(std::log2(static_cast<double>(n_global) + 1.0)));
    std::array<int64_t, 2> sums = {0, 0};
    std::array<double, 2> scale = {0.0, 0.0};
    for (size_t k = 0; k < 2; k++) {
        if (!(largest[k] > 0.0)) continue;
        const double s = std::ldexp(1.0, bits) / largest[k];
        scale[k] = s;
        int64_t local = 0;
        Kokkos::parallel_reduce(
            "dynamic_sum", n_owned,
            KOKKOS_LAMBDA(const uint32_t c, int64_t & sum) {
                sum += static_cast<int64_t>(Kokkos::round(terms(c, k) * s));
            },
            Kokkos::Sum<int64_t>(local));
        sums[k] = local;
    }
    if (distributed) sums = comm::allreduce(sums, comm::Op::SUM);
    return {scale[0] > 0.0 ? static_cast<double>(sums[0]) / scale[0] : 0.0,
            scale[1] > 0.0 ? static_cast<double>(sums[1]) / scale[1] : 0.0};
}

/**
 * @brief Extents h_1 <= h_2 <= h_3 of a 3D cell: V over the eigenvalues of its projected-area tensor
 *        1/2 sum_f A_f A_f^T / |A_f| (exact for boxes; equal for regular tetrahedra). Faces only, so cells
 *        on periodic boundaries need no node unwrapping.
 */
std::array<double, 3> cell_extents(const Mesh & mesh, const uint32_t c) {
    // Faces summed in a canonical order, so the extents do not depend on the local face numbering
    std::vector<std::array<double, 3>> normals;
    for (uint32_t k = mesh.h_offsets_faces_of_cell(c); k < mesh.h_offsets_faces_of_cell(c + 1); k++) {
        const uint32_t f = mesh.h_faces_of_cell(k);
        std::array<double, 3> n = {};
        FOR_I_DIM n[i] = static_cast<double>(mesh.h_face_normals(f, i));
        // n n^T does not depend on the orientation: make the first nonzero component positive
        const double sign = n[0] != 0.0 ? n[0] : (n[1] != 0.0 ? n[1] : n[2]);
        if (sign < 0.0) {
            for (double & v : n) v = -v;
        }
        normals.push_back(n);
    }
    std::sort(normals.begin(), normals.end());
    double a[3][3] = {};
    for (const auto & n : normals) {
        const double norm = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) a[i][j] += 0.5 * n[i] * n[j] / norm;
        }
    }
    // Cyclic Jacobi rotations of the symmetric tensor
    for (int sweep = 0; sweep < 50; sweep++) {
        const double off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
        if (off < 1e-30 * (a[0][0] * a[0][0] + a[1][1] * a[1][1] + a[2][2] * a[2][2])) break;
        for (int p = 0; p < 2; p++) {
            for (int q = p + 1; q < 3; q++) {
                if (a[p][q] == 0.0) continue;
                const double theta = 0.5 * std::atan2(2.0 * a[p][q], a[q][q] - a[p][p]);
                const double cs = std::cos(theta), sn = std::sin(theta);
                for (int k = 0; k < 3; k++) {
                    const double akp = a[k][p], akq = a[k][q];
                    a[k][p] = cs * akp - sn * akq;
                    a[k][q] = sn * akp + cs * akq;
                }
                for (int k = 0; k < 3; k++) {
                    const double apk = a[p][k], aqk = a[q][k];
                    a[p][k] = cs * apk - sn * aqk;
                    a[q][k] = sn * apk + cs * aqk;
                }
            }
        }
    }
    const double V = static_cast<double>(mesh.h_cell_volume(c));
    std::array<double, 3> h = {V / a[0][0], V / a[1][1], V / a[2][2]};
    std::sort(h.begin(), h.end());
    return h;
}

} // namespace

void Solver::update_thickened_flame() {
    // calc_dt has filled the halo of the state
    eddy_viscosity_of_state(mesh->n_cells);
    Kokkos::parallel_for("tfles_vorticity", mesh->n_cells, VorticityFunctor{transport_gradients, tfles_vorticity});
    // The vorticity of outer halo cells misses neighbors: take the owners' before differentiating it again
    exchange_cell_vectors(tfles_vorticity);
    Kokkos::parallel_for("tfles_vorticity_gradient", mesh->n_cells,
                         VorticityGradientFunctor{viscous_gradient, tfles_vorticity, tfles_gradients});
    Kokkos::parallel_for("tfles_fields", mesh->n_owned(),
                         ThickenedFlameFunctor{thickened_flame, mesh->offsets_faces_of_cell, mesh->faces_of_cell,
                                               mesh->cells_of_face, mesh->face_normals, mesh->face_area,
                                               mesh->cell_volume, mesh->cell_coords, mesh->shifts, mesh->face_shift,
                                               tfles_vorticity, tfles_gradients, W_cells, transport_values,
                                               cell_transport, les_delta, tfles_fields, chem_time_scale,
                                               les_coefficients});
    // Halo cells take their owners' fields: the face diffusivities need them
    exchange_cell_vectors(tfles_fields);
}

void Solver::update_partially_stirred_reactor() {
    const uint32_t n_owned = mesh->n_owned();
    cell_chemistry.heat_release(conservatives, species, T_seed, hrr, production, n_owned);
    const PartiallyStirredReactor model = pasr;
    const Mixture gas = mixture;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W = W_cells;
    ScalarView scalars = cell_scalars;
    Kokkos::View<rtype **, Kokkos::LayoutRight> values = transport_values;
    Kokkos::View<rtype *[3]> transport = cell_transport, sgs = les_coefficients;
    Kokkos::View<rtype *> q = hrr, delta = les_delta, kappa = chem_time_scale;
    Kokkos::parallel_for("pasr_fraction", n_owned, KOKKOS_LAMBDA(const uint32_t c) {
        const double rho = static_cast<double>(W(c, 0));
        const double T = static_cast<double>(values(c, N_DIM));
        const ScalarMassFractions y{&scalars(c, 0)};
        const double rho_cp_T = rho * gas.thermo.cp_mass(T, y) * T;
        kappa(c) = static_cast<rtype>(model.fraction(rho_cp_T, static_cast<double>(q(c)), static_cast<double>(delta(c)),
                                                     static_cast<double>(transport(c, 0)) / rho,
                                                     static_cast<double>(sgs(c, 0)) / rho));
    });
}

void Solver::update_dynamic_constant() {
    const uint32_t n_owned = mesh->n_owned();
    StateView flow = dynamic_halo.flow;
    SpeciesView g = dynamic_halo.species;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W = W_cells;
    if (is_mixture()) {
        Kokkos::View<rtype ***, Kokkos::LayoutRight> tg = transport_gradients;
        Kokkos::parallel_for("dynamic_pack", n_owned, KOKKOS_LAMBDA(const uint32_t c) {
            FOR_I_CONSERVATIVE flow(c, i) = W(c, i);
            for (uint8_t i = 0; i < N_DIM; i++) {
                for (uint8_t j = 0; j < N_DIM; j++) g(c, 3 * i + j) = tg(c, i, j);
            }
        });
    } else {
        Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> vg = viscous_gradients;
        Kokkos::parallel_for("dynamic_pack", n_owned, KOKKOS_LAMBDA(const uint32_t c) {
            FOR_I_CONSERVATIVE flow(c, i) = W(c, i);
            for (uint8_t i = 0; i < N_DIM; i++) {
                for (uint8_t j = 0; j < N_DIM; j++) g(c, 3 * i + j) = vg(c, 1 + i, j);
            }
        });
    }
    if (halo.active()) halo.exchange(dynamic_halo);
    Kokkos::parallel_for("dynamic_stresses", mesh->n_cells,
                         DynamicModelStressFunctor{les.model, flow, g, les_width, dynamic_stresses});
    Kokkos::parallel_for("dynamic_terms", n_owned,
                         DynamicProcedureFunctor{les.model, mesh->offsets_cells_of_cell, mesh->cells_of_cell,
                                                 mesh->cells_of_cell_shift, mesh->shifts, mesh->cell_coords,
                                                 mesh->cell_volume, les_width, flow, g, dynamic_stresses,
                                                 dynamic_terms});
    const auto [lm, mm] = exact_sums(dynamic_terms, n_owned, is_distributed());
    // C^2 (Vreman: c) = <L:M> / <M:M>, clipped at zero; a field without resolved strain keeps the previous one
    if (mm > 0.0) {
        dynamic_ratio = lm / mm;
        const double c2 = std::max(0.0, dynamic_ratio);
        les.C = static_cast<rtype>(les.model == SGSModel::VREMAN ? c2 : std::sqrt(c2));
    }
}

void Solver::exchange_cell_vectors(const Kokkos::View<rtype *[3]> & v) {
    if (!halo.active()) return;
    SpeciesView packed = tfles_halo.species;
    Kokkos::parallel_for("halo_pack", mesh->n_cells, KOKKOS_LAMBDA(const uint32_t c) {
        for (int k = 0; k < 3; k++) packed(c, k) = v(c, k);
    });
    halo.exchange(tfles_halo);
    Kokkos::parallel_for("halo_unpack", mesh->n_cells, KOKKOS_LAMBDA(const uint32_t c) {
        for (int k = 0; k < 3; k++) v(c, k) = packed(c, k);
    });
}

void Solver::thicken_transport() {
    Kokkos::parallel_for("tfles_thicken", mesh->n_cells,
                         ThickenFunctor{tfles_fields, cell_transport, cell_diffusion, les_coefficients,
                                        mixture.n_species});
}

void Solver::init_les() {
    les_on = LES::in_input(input);
    if (!les_on) return;
    les = LES::from_input(input);
    if (axisymmetric) throw InputError("[les] is not available for axisymmetric runs.");
    les_delta = Kokkos::View<rtype *>("les_delta", mesh->n_cells);
    auto h_delta = Kokkos::create_mirror_view(les_delta);
    for (uint32_t c = 0; c < mesh->n_cells; c++) h_delta(c) = std::pow(mesh->h_cell_volume(c), 1.0_r / N_DIM);
    Kokkos::deep_copy(les_delta, h_delta);
    les_width = les_delta;
    if (les.scotti) {
        les_width = Kokkos::View<rtype *>("les_width", mesh->n_cells);
        auto h_width = Kokkos::create_mirror_view(les_width);
        for (uint32_t c = 0; c < mesh->n_owned(); c++) {
            const std::array<double, 3> h = cell_extents(*mesh, c);
            h_width(c) = h_delta(c) * static_cast<rtype>(LES::scotti_factor(h[0], h[1], h[2]));
        }
        // Halo cells lack some of their faces here: take their owners' widths
        if (halo.active()) {
            State packed("les_width_halo", mesh->n_cells, 1);
            auto h_packed = Kokkos::create_mirror_view(packed.species);
            for (uint32_t c = 0; c < mesh->n_owned(); c++) h_packed(c, 0) = h_width(c);
            Kokkos::deep_copy(packed.species, h_packed);
            halo.exchange(packed);
            Kokkos::deep_copy(h_packed, packed.species);
            for (uint32_t c = mesh->n_owned(); c < mesh->n_cells; c++) h_width(c) = h_packed(c, 0);
        }
        Kokkos::deep_copy(les_width, h_width);
    }
    if (les.dynamic) {
        dynamic_halo = State("dynamic_halo", mesh->n_cells, 9);
        dynamic_terms = Kokkos::View<double *[2]>("dynamic_terms", mesh->n_owned());
        dynamic_stresses = Kokkos::View<double *[6]>("dynamic_stresses", mesh->n_cells);
    }
    les_coefficients = Kokkos::View<rtype *[3]>("les_coefficients", mesh->n_cells);
    h_les_coefficients = Kokkos::create_mirror_view(les_coefficients);
    if (!input.at("les").contains("combustion")) return;
    if (!reacting || !is_viscous()) {
        throw InputError("[les.combustion] needs a reacting viscous mixture (gas = \"mixture\", navier_stokes, [chemistry]).");
    }
    const toml::value & combustion = input.at("les").at("combustion");
    if (toml::find_or<std::string>(combustion, "model", "") == "pasr") {
        pasr = PartiallyStirredReactor::from_input(combustion);
        pasr_on = true;
        chem_time_scale = Kokkos::View<rtype *>("chem_time_scale", mesh->n_cells);
        h_chem_time_scale = Kokkos::create_mirror_view(chem_time_scale);
        Kokkos::deep_copy(chem_time_scale, 1.0_r);
        return;
    }
    thickened_flame = ThickenedFlame::from_input(input.at("les").at("combustion"));
    tfles_on = true;
    tfles_fields = Kokkos::View<rtype *[3]>("tfles_fields", mesh->n_cells);
    h_tfles_fields = Kokkos::create_mirror_view(tfles_fields);
    chem_time_scale = Kokkos::View<rtype *>("chem_time_scale", mesh->n_cells);
    tfles_vorticity = Kokkos::View<rtype *[3]>("tfles_vorticity", mesh->n_cells);
    tfles_gradients = Kokkos::View<rtype *[3][N_DIM]>("tfles_gradients", mesh->n_cells);
    tfles_halo = State("tfles_halo", mesh->n_cells, 3);
}

void Solver::update_eddy_viscosity(const uint32_t n) {
    if (is_mixture()) {
        Kokkos::parallel_for("eddy_viscosity", n,
                             MixtureEddyViscosityFunctor{les, mixture, W_cells, cell_scalars, transport_values,
                                                         transport_gradients, les_width, cell_transport,
                                                         les_coefficients});
    } else {
        Kokkos::parallel_for("eddy_viscosity", n,
                             EddyViscosityFunctor{les, physics.cp, W_cells, viscous_gradients, les_width, les_coefficients});
    }
}

void Solver::eddy_viscosity_of_state(const uint32_t n) {
    update_boundary_states(t);
    if (is_mixture()) {
        update_cell_states(state(), false);
        Kokkos::parallel_for("mixture_transport", mesh->n_cells,
                             MixtureTransportFunctor{mixture, W_cells, cell_scalars, cell_transport, cell_diffusion,
                                                     transport_values});
        Kokkos::parallel_for("mixture_gradients", n,
                             MixtureGradientFunctor{viscous_gradient, transport_values, transport_gradients,
                                                    bc_transport_values, mixture.n_species});
    } else {
        const Euler phys = physics;
        StateView U = conservatives;
        Kokkos::View<rtype *[N_CONSERVATIVE]> W = W_cells;
        Kokkos::parallel_for("les_W", mesh->n_cells, KOKKOS_LAMBDA(const uint32_t c) {
            rtype cons[N_CONSERVATIVE], W_c[N_CONSERVATIVE];
            FOR_I_CONSERVATIVE cons[i] = U(c, i);
            phys.compute_W_from_conservatives(W_c, cons);
            FOR_I_CONSERVATIVE W(c, i) = W_c[i];
        });
        Kokkos::parallel_for("les_gradients", HeavyRange<>(0, n), viscous_gradient);
    }
    update_eddy_viscosity(n);
}

rtype Solver::kinetic_energy_rate() const {
    rtype sum = 0.0_r;
    Kokkos::parallel_reduce("kinetic_energy_rate", mesh->n_owned(),
                            KineticEnergyRateFunctor{mesh->offsets_faces_of_cell, mesh->faces_of_cell,
                                                     mesh->cells_of_face, face_flux, W_cells},
                            sum);
    return sum;
}

rtype Solver::pressure_work_rate() const {
    rtype sum = 0.0_r;
    Kokkos::parallel_reduce("pressure_work_rate", mesh->n_owned(),
                            PressureWorkFunctor{mesh->offsets_faces_of_cell, mesh->faces_of_cell, mesh->cells_of_face,
                                                mesh->face_normals, face_flux, W_cells},
                            sum);
    return sum;
}

KineticEnergyBudget Solver::kinetic_energy_budget() {
    State scratch("budget_rhs", mesh->n_cells, static_cast<uint32_t>(species_names.size()));
    budget = KineticEnergyBudget{};
    // Freezing moves the temperature seeds: restored, so that output does not change the solution
    Kokkos::View<rtype *> seeds;
    if (double_flux) {
        seeds = Kokkos::View<rtype *>("budget_T_seed", T_seed.extent(0));
        Kokkos::deep_copy(seeds, T_seed);
        freeze_thermodynamics();
        cells_frozen = true;
    }
    budget_pass = true;
    calc_rhs(state(), scratch, t);
    budget_pass = false;
    if (double_flux) {
        cells_frozen = false;
        Kokkos::deep_copy(T_seed, seeds);
    }
    const std::array<rtype, 4> local = {budget.convective, budget.viscous, budget.sgs, budget.pressure_work};
    const std::array<rtype, 4> total = comm::allreduce(local, comm::Op::SUM);
    return KineticEnergyBudget{total[0], total[1], total[2], total[3]};
}
