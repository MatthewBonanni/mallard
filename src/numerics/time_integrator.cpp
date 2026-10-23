/**
 * @file time_integrator.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Time integrator class implementations.
 * @version 0.2
 * @date 2023-12-22
 *
 * @copyright Copyright (c) 2023 Matthew Bonanni
 *
 */

#include "time_integrator.h"

#include <initializer_list>
#include <utility>


void axpby(const rtype a_host, const State & x, const rtype b, const State & y, const rtype * a_device) {
    StateView x_flow = x.flow, y_flow = y.flow;
    Kokkos::parallel_for("axpby", x_flow.extent(0), KOKKOS_LAMBDA(const uint32_t i_cell) {
        const rtype a = a_device ? *a_device : a_host;
        FOR_I_CONSERVATIVE y_flow(i_cell, i) = a * x_flow(i_cell, i) + b * y_flow(i_cell, i);
    });
    if (x.species.span() == 0) return;
    const rtype * x_s = x.species.data();
    rtype * y_s = y.species.data();
    Kokkos::parallel_for("axpby_species", x.species.span(), KOKKOS_LAMBDA(const size_t k) {
        const rtype a = a_device ? *a_device : a_host;
        y_s[k] = a * x_s[k] + b * y_s[k];
    });
}

void lerp(const rtype a, const State & x, const State & y) {
    StateView x_flow = x.flow, y_flow = y.flow;
    Kokkos::parallel_for("lerp", x_flow.extent(0), KOKKOS_LAMBDA(const uint32_t i_cell) {
        FOR_I_CONSERVATIVE y_flow(i_cell, i) += a * (x_flow(i_cell, i) - y_flow(i_cell, i));
    });
    if (x.species.span() == 0) return;
    const rtype * x_s = x.species.data();
    rtype * y_s = y.species.data();
    Kokkos::parallel_for("lerp_species", x.species.span(), KOKKOS_LAMBDA(const size_t k) {
        y_s[k] += a * (x_s[k] - y_s[k]);
    });
}

namespace {

const rtype * scale(const rtype * dt_device, const DtScale k) { return dt_device ? dt_device + k : nullptr; }

} // namespace

FE::FE() {
    type = TimeIntegratorType::FE;
    n_solution_vectors = 1;
    n_rhs_vectors = 1;
}

void FE::take_step(const rtype t, const rtype dt,
                   std::vector<State> & solution_vec,
                   std::vector<State> & rhs_vec,
                   const RHSFunction & calc_rhs, const rtype * dt_device) {
    State U = solution_vec[0];
    State k1 = rhs_vec[0];
    calc_rhs(U, k1, t);
    axpby(dt, k1, 1.0, U, scale(dt_device, DT));
}

RK4::RK4() {
    type = TimeIntegratorType::RK4;
    n_solution_vectors = 2;
    n_rhs_vectors = 4;
}

void RK4::take_step(const rtype t, const rtype dt,
                    std::vector<State> & solution_vec,
                    std::vector<State> & rhs_vec,
                    const RHSFunction & calc_rhs, const rtype * dt_device) {
    State U = solution_vec[0];
    State U_temp = solution_vec[1];
    State k1 = rhs_vec[0];
    State k2 = rhs_vec[1];
    State k3 = rhs_vec[2];
    State k4 = rhs_vec[3];

    calc_rhs(U, k1, t);
    deep_copy(Kokkos::DefaultExecutionSpace(), U_temp, U);
    axpby(0.5_r * dt, k1, 1.0_r, U_temp, scale(dt_device, DT_HALF));

    calc_rhs(U_temp, k2, t + 0.5_r * dt);
    deep_copy(Kokkos::DefaultExecutionSpace(), U_temp, U);
    axpby(0.5_r * dt, k2, 1.0_r, U_temp, scale(dt_device, DT_HALF));

    calc_rhs(U_temp, k3, t + 0.5_r * dt);
    deep_copy(Kokkos::DefaultExecutionSpace(), U_temp, U);
    axpby(dt, k3, 1.0, U_temp, scale(dt_device, DT));

    calc_rhs(U_temp, k4, t + dt);
    axpby(dt / 6.0_r, k1, 1.0_r, U, scale(dt_device, DT_SIXTH));
    axpby(dt / 3.0_r, k2, 1.0_r, U, scale(dt_device, DT_THIRD));
    axpby(dt / 3.0_r, k3, 1.0_r, U, scale(dt_device, DT_THIRD));
    axpby(dt / 6.0_r, k4, 1.0_r, U, scale(dt_device, DT_SIXTH));
}

SSPRK3::SSPRK3() {
    type = TimeIntegratorType::SSPRK3;
    n_solution_vectors = 2;
    n_rhs_vectors = 1;
}

void SSPRK3::take_step(const rtype t, const rtype dt,
                       std::vector<State> & solution_vec,
                       std::vector<State> & rhs_vec,
                       const RHSFunction & calc_rhs, const rtype * dt_device) {
    State U = solution_vec[0];
    State U_temp = solution_vec[1];
    State k = rhs_vec[0];

    // U1 = U + dt L(U)
    calc_rhs(U, k, t);
    deep_copy(Kokkos::DefaultExecutionSpace(), U_temp, U);
    axpby(dt, k, 1.0, U_temp, scale(dt_device, DT));

    // U2 = 3/4 U + 1/4 (U1 + dt L(U1))
    calc_rhs(U_temp, k, t + dt);
    axpby(dt, k, 1.0, U_temp, scale(dt_device, DT));
    axpby(0.75, U, 0.25, U_temp);

    // U^{n+1} = 1/3 U + 2/3 (U2 + dt L(U2))
    calc_rhs(U_temp, k, t + 0.5_r * dt);
    axpby(dt, k, 1.0, U_temp, scale(dt_device, DT));
    lerp(2.0_r / 3.0_r, U_temp, U);
}

namespace {

/** @brief One stage of LowStorage3S on a block of n values (see LowStorageTableau). */
struct Stage3S {
    rtype beta, gamma1, gamma2, delta, e;
    bool first;    // Y_1: U^n = Y, Y += beta dt k
    bool s_live;   // S holds earlier deviations (else it is zero)
};

void stage_3s(const Stage3S st, const size_t n, rtype * y, rtype * s, rtype * y0, const rtype * k, rtype * err,
              const rtype dt_host, const rtype * dt_device) {
    Kokkos::parallel_for("low_storage_stage", n, KOKKOS_LAMBDA(const size_t j) {
        const rtype dt = dt_device ? *dt_device : dt_host;
        const rtype h = st.beta * dt;
        if (err) {
            const rtype h_e = st.e * dt;
            err[j] = st.first ? h_e * k[j] : err[j] + h_e * k[j];
        }
        if (st.first) {
            const rtype y_j = y[j];
            y0[j] = y_j;
            y[j] = y_j + h * k[j];
            return;
        }
        const rtype base = y0[j], d = y[j] - base;
        rtype s_j = 0.0;
        if (st.delta != 0.0_r) {
            s_j = st.s_live ? s[j] + st.delta * d : st.delta * d;
            s[j] = s_j;
        } else if (st.gamma2 != 0.0_r) {
            s_j = s[j];
        }
        y[j] = base + st.gamma1 * d + st.gamma2 * s_j + h * k[j];
    });
}

/** @brief One stage of the 2N method: D = A D + dt k (D = dt k if first), Y += B D. */
void stage_2n(const rtype a, const rtype b, const bool first, const size_t n, rtype * y, rtype * d, const rtype * k,
              const rtype dt_host, const rtype * dt_device) {
    Kokkos::parallel_for("low_storage_2n_stage", n, KOKKOS_LAMBDA(const size_t j) {
        const rtype dt = dt_device ? *dt_device : dt_host;
        const rtype d_j = first ? dt * k[j] : a * d[j] + dt * k[j];
        d[j] = d_j;
        y[j] += b * d_j;
    });
}

std::vector<rtype> coefficients(const std::initializer_list<double> values) {
    return std::vector<rtype>(values.begin(), values.end());
}

rtype * data_or_null(const State & s, const bool species) {
    if (species) return s.species.span() > 0 ? s.species.data() : nullptr;
    return s.flow.data();
}

} // namespace

LowStorage3S::LowStorage3S(const TimeIntegratorType type_, LowStorageTableau tableau_) : tableau(std::move(tableau_)) {
    type = type_;
    n_solution_vectors = 3;
    n_rhs_vectors = tableau.e.empty() ? 1 : 2;
}

void LowStorage3S::take_step(const rtype t, const rtype dt,
                             std::vector<State> & solution_vec,
                             std::vector<State> & rhs_vec,
                             const RHSFunction & calc_rhs, const rtype * dt_device) {
    const State U = solution_vec[0], S = solution_vec[1], U0 = solution_vec[2];
    const State k = rhs_vec[0];
    const bool embedded = !tableau.e.empty();
    const rtype * dt_d = scale(dt_device, DT);
    bool s_live = false;
    for (size_t i = 0; i < tableau.beta.size(); i++) {
        calc_rhs(U, k, t + tableau.c[i] * dt);
        const Stage3S st{tableau.beta[i], i ? tableau.gamma1[i] : 0.0_r, i ? tableau.gamma2[i] : 0.0_r,
                         i ? tableau.delta[i] : 0.0_r, embedded ? tableau.e[i] : 0.0_r, i == 0, s_live};
        for (const bool species : {false, true}) {
            const size_t n = species ? U.species.span() : U.flow.span();
            if (n == 0) continue;
            stage_3s(st, n, data_or_null(U, species), data_or_null(S, species), data_or_null(U0, species),
                     data_or_null(k, species), embedded ? data_or_null(rhs_vec[1], species) : nullptr, dt, dt_d);
        }
        s_live = s_live || st.delta != 0.0_r;
    }
}

SSPRK54::SSPRK54()
    // Spiteri & Ruuth's Shu-Osher form, re-solved to 50 digits (order
    // conditions to 1e-50, SSP coefficient 1.50818004918983792); delta_4 and
    // gamma3 vanish, so U^n, the solution and S = alpha_52 (Y_2 - U^n) suffice
    : LowStorage3S(TimeIntegratorType::SSPRK54, {
          .beta = coefficients({0.39175222657188905833, 0.36841059305037202075, 0.25189177427169263984,
                   0.54497475022851992204, 0.22600748323690765039}),
          .gamma1 = coefficients({0.0, 0.55562950634876743545, 0.37989814851159695606, 0.82192004560686881178,
                     0.50358094716548092920}),
          .gamma2 = coefficients({0.0, 0.0, 0.0, 0.0, 1.0}),
          .delta = coefficients({0.0, 0.0, 0.51723167197058318705, 0.0, 0.0}),
          .c = coefficients({0.0, 0.39175222657188905833, 0.58607968931154123111, 0.47454236312139913362,
                0.93501063096765159845}),
          .e = coefficients({}),
      }) {}

RDPK3Sp35::RDPK3Sp35()
    // Ranocha et al. (2022), as in OrdinaryDiffEq.jl, in the deviation form
    // (gamma3 = 1 - gamma1 - gamma2 (1 + sum delta) to 1e-37)
    : LowStorage3S(TimeIntegratorType::RDPK3SP35, {
          .beta = coefficients({2.300285062878154351930669430512780706e-01, 3.021457892454169700189445968126242994e-01,
                   8.025601039472704213300183888573974531e-01, 4.362158997637629844305216319994356355e-01,
                   1.129268494470295369172265188216779157e-01}),
          .gamma1 = coefficients({0.0, 2.587669070352079020144955303389306026e-01, -1.324366873994502973977035353758550057e-01,
                     5.055601231460399101814291350373559483e-02, 5.670552807902877312521811889846000976e-01}),
          .gamma2 = coefficients({0.0, 5.528418745102160639901976698795928733e-01, 6.731844400389673824374042790213570079e-01,
                     2.803103804507635075215805236096803381e-01, 5.521508873507393276457754945308880998e-01}),
          .delta = coefficients({0.0, 3.407687209321455242558804921815861422e-01, 3.414399280584625023244387687873774697e-01,
                    7.229302732875589702087936723400941329e-01, 0.0}),
          .c = coefficients({0.0, 2.300285062878154351930669430512780706e-01, 4.050049049262914975700372321130661410e-01,
                8.947823877926760224705450466361360720e-01, 7.235108137218888081489570284485201518e-01}),
          // b_hat - b
          .e = coefficients({-1.0156819201580692e-02, 5.868722797238998e-03, 1.2658792818943663e-02, -2.4554999795976132e-03,
                -5.915196435004355e-03}),
      }) {
    error_order = 3;
    controller_beta = {0.64_r, -0.31_r, 0.04_r};
}

CK45::CK45() {
    type = TimeIntegratorType::CK45;
    n_solution_vectors = 1;
    n_rhs_vectors = 2;
}

void CK45::take_step(const rtype t, const rtype dt,
                     std::vector<State> & solution_vec,
                     std::vector<State> & rhs_vec,
                     const RHSFunction & calc_rhs, const rtype * dt_device) {
    static constexpr double A[5] = {0.0, -567301805773.0 / 1357537059087.0, -2404267990393.0 / 2016746695238.0,
                                    -3550918686646.0 / 2091501179385.0, -1275806237668.0 / 842570457699.0};
    static constexpr double B[5] = {1432997174477.0 / 9575080441755.0, 5161836677717.0 / 13612068292357.0,
                                    1720146321549.0 / 2090206949498.0, 3134564353537.0 / 4481467310338.0,
                                    2277821191437.0 / 14882151754819.0};
    static constexpr double C[5] = {0.0, 1432997174477.0 / 9575080441755.0, 2526269341429.0 / 6820363962896.0,
                                    2006345519317.0 / 3224310063776.0, 2802321613138.0 / 2924317926251.0};
    const State U = solution_vec[0], k = rhs_vec[0], D = rhs_vec[1];
    const rtype * dt_d = scale(dt_device, DT);
    for (int i = 0; i < 5; i++) {
        calc_rhs(U, k, t + static_cast<rtype>(C[i]) * dt);
        for (const bool species : {false, true}) {
            const size_t n = species ? U.species.span() : U.flow.span();
            if (n == 0) continue;
            stage_2n(static_cast<rtype>(A[i]), static_cast<rtype>(B[i]), i == 0, n, data_or_null(U, species),
                     data_or_null(D, species), data_or_null(k, species), dt, dt_d);
        }
    }
}
