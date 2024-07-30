/**
 * @file time_integrator_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests for numerics/time_integrator
 * @version 0.1
 * @date 2023-12-24
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "test_utils.h"
#include "time_integrator.h"

#include "common_typedef.h"

const int N_CELLS = 2;
const rtype TOL = 1e-6;

void calc_rhs_test(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                   Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution,
                   Kokkos::View<rtype *[N_CONSERVATIVE]> rhs) {
    Kokkos::parallel_for(N_CELLS, KOKKOS_LAMBDA(const u_int8_t i) {
        for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
            rhs(i, j) = i * static_cast<rtype>(N_CONSERVATIVE) + j;
        }
    });
}

TEST(TimeIntegratorTest, FE) {
    FE integrator;
    std::vector<Kokkos::View<rtype *[N_CONSERVATIVE]>> solution_vec;
    std::vector<Kokkos::View<rtype *[N_CONSERVATIVE]>::HostMirror> h_solution_vec;
    Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution;
    Kokkos::View<rtype *[2][N_CONSERVATIVE]>::HostMirror h_face_solution;
    std::vector<Kokkos::View<rtype *[N_CONSERVATIVE]>> rhs_vec;
    std::vector<Kokkos::View<rtype *[N_CONSERVATIVE]>::HostMirror> h_rhs_vec;

    for (u_int8_t i = 0; i < integrator.get_n_solution_vectors(); i++) {
        Kokkos::View<rtype *[N_CONSERVATIVE]> solution("solution", N_CELLS);
        solution_vec.push_back(solution);
        h_solution_vec.push_back(Kokkos::create_mirror_view(solution));
    }
    Kokkos::resize(face_solution, N_CELLS);
    for (u_int8_t i = 0; i < integrator.get_n_rhs_vectors(); i++) {
        Kokkos::View<rtype *[N_CONSERVATIVE]> rhs("rhs", N_CELLS);
        rhs_vec.push_back(rhs);
        h_rhs_vec.push_back(Kokkos::create_mirror_view(rhs));
    }

    std::function<void(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                       Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution,
                       Kokkos::View<rtype *[N_CONSERVATIVE]> rhs)> rhs_func = calc_rhs_test;

    for (u_int8_t i = 0; i < N_CELLS; i++) {
        for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
            h_solution_vec[0](i, j) = i * static_cast<rtype>(N_CONSERVATIVE) + j;
        }
    }

    for (u_int8_t i = 0; i < integrator.get_n_solution_vectors(); i++){
        Kokkos::deep_copy(solution_vec[i], h_solution_vec[i]);
    }

    integrator.take_step(0.1, solution_vec, face_solution, rhs_vec, &rhs_func);

    for (u_int8_t i = 0; i < integrator.get_n_solution_vectors(); i++){
        Kokkos::deep_copy(h_solution_vec[i], solution_vec[i]);
    }
    for (u_int8_t i = 0; i < integrator.get_n_rhs_vectors(); i++){
        Kokkos::deep_copy(h_rhs_vec[i], rhs_vec[i]);
    }

    for (u_int8_t i = 0; i < N_CELLS; i++) {
        for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
            rtype icpj = i * static_cast<rtype>(N_CONSERVATIVE) + j;
            EXPECT_NEAR(h_rhs_vec[0](i, j), icpj, TOL);
            EXPECT_NEAR(h_solution_vec[0](i, j), 1.1 * icpj, TOL);
        }
    }
}

TEST(TimeIntegratorTest, RK4) {
    RK4 integrator;
    std::vector<Kokkos::View<rtype *[N_CONSERVATIVE]>> solution_vec;
    std::vector<Kokkos::View<rtype *[N_CONSERVATIVE]>::HostMirror> h_solution_vec;
    Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution;
    Kokkos::View<rtype *[2][N_CONSERVATIVE]>::HostMirror h_face_solution;
    std::vector<Kokkos::View<rtype *[N_CONSERVATIVE]>> rhs_vec;
    std::vector<Kokkos::View<rtype *[N_CONSERVATIVE]>::HostMirror> h_rhs_vec;

    for (u_int8_t i = 0; i < integrator.get_n_solution_vectors(); i++) {
        Kokkos::View<rtype *[N_CONSERVATIVE]> solution("solution", N_CELLS);
        solution_vec.push_back(solution);
        h_solution_vec.push_back(Kokkos::create_mirror_view(solution));
    }
    Kokkos::resize(face_solution, N_CELLS);
    for (u_int8_t i = 0; i < integrator.get_n_rhs_vectors(); i++) {
        Kokkos::View<rtype *[N_CONSERVATIVE]> rhs("rhs", N_CELLS);
        rhs_vec.push_back(rhs);
        h_rhs_vec.push_back(Kokkos::create_mirror_view(rhs));
    }

    std::function<void(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                       Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution,
                       Kokkos::View<rtype *[N_CONSERVATIVE]> rhs)> rhs_func = calc_rhs_test;

    for (u_int8_t i = 0; i < N_CELLS; i++) {
        for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
            h_solution_vec[0](i, j) = i * static_cast<rtype>(N_CONSERVATIVE) + j;
        }
    }

    for (u_int8_t i = 0; i < integrator.get_n_solution_vectors(); i++){
        Kokkos::deep_copy(solution_vec[i], h_solution_vec[i]);
    }

    integrator.take_step(0.1, solution_vec, face_solution, rhs_vec, &rhs_func);

    for (u_int8_t i = 0; i < integrator.get_n_solution_vectors(); i++){
        Kokkos::deep_copy(h_solution_vec[i], solution_vec[i]);
    }
    for (u_int8_t i = 0; i < integrator.get_n_rhs_vectors(); i++){
        Kokkos::deep_copy(h_rhs_vec[i], rhs_vec[i]);
    }

    for (u_int8_t i = 0; i < N_CELLS; i++) {
        for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
            rtype icpj = i * static_cast<rtype>(N_CONSERVATIVE) + j;
            EXPECT_NEAR(h_rhs_vec[0](i, j), icpj, TOL);
            EXPECT_NEAR(h_solution_vec[0](i, j), 1.1 * icpj, TOL);
        }
    }
}

TEST(TimeIntegratorTest, SSPRK3) {
    SSPRK3 integrator;
    std::vector<Kokkos::View<rtype *[N_CONSERVATIVE]>> solution_vec;
    std::vector<Kokkos::View<rtype *[N_CONSERVATIVE]>::HostMirror> h_solution_vec;
    Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution;
    Kokkos::View<rtype *[2][N_CONSERVATIVE]>::HostMirror h_face_solution;
    std::vector<Kokkos::View<rtype *[N_CONSERVATIVE]>> rhs_vec;
    std::vector<Kokkos::View<rtype *[N_CONSERVATIVE]>::HostMirror> h_rhs_vec;

    for (u_int8_t i = 0; i < integrator.get_n_solution_vectors(); i++) {
        Kokkos::View<rtype *[N_CONSERVATIVE]> solution("solution", N_CELLS);
        solution_vec.push_back(solution);
        h_solution_vec.push_back(Kokkos::create_mirror_view(solution));
    }
    Kokkos::resize(face_solution, N_CELLS);
    for (u_int8_t i = 0; i < integrator.get_n_rhs_vectors(); i++) {
        Kokkos::View<rtype *[N_CONSERVATIVE]> rhs("rhs", N_CELLS);
        rhs_vec.push_back(rhs);
        h_rhs_vec.push_back(Kokkos::create_mirror_view(rhs));
    }

    std::function<void(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                       Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution,
                       Kokkos::View<rtype *[N_CONSERVATIVE]> rhs)> rhs_func = calc_rhs_test;

    for (u_int8_t i = 0; i < N_CELLS; i++) {
        for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
            h_solution_vec[0](i, j) = i * static_cast<rtype>(N_CONSERVATIVE) + j;
        }
    }

    for (u_int8_t i = 0; i < integrator.get_n_solution_vectors(); i++){
        Kokkos::deep_copy(solution_vec[i], h_solution_vec[i]);
    }

    integrator.take_step(0.1, solution_vec, face_solution, rhs_vec, &rhs_func);

    for (u_int8_t i = 0; i < integrator.get_n_solution_vectors(); i++){
        Kokkos::deep_copy(h_solution_vec[i], solution_vec[i]);
    }
    for (u_int8_t i = 0; i < integrator.get_n_rhs_vectors(); i++){
        Kokkos::deep_copy(h_rhs_vec[i], rhs_vec[i]);
    }

    for (u_int8_t i = 0; i < N_CELLS; i++) {
        for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
            rtype icpj = i * static_cast<rtype>(N_CONSERVATIVE) + j;
            EXPECT_NEAR(h_rhs_vec[0](i, j), icpj, TOL);
            EXPECT_NEAR(h_solution_vec[0](i, j), 1.1 * icpj, TOL);
        }
    }
}
