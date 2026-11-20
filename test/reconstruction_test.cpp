/**
 * @file reconstruction_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests for gradients, limiters and face reconstruction.
 * @version 0.1
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "data_paths.h"
#include "test_fixtures.h"
#include "face_reconstruction.h"
#include "gradient.h"
#include "scalar_reconstruction.h"
#include "teno_scalars.h"
#include "thermo.h"

namespace {

// Linear field W(x, y) = c + gx * x + gy * y, positive for x, y in [0, 1]
constexpr rtype C[N_CONSERVATIVE] = {1.0_r, 0.3_r, -0.2_r, 2.0_r};
constexpr rtype GX[N_CONSERVATIVE] = {0.5, -1.0, 0.25, 0.75};
constexpr rtype GY[N_CONSERVATIVE] = {-0.25, 0.5, 1.5, -0.5};

rtype linear(uint8_t i, rtype x, rtype y) {
    return C[i] + GX[i] * x + GY[i] * y;
}

Kokkos::View<rtype *[N_CONSERVATIVE]> linear_cell_field(const Mesh & mesh) {
    Kokkos::View<rtype *[N_CONSERVATIVE]> W("W", mesh.n_cells);
    auto h_W = Kokkos::create_mirror_view(W);
    for (uint32_t i_cell = 0; i_cell < mesh.n_cells; i_cell++) {
        FOR_I_CONSERVATIVE h_W(i_cell, i) = linear(i, mesh.h_cell_coords(i_cell, 0),
                                                       mesh.h_cell_coords(i_cell, 1));
    }
    Kokkos::deep_copy(W, h_W);
    return W;
}

std::unique_ptr<MUSCL> make_muscl(std::shared_ptr<Mesh> mesh, const BoundaryData & bd,
                                  const std::string & limiter) {
    auto muscl = std::make_unique<MUSCL>();
    muscl->set_mesh(mesh);
    muscl->set_boundaries(bd);
    muscl->init(parse_toml("type = \"MUSCL\"\nlimiter = \"" + limiter + "\"\n"));
    return muscl;
}

class MeshTypes : public ::testing::TestWithParam<std::string> {};

} // namespace

TEST_P(MeshTypes, LSQGradientExactForLinearFieldInInterior) {
    auto mesh = make_mesh(GetParam(), 8, 7);
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::OUTLET_EXTRAPOLATION);
    auto W = linear_cell_field(*mesh);
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> grad("grad", mesh->n_cells);
    LSQGradientFunctor functor = make_gradient(*mesh, bd, W, grad);
    Kokkos::parallel_for(mesh->n_cells, functor);
    auto h_grad = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), grad);
    for (uint32_t i_cell = 0; i_cell < mesh->n_cells; i_cell++) {
        if (is_boundary_cell(*mesh, i_cell)) continue;
        FOR_I_CONSERVATIVE {
            EXPECT_NEAR(h_grad(i_cell, i, 0), GX[i], roundoff(1e-10)) << "cell " << i_cell << " var " << static_cast<int>(i);
            EXPECT_NEAR(h_grad(i_cell, i, 1), GY[i], roundoff(1e-10)) << "cell " << i_cell << " var " << static_cast<int>(i);
        }
    }
}

TEST_P(MeshTypes, LSQGradientExactForLinearFieldWithDirichletBoundaries) {
    // A Dirichlet state is the value at the face, so boundary cells are exact too
    auto mesh = make_mesh(GetParam(), 8, 7);
    std::vector<int32_t> face_bc(mesh->n_faces, -1);
    for (uint32_t f = 0; f < mesh->n_faces; f++) face_bc[f] = mesh->h_cells_of_face(f, 1) < 0 ? 0 : -1;
    BoundaryCondition dir;
    dir.type = BoundaryType::DIRICHLET;
    BoundaryData bd = make_boundary_data(*mesh, face_bc, {dir}, 1.4_r);
    auto h_index = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bd.face_state_index);
    auto h_state = Kokkos::create_mirror_view(bd.face_state);
    for (uint32_t f = 0; f < mesh->n_faces; f++) {
        if (h_index(f) < 0) continue;
        FOR_I_CONSERVATIVE h_state(h_index(f), i) = linear(i, mesh->h_face_coords(f, 0), mesh->h_face_coords(f, 1));
    }
    Kokkos::deep_copy(bd.face_state, h_state);
    auto W = linear_cell_field(*mesh);
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> grad("grad", mesh->n_cells);
    LSQGradientFunctor functor = make_gradient(*mesh, bd, W, grad);
    Kokkos::parallel_for(mesh->n_cells, functor);
    auto h_grad = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), grad);
    for (uint32_t i_cell = 0; i_cell < mesh->n_cells; i_cell++) {
        FOR_I_CONSERVATIVE {
            EXPECT_NEAR(h_grad(i_cell, i, 0), GX[i], roundoff(1e-10)) << "cell " << i_cell << " var " << static_cast<int>(i);
            EXPECT_NEAR(h_grad(i_cell, i, 1), GY[i], roundoff(1e-10)) << "cell " << i_cell << " var " << static_cast<int>(i);
        }
    }
}

TEST_P(MeshTypes, VertexLSQGradientExactForQuadraticFieldWithDirichletBoundaries) {
    // The viscous gradients fit a quadratic, so they are exact for quadratic
    // fields in every cell, including boundary cells with one-sided stencils
    // (a linear fit is off by O(h) there)
    auto quadratic = [](uint8_t i, rtype x, rtype y) {
        return linear(i, x, y) + (0.3_r + 0.1_r * i) * x * x - 0.4_r * x * y + (0.2_r - 0.1_r * i) * y * y;
    };
    auto mesh = make_mesh(GetParam(), 8, 7);
    std::vector<int32_t> face_bc(mesh->n_faces, -1);
    for (uint32_t f = 0; f < mesh->n_faces; f++) face_bc[f] = mesh->h_cells_of_face(f, 1) < 0 ? 0 : -1;
    BoundaryCondition dir;
    dir.type = BoundaryType::DIRICHLET;
    BoundaryData bd = make_boundary_data(*mesh, face_bc, {dir}, 1.4_r);
    auto h_index = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bd.face_state_index);
    auto h_state = Kokkos::create_mirror_view(bd.face_state);
    for (uint32_t f = 0; f < mesh->n_faces; f++) {
        if (h_index(f) < 0) continue;
        FOR_I_CONSERVATIVE h_state(h_index(f), i) = quadratic(i, mesh->h_face_coords(f, 0), mesh->h_face_coords(f, 1));
    }
    Kokkos::deep_copy(bd.face_state, h_state);
    Kokkos::View<rtype *[N_CONSERVATIVE]> W("W", mesh->n_cells);
    auto h_W = Kokkos::create_mirror_view(W);
    for (uint32_t c = 0; c < mesh->n_cells; c++) {
        FOR_I_CONSERVATIVE h_W(c, i) = quadratic(i, mesh->h_cell_coords(c, 0), mesh->h_cell_coords(c, 1));
    }
    Kokkos::deep_copy(W, h_W);
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> grad("grad", mesh->n_cells);
    LSQGradientFunctor faces = make_gradient(*mesh, bd, W, grad);
    Kokkos::parallel_for(mesh->n_cells, make_vertex_gradient(faces, *mesh));
    auto h_grad = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), grad);
    for (uint32_t c = 0; c < mesh->n_cells; c++) {
        const rtype x = mesh->h_cell_coords(c, 0), y = mesh->h_cell_coords(c, 1);
        FOR_I_CONSERVATIVE {
            EXPECT_NEAR(h_grad(c, i, 0), GX[i] + 2.0_r * (0.3_r + 0.1_r * i) * x - 0.4_r * y, roundoff(1e-9)) << "cell " << c;
            EXPECT_NEAR(h_grad(c, i, 1), GY[i] - 0.4_r * x + 2.0_r * (0.2_r - 0.1_r * i) * y, roundoff(1e-9)) << "cell " << c;
        }
    }
}

TEST_P(MeshTypes, UnlimitedMUSCLReproducesLinearFieldAtInteriorFaces) {
    auto mesh = make_mesh(GetParam(), 8, 7);
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::OUTLET_EXTRAPOLATION);
    auto W = linear_cell_field(*mesh);
    auto muscl = make_muscl(mesh, bd, "none");
    Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_W("face_W", mesh->n_faces, 1);
    muscl->calc_face_values(W, face_W);
    auto h_face_W = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), face_W);
    for (uint32_t i_face = 0; i_face < mesh->n_faces; i_face++) {
        for (uint8_t side = 0; side < 2; side++) {
            const int32_t c = mesh->h_cells_of_face(i_face, side);
            if (c < 0 || is_boundary_cell(*mesh, static_cast<uint32_t>(c))) continue;
            FOR_I_CONSERVATIVE {
                EXPECT_NEAR(h_face_W(i_face, 0, side, i),
                            linear(i, mesh->h_face_coords(i_face, 0), mesh->h_face_coords(i_face, 1)),
                            roundoff(1e-10)) << "face " << i_face << " side " << static_cast<int>(side);
            }
        }
    }
}

TEST_P(MeshTypes, BarthJespersenFaceValuesStayWithinNeighborBounds) {
    auto mesh = make_mesh(GetParam(), 10, 10);
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::OUTLET_EXTRAPOLATION);
    // Discontinuous field: a step in x plus a smooth bump
    Kokkos::View<rtype *[N_CONSERVATIVE]> W("W", mesh->n_cells);
    auto h_W = Kokkos::create_mirror_view(W);
    for (uint32_t i_cell = 0; i_cell < mesh->n_cells; i_cell++) {
        const rtype x = mesh->h_cell_coords(i_cell, 0);
        const rtype y = mesh->h_cell_coords(i_cell, 1);
        FOR_I_CONSERVATIVE h_W(i_cell, i) = 1.0_r + (x > 0.45_r ? 1.0_r : 0.0_r) + 0.3_r * std::sin(6.0_r * y + i);
    }
    Kokkos::deep_copy(W, h_W);
    auto muscl = make_muscl(mesh, bd, "barth_jespersen");
    Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_W("face_W", mesh->n_faces, 1);
    muscl->calc_face_values(W, face_W);
    auto h_face_W = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), face_W);
    for (uint32_t i_face = 0; i_face < mesh->n_faces; i_face++) {
        for (uint8_t side = 0; side < 2; side++) {
            const int32_t c = mesh->h_cells_of_face(i_face, side);
            if (c < 0) continue;
            FOR_I_CONSERVATIVE {
                rtype lo = h_W(c, i), hi = h_W(c, i);
                for (uint32_t k = 0; k < mesh->h_n_faces_of_cell(static_cast<uint32_t>(c)); k++) {
                    const uint32_t f = mesh->h_face_of_cell(static_cast<uint32_t>(c), k);
                    const int32_t c0 = mesh->h_cells_of_face(f, 0);
                    const int32_t c1 = mesh->h_cells_of_face(f, 1);
                    const int32_t nb = (c0 == c) ? c1 : c0;
                    if (nb < 0) continue;
                    lo = std::min(lo, h_W(nb, i));
                    hi = std::max(hi, h_W(nb, i));
                }
                EXPECT_GE(h_face_W(i_face, 0, side, i), double(lo) - roundoff(1e-12));
                EXPECT_LE(h_face_W(i_face, 0, side, i), double(hi) + roundoff(1e-12));
            }
        }
    }
}

TEST_P(MeshTypes, FirstOrderCopiesCellValuesAndSkipsGhostSide) {
    auto mesh = make_mesh(GetParam(), 4, 3);
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::OUTLET_EXTRAPOLATION);
    auto W = linear_cell_field(*mesh);
    FirstOrder fo;
    fo.set_mesh(mesh);
    fo.set_boundaries(bd);
    fo.init(parse_toml("type = \"FO\"\n"));
    Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_W("face_W", mesh->n_faces, 1);
    Kokkos::deep_copy(face_W, -7.0);
    fo.calc_face_values(W, face_W);
    auto h_face_W = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), face_W);
    auto h_W = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), W);
    for (uint32_t i_face = 0; i_face < mesh->n_faces; i_face++) {
        for (uint8_t side = 0; side < 2; side++) {
            const int32_t c = mesh->h_cells_of_face(i_face, side);
            FOR_I_CONSERVATIVE {
                EXPECT_EQ(h_face_W(i_face, 0, side, i), c < 0 ? -7.0_r : h_W(c, i));
            }
        }
    }
}

namespace {

// [HE, N2, O2]: helium is monatomic and light, so moving slope between it and air changes gamma and e0 at faces
const std::string HE_AIR = source_dir() + "/mechanisms/he_air.yaml";
constexpr uint32_t N_AIR = 3;
constexpr uint32_t HE = 0, N2 = 1, O2 = 2;
constexpr double T_CELLS = 300.0;

using HostThermo = chemistry::ThermoTable<Kokkos::HostSpace>;
using AirComposition = std::function<std::array<double, N_AIR>(double x, double y)>;

/** @brief [gamma, e0] of mass fractions Y at T_CELLS, as Mixture::cell_state defines them. */
std::array<double, 2> surrogates(const HostThermo & thermo, const std::array<double, N_AIR> & Y) {
    const chemistry::MassFractions y{Y.data()};
    double e, cv;
    thermo.e_cv(T_CELLS, y, e, cv);
    const double R = thermo.gas_constant(y);
    return {(cv + R) / cv, e - cv * T_CELLS};
}

/** @brief Cells at rest at T_CELLS: scalars [Y, gamma, e0] and W with p = rho R T. */
struct AirCells {
    ScalarView S;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
};

AirCells air_cells(const Mesh & mesh, const HostThermo & thermo, const AirComposition & composition) {
    AirCells cells{ScalarView("S", mesh.n_cells, N_AIR + 2), Kokkos::View<rtype *[N_CONSERVATIVE]>("W", mesh.n_cells)};
    auto h_S = Kokkos::create_mirror_view(cells.S);
    auto h_W = Kokkos::create_mirror_view(cells.W);
    for (uint32_t c = 0; c < mesh.n_cells; c++) {
        const auto Y = composition(double(mesh.h_cell_coords(c, 0)), double(mesh.h_cell_coords(c, 1)));
        const auto th = surrogates(thermo, Y);
        for (uint32_t k = 0; k < N_AIR; k++) h_S(c, k) = rtype(Y[k]);
        h_S(c, N_AIR) = rtype(th[0]);
        h_S(c, N_AIR + 1) = rtype(th[1]);
        FOR_I_CONSERVATIVE h_W(c, i) = 0.0_r;
        h_W(c, 0) = 1.0_r;
        h_W(c, N_DIM + 1) = rtype(thermo.gas_constant(chemistry::MassFractions{Y.data()}) * T_CELLS);
    }
    Kokkos::deep_copy(cells.S, h_S);
    Kokkos::deep_copy(cells.W, h_W);
    return cells;
}

double he_linear(double x, double y) { return 0.2 + 0.05 * x + 0.02 * y; }

/** @brief HE linear in space, N2 = trace in every third column of cells (zero elsewhere), O2 the rest. */
AirComposition trace_composition(const uint32_t nx, const double trace) {
    return [=](const double x, const double y) -> std::array<double, N_AIR> {
        std::array<double, N_AIR> Y;
        Y[HE] = he_linear(x, y);
        Y[N2] = static_cast<uint32_t>(x * nx) % 3 == 0 ? trace : 0.0;
        Y[O2] = 1.0 - Y[HE] - Y[N2];
        return Y;
    };
}

/** @brief Mass fractions at each face from each side, (face, side, k) on the host; one point per face. */
void fill_face_Y(const Mesh & mesh, const ScalarFaceValues & values, const Kokkos::View<rtype *[2][N_AIR]> & Y) {
    Kokkos::parallel_for(mesh.n_cells, KOKKOS_LAMBDA(const uint32_t c) {
        const uint32_t begin = values.offsets_faces_of_cell(c);
        const uint32_t n = values.offsets_faces_of_cell(c + 1) - begin;
        for (uint32_t j = 0; j < N_AIR; j++) {
            FacePointValues S[teno::MAX_FACES] = {};
            values.cell_values(c, j, S);
            for (uint32_t k = 0; k < n; k++) {
                const uint32_t f = values.faces_of_cell(begin + k);
                Y(f, values.cells_of_face(f, 0) == static_cast<int32_t>(c) ? 0 : 1, j) = S[k][0];
            }
        }
    });
}

auto face_Y(const Mesh & mesh, const ScalarFaceValues & values) {
    Kokkos::View<rtype *[2][N_AIR]> Y("face_Y", mesh.n_faces);
    fill_face_Y(mesh, values, Y);
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Y);
}

/** @brief Face values from each side of each face on the host (one point per face). */
struct FaceStates {
    decltype(face_Y(std::declval<const Mesh &>(), std::declval<const ScalarFaceValues &>())) Y;  // (face, side, k)
    Kokkos::View<rtype **[2][N_CONSERVATIVE]>::host_mirror_type W;
    Kokkos::View<rtype **[2][2]>::host_mirror_type thermo;  // [gamma, e0]
};

/**
 * @brief At every face of every interior cell, the state of the face
 *        composition at the cells' temperature: mass fractions in [0, 1]
 *        summing to one, e0 (linear in Y at fixed T, so to round-off),
 *        gamma within gamma_tol and p / (rho R) within T_tol (neither is
 *        linear in Y).
 */
void expect_face_state_of_face_composition(const Mesh & mesh, const HostThermo & thermo, const FaceStates & face,
                                           const double gamma_tol, const double T_tol, const std::string & what) {
    const auto & Y = face.Y;
    for (uint32_t f = 0; f < mesh.n_faces; f++) {
        for (uint8_t side = 0; side < 2; side++) {
            const int32_t c = mesh.h_cells_of_face(f, side);
            if (c < 0 || is_boundary_cell(mesh, static_cast<uint32_t>(c))) continue;
            std::array<double, N_AIR> Y_f;
            double sum = 0.0;
            for (uint32_t k = 0; k < N_AIR; k++) {
                Y_f[k] = double(Y(f, side, k));
                EXPECT_GE(Y_f[k], 0.0) << what << " face " << f << " species " << k;
                EXPECT_LE(Y_f[k], 1.0) << what << " face " << f << " species " << k;
                sum += Y_f[k];
            }
            EXPECT_NEAR(sum, 1.0, roundoff(1e-15)) << what << " face " << f;
            const auto th = surrogates(thermo, Y_f);
            EXPECT_NEAR(double(face.thermo(f, 0, side, 0)), th[0], gamma_tol) << what << " face " << f;
            EXPECT_NEAR(double(face.thermo(f, 0, side, 1)), th[1], roundoff(1e-12) * std::abs(th[1]))
                << what << " face " << f;
            const double R = thermo.gas_constant(chemistry::MassFractions{Y_f.data()});
            EXPECT_NEAR(double(face.W(f, 0, side, N_DIM + 1)) / (double(face.W(f, 0, side, 0)) * R), T_CELLS, T_tol)
                << what << " face " << f;
        }
    }
}

/** @brief HE at every face of every interior cell: its linear field, so the shared limiter stayed at one. */
template <typename FaceY>
void expect_he_linear(const Mesh & mesh, const FaceY & Y, const std::string & what) {
    for (uint32_t f = 0; f < mesh.n_faces; f++) {
        for (uint8_t side = 0; side < 2; side++) {
            const int32_t c = mesh.h_cells_of_face(f, side);
            if (c < 0 || is_boundary_cell(mesh, static_cast<uint32_t>(c))) continue;
            EXPECT_NEAR(double(Y(f, side, HE)),
                        he_linear(double(mesh.h_face_coords(f, 0)), double(mesh.h_face_coords(f, 1))),
                        std::max(1.0e-8, roundoff(1e-14))) << what << " face " << f;
        }
    }
}

/** @brief gamma at every face point, (face, q, side). */
template <typename Eval>
Kokkos::View<rtype **[2]> face_gamma(const Mesh & mesh, const Eval & values, const uint8_t nq) {
    Kokkos::View<rtype **[2]> gamma("gamma", mesh.n_faces, nq);
    Kokkos::parallel_for(mesh.n_cells, KOKKOS_LAMBDA(const uint32_t c) {
        FacePointValues G[teno::MAX_FACES] = {};
        values.cell_values(c, N_AIR, G);
        for (uint8_t k = 0; k < values.n_faces(c); k++) {
            const uint32_t f = values.faces_of_cell(values.offsets_faces_of_cell(c) + k);
            const uint8_t side = values.cells_of_face(f, 0) == static_cast<int32_t>(c) ? 0 : 1;
            for (uint8_t q = 0; q < nq; q++) gamma(f, q, side) = G[k][q];
        }
    });
    return gamma;
}

/** @brief Smallest mass fraction at any face point of any cell. */
template <typename Eval>
rtype min_face_Y(const Mesh & mesh, const Eval & values, const uint8_t nq) {
    rtype result;
    Kokkos::parallel_reduce(mesh.n_cells, KOKKOS_LAMBDA(const uint32_t c, rtype & lo) {
        const uint32_t n = values.offsets_faces_of_cell(c + 1) - values.offsets_faces_of_cell(c);
        for (uint32_t j = 0; j < N_AIR; j++) {
            FacePointValues S[teno::MAX_FACES] = {};
            values.cell_values(c, j, S);
            for (uint32_t k = 0; k < n; k++) {
                for (uint8_t q = 0; q < nq; q++) lo = Kokkos::fmin(lo, S[k][q]);
            }
        }
    }, Kokkos::Min<rtype>(result));
    return result;
}

class AirScalars : public ::testing::Test {
    protected:
        const HostThermo host = chemistry::make_thermo_table<Kokkos::HostSpace>(chemistry::read_mechanism(HE_AIR));
        const chemistry::ThermoTable<> thermo = chemistry::make_thermo_table(chemistry::read_mechanism(HE_AIR));
        static constexpr uint32_t NX = 9, NY = 8;
        std::shared_ptr<Mesh> mesh = make_mesh("cartesian", NX, NY);
        BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::OUTLET_EXTRAPOLATION);

        /** @brief Face states of the cells of a composition, with the flow block's MUSCL and limiter. */
        FaceStates reconstruct(const std::string & limiter, const AirComposition & composition) {
            auto muscl = make_muscl(mesh, bd, limiter);
            ScalarReconstruction recon;
            recon.init(mesh, bd, thermo, *muscl);
            const AirCells cells = air_cells(*mesh, host, composition);
            Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_W("face_W", mesh->n_faces, 1);
            Kokkos::View<rtype **[2][2]> face_thermo("face_thermo", mesh->n_faces, 1);
            muscl->calc_face_values(cells.W, face_W);
            recon.calc(cells.S, cells.W, face_W, face_thermo);
            return {face_Y(*mesh, recon.face_values(cells.S)),
                    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), face_W),
                    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), face_thermo)};
        }
};

class ScalarLimiters : public AirScalars, public ::testing::WithParamInterface<std::string> {};

/**
 * @brief Tolerance of the face temperature p / (rho R): tol without a
 *        limiter, none with one, since the flow block then limits the
 *        pressure, which varies with the composition here, by its own value.
 */
double face_T_tol(const std::string & limiter, const double tol) {
    return limiter == "none" ? tol : std::numeric_limits<double>::infinity();
}

} // namespace

TEST_P(ScalarLimiters, RoundOffTraceSpeciesKeepsTheOtherScalarsSecondOrder) {
    // A species absent from a cell (0) next to round-off (1e-34) must not cut
    // the cell's shared limiter: the linear HE stays exact at its faces
    const FaceStates face = reconstruct(GetParam(), trace_composition(NX, 1.0e-34));
    expect_he_linear(*mesh, face.Y, GetParam());
    expect_face_state_of_face_composition(*mesh, host, face, 1.0e-5, std::max(1.0e-6, roundoff(1.0e-12)), GetParam());
}

TEST_F(AirScalars, TraceSpeciesAtItsBoundLimitsOnlyItself) {
    // N2 = 1e-6 next to 0 needs its slope cut at the faces toward the zeros,
    // and O2 takes it: HE keeps its slope, and the face state follows the
    // face composition
    for (const char * limiter : {"none", "venkatakrishnan"}) {
        const FaceStates face = reconstruct(limiter, trace_composition(NX, 1.0e-6));
        expect_he_linear(*mesh, face.Y, limiter);
        expect_face_state_of_face_composition(*mesh, host, face, 1.0e-5, std::max(1.0e-6, roundoff(1.0e-12)), limiter);
    }
}

TEST_F(AirScalars, CutsUpToOwnCutLoLimitOnlyTheirSpeciesAndFromOwnCutHiAllScalars) {
    // The N2 trace next to two zeros: its cell's neighbor without it needs
    // its N2 cut by trace / 4 at the far face. Below OWN_CUT_LO only N2 is
    // cut and HE keeps its linear profile; above OWN_CUT_HI, the cut limits
    // every scalar, HE included, since cutting N2 alone where it vanishes
    // over a cell would diffuse it alone
    expect_he_linear(*mesh, reconstruct("none", trace_composition(NX, 0.99 * 4.0 * double(OWN_CUT_LO))).Y,
                     "below OWN_CUT_LO");
    const FaceStates face = reconstruct("none", trace_composition(NX, 1.01 * 4.0 * double(OWN_CUT_HI)));
    double HE_off = 0.0;
    for (uint32_t f = 0; f < mesh->n_faces; f++) {
        for (uint8_t side = 0; side < 2; side++) {
            const int32_t c = mesh->h_cells_of_face(f, side);
            if (c < 0 || is_boundary_cell(*mesh, static_cast<uint32_t>(c))) continue;
            HE_off = std::max(HE_off, std::abs(double(face.Y(f, side, HE)) -
                                               he_linear(double(mesh->h_face_coords(f, 0)),
                                                         double(mesh->h_face_coords(f, 1)))));
        }
    }
    EXPECT_GT(HE_off, 1.0e-3) << "above OWN_CUT_HI";
}

TEST_F(AirScalars, FaceValuesAreContinuousAsACutCrossesFromOwnToShared) {
    // The N2 trace swept so that its cut goes from OWN_CUT_LO / 2 to
    // 2 OWN_CUT_HI in small steps: the face values of every scalar must move
    // in small steps too. A switch between the species' own slope and the
    // shared limiter moves HE and the thermodynamics of the cells next to the
    // trace by their whole change at once (issue #1242)
    constexpr size_t STEPS = 80;
    const double c_first = 0.5 * double(OWN_CUT_LO), ratio = 4.0 * double(OWN_CUT_HI) / double(OWN_CUT_LO);
    std::vector<FaceStates> faces;
    for (size_t i = 0; i <= STEPS; i++) {
        const double cut = c_first * std::pow(ratio, double(i) / double(STEPS));
        faces.push_back(reconstruct("none", trace_composition(NX, 4.0 * cut)));
    }
    const auto value = [&](const FaceStates & face, const uint32_t f, const uint8_t side, const uint32_t j) {
        return j < N_AIR ? double(face.Y(f, side, j)) : double(face.thermo(f, 0, side, j - N_AIR));
    };
    for (uint32_t j = 0; j < N_AIR + 2; j++) {
        double range = 0.0, step = 0.0;
        for (uint32_t f = 0; f < mesh->n_faces; f++) {
            for (uint8_t side = 0; side < 2; side++) {
                const int32_t c = mesh->h_cells_of_face(f, side);
                if (c < 0 || is_boundary_cell(*mesh, static_cast<uint32_t>(c))) continue;
                const double first = value(faces[0], f, side, j);
                for (size_t i = 1; i <= STEPS; i++) {
                    const double v = value(faces[i], f, side, j);
                    range = std::max(range, std::abs(v - first));
                    step = std::max(step, std::abs(v - value(faces[i - 1], f, side, j)));
                }
            }
        }
        EXPECT_GT(range, 0.0) << "scalar " << j;
        EXPECT_LT(step, 0.1 * range) << "scalar " << j << ": the largest step of a face value against its whole change";
    }
}

TEST_P(ScalarLimiters, MassFractionsStayInBoundsWithTheirThermodynamics) {
    // N2 = 0.5 next to 0 is cut beyond OWN_CUT_HI, which limits all scalars;
    // Barth-Jespersen limits a trace (1e-6) without any cut at all
    std::vector<double> traces = {0.5};
    if (GetParam() == "barth_jespersen") traces.push_back(1.0e-6);
    for (const double trace : traces) {
        const FaceStates face = reconstruct(GetParam(), trace_composition(NX, trace));
        // The shared limiter leaves the flow block's density alone, which is not that of the face composition
        expect_face_state_of_face_composition(*mesh, host, face, 1.0e-3, std::numeric_limits<double>::infinity(),
                                              GetParam() + " trace " + std::to_string(trace));
    }
}

TEST_P(ScalarLimiters, SlopeGivenToAMajorSpeciesBesidesTheComplementCarriesItsThermodynamics) {
    // HE = 1e-6 in every third column against N2 = 0.3 - HE, in O2 = 0.7: a
    // cut HE gives its slope to O2, the most abundant species, and not to its
    // complement N2, so the face gamma, e0 and density must move from HE's
    // to O2's. Barth-Jespersen keeps HE between its neighbors, so it never cuts.
    SKIP_IN_SINGLE_PRECISION("O2 = 0.7 moves by less than its round-off");
    constexpr double TRACE = 1.0e-6;
    const AirComposition composition = [](const double x, double) -> std::array<double, N_AIR> {
        std::array<double, N_AIR> Y;
        Y[HE] = static_cast<uint32_t>(x * NX) % 3 == 0 ? TRACE : 0.0;
        Y[N2] = 0.3 - Y[HE];
        Y[O2] = 0.7;
        return Y;
    };
    const FaceStates face = reconstruct(GetParam(), composition);
    expect_face_state_of_face_composition(*mesh, host, face, std::max(1.0e-8, roundoff(1.0e-14)),
                                          face_T_tol(GetParam(), std::max(1.0e-5, roundoff(1.0e-12))), GetParam());
    double O2_moved = 0.0;
    for (uint32_t f = 0; f < mesh->n_faces; f++) {
        for (uint8_t side = 0; side < 2; side++) {
            if (mesh->h_cells_of_face(f, side) < 0) continue;
            O2_moved = std::max(O2_moved, std::abs(double(face.Y(f, side, O2)) - 0.7));
        }
    }
    if (GetParam() == "barth_jespersen") {
        EXPECT_LT(O2_moved, roundoff(1e-15));
    } else {
        EXPECT_GT(O2_moved, 0.1 * TRACE);
    }
}

INSTANTIATE_TEST_SUITE_P(Reconstruction, ScalarLimiters,
                         ::testing::Values("none", "venkatakrishnan", "barth_jespersen"));

TEST(TENOScalarsTest, RoundOffTraceSpeciesKeepsThetaAtOneAndStaysNonNegative) {
    constexpr uint32_t NX = 12, NY = 12;
    auto mesh = make_mesh("cartesian", NX, NY);
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::OUTLET_EXTRAPOLATION);
    auto teno = std::make_unique<TENO>();
    teno->set_mesh(mesh);
    teno->set_boundaries(bd);
    teno->init(parse_toml("type = \"TENO\"\norder = 5\n"));
    const uint8_t nq = teno->n_face_quadrature_points();
    const auto mech = chemistry::read_mechanism(HE_AIR);
    const AirCells cells = air_cells(*mesh, chemistry::make_thermo_table<Kokkos::HostSpace>(mech),
                                     trace_composition(NX, 1.0e-34));
    Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_W("face_W", mesh->n_faces, nq);
    teno->calc_face_values(cells.W, face_W);
    ScalarReconstruction recon;
    recon.init(mesh, bd, chemistry::make_thermo_table(mech), *teno);
    Kokkos::View<rtype **[2][2]> face_thermo("face_thermo", mesh->n_faces, nq);
    recon.calc(cells.S, cells.W, face_W, face_thermo);
    // theta = 1: gamma at every face point is its unscaled reconstruction
    const auto unscaled = make_teno_scalar_values<4>(*teno, *mesh, bd, cells.S, Kokkos::View<rtype *>(), N_AIR);
    auto h_thermo = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), face_thermo);
    auto h_gamma = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), face_gamma(*mesh, unscaled, nq));
    for (uint32_t f = 0; f < mesh->n_faces; f++) {
        for (uint8_t side = 0; side < 2; side++) {
            if (mesh->h_cells_of_face(f, side) < 0) continue;
            for (uint8_t q = 0; q < nq; q++) {
                EXPECT_NEAR(h_thermo(f, q, side, 0), h_gamma(f, q, side), roundoff(1e-14))
                    << "face " << f << " point " << static_cast<int>(q);
            }
        }
    }
    EXPECT_GE(min_face_Y(*mesh, unscaled, nq), 0.0_r);
}

INSTANTIATE_TEST_SUITE_P(Reconstruction, MeshTypes,
                         ::testing::Values("cartesian", "cartesian_tri", "wedge"));
