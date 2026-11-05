/**
 * @file initial_state_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Initial states from layered regions and tabulated data: exact volume
 *        fractions, layering rules, interpolation orders, nearest-point search
 *        and restarts with regions.
 * @version 0.1
 * @date 2026-10-23
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "gmsh_fixtures.h"
#include "overlap.h"
#include "scratch.h"
#include "solver.h"
#include "tabulated.h"
#include "test_fixtures.h"
#include "test_utils.h"

namespace {

constexpr double PI = 3.14159265358979323846;

const std::string UNIFORM = std::string("type = \"analytical\"\nrho = \"1\"\np = \"1\"\nu = ") +
                            (N_DIM == 2 ? "[\"0\", \"0\"]\n" : "[\"0\", \"0\", \"0\"]\n");

/** @brief A unit box (cube) of n cells per side starting from base (rho = 1, u = 0, p = 1), with init added to [initialize]. */
std::string box(const std::string & mesh, const uint32_t n, const std::string & init, const std::string & base = UNIFORM) {
    std::ostringstream s;
    s << "[run]\nn_steps = 1\ncfl = 0.25\n"
      << "[mesh]\ntype = \"" << mesh << "\"\nNx = " << n << "\nNy = " << n << "\nLx = 1.0\nLy = 1.0\n"
      << (N_DIM == 3 ? "Nz = " + std::to_string(n) + "\nLz = 1.0\n" : "")
      << "[initialize]\n" << base << init;
    for (const char * name : {"left", "right", "top", "bottom", "front", "back"}) {
        if (N_DIM == 2 && (std::string(name) == "front" || std::string(name) == "back")) continue;
        s << "[[boundaries]]\nname = \"" << name << "\"\ntype = \"symmetry\"\n";
    }
    s << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 1000000\n";
    return s.str();
}

std::unique_ptr<Solver> initialized(const std::string & input) {
    auto solver = std::make_unique<Solver>();
    solver->set_distributed(false);
    solver->init(parse_toml(input));
    return solver;
}

const char * TRI = N_DIM == 2 ? "cartesian_tri" : "cartesian_tet";

/** @brief Integral of rho over the owned cells. */
double mass(const Solver & s) {
    double m = 0.0;
    for (uint32_t c = 0; c < s.get_mesh()->n_owned(); c++) m += double(s.h_conservatives(c, 0)) * double(s.get_mesh()->h_cell_volume(c));
    return m;
}

/** @brief Cells whose density is neither 1 nor 2 exactly. */
uint32_t mixed_cells(const Solver & s) {
    uint32_t n = 0;
    for (uint32_t c = 0; c < s.get_mesh()->n_owned(); c++) n += s.h_conservatives(c, 0) != 1.0_r && s.h_conservatives(c, 0) != 2.0_r;
    return n;
}

} // namespace

TEST(InitialStateTest, VolumeFractionsOfFlatRegionsAreExact) {
    // Regions bounded by planes, rotated against the mesh, with booleans: the
    // cut cells are split exactly, so the mass is exact to rounding
    struct Case {
        std::string region;
        double volume;
    };
    std::vector<Case> cases;
    if constexpr (N_DIM == 2) {
        cases = {
            {"{type = \"rectangle\", center = [0.5, 0.5], size = [0.4, 0.3], pivot = [0.5, 0.5], rotate = 30}", 0.12},
            // An L-shaped (non-convex) polygon
            {"{type = \"polygon\", vertices = [[0.13, 0.17], [0.81, 0.21], [0.77, 0.43], [0.41, 0.39], [0.37, 0.88], [0.15, 0.83]]}",
             0.0},
            {"{type = \"difference\", a = {type = \"rectangle\", center = [0.5, 0.5], size = [0.6, 0.6], pivot = [0.5, 0.5], "
             "rotate = 17}, b = {type = \"half_space\", point = [0.5, 0.5], normal = [0.3, 0.7]}}",
             0.18},
        };
        // Shoelace area of the polygon
        const double v[6][2] = {{0.13, 0.17}, {0.81, 0.21}, {0.77, 0.43}, {0.41, 0.39}, {0.37, 0.88}, {0.15, 0.83}};
        double a = 0.0;
        for (int i = 0; i < 6; i++) a += v[i][0] * v[(i + 1) % 6][1] - v[(i + 1) % 6][0] * v[i][1];
        cases[1].volume = 0.5 * std::abs(a);
    } else {
        cases = {
            {"{type = \"box\", center = [0.5, 0.5, 0.5], size = [0.4, 0.3, 0.25], pivot = [0.5, 0.5, 0.5], "
             "rotate = {axis = [1, 1, 1], angle = 30}}",
             0.03},
            {"{type = \"difference\", a = {type = \"box\", center = [0.5, 0.5, 0.5], size = [0.5, 0.5, 0.5], "
             "pivot = [0.5, 0.5, 0.5], rotate = {axis = [1, 2, 0], angle = 25}}, "
             "b = {type = \"half_space\", point = [0.5, 0.5, 0.5], normal = [0.2, 0.3, 0.9]}}",
             0.0625},
        };
    }
    for (const auto & c : cases) {
        SCOPED_TRACE(c.region);
        const auto s = initialized(box(TRI, N_DIM == 2 ? 12 : 6, "[[initialize.regions]]\nrho = \"2\"\nregion = " + c.region + "\n"));
        EXPECT_NEAR(mass(*s), 1.0 + c.volume, roundoff(1e-13));
        EXPECT_GT(mixed_cells(*s), 0u);
    }
}

TEST(InitialStateTest, PolynomialStatesInFlatRegionsHaveExactCellAverages) {
    // A cubic density with a quadratic one layered inside a rotated box: each
    // cell average equals the exact integral over the cell and its exact
    // intersection with the box (overlap.h, an independent clipper)
    SKIP_IN_SINGLE_PRECISION("compares with double-precision integrals");
    const double h0 = 0.2, h1 = 0.15, h2 = 0.125, angle = 30.0 * PI / 180.0;
    std::vector<overlap::Point> corners;
    for (const double sz : N_DIM == 2 ? std::vector<double>{0.0} : std::vector<double>{-h2, h2}) {
        for (const auto & [a, b] : {std::pair{-h0, -h1}, {h0, -h1}, {h0, h1}, {-h0, h1}}) {
            overlap::Point p;
            p[0] = 0.5 + a * std::cos(angle) - b * std::sin(angle);
            p[1] = 0.5 + a * std::sin(angle) + b * std::cos(angle);
            if constexpr (N_DIM == 3) p[2] = 0.5 + sz;
            corners.push_back(p);
        }
    }
    const overlap::Region box_region = overlap::region(corners);
    const auto base = [](const overlap::Point & p) {
        return 2.0 + p[0] * p[1] * p[1] - 0.5 * p[0] * p[0] * p[0] + (N_DIM == 3 ? p[2] * p[2] * p[0] : 0.0);
    };
    const auto inside = [](const overlap::Point & p) {
        return 3.0 + p[0] * p[0] - p[0] * p[1] + (N_DIM == 3 ? p[1] * p[2] : 0.0);
    };
    const std::string base_state =
        N_DIM == 2 ? "type = \"analytical\"\nrho = \"2 + x * y^2 - 0.5 * x^3\"\np = \"1\"\nu = [\"0\", \"0\"]\n"
                   : "type = \"analytical\"\nrho = \"2 + x * y^2 - 0.5 * x^3 + z^2 * x\"\np = \"1\"\nu = [\"0\", \"0\", \"0\"]\n";
    const std::string layer =
        N_DIM == 2 ? "[[initialize.regions]]\nrho = \"3 + x^2 - x * y\"\n"
                     "region = {type = \"rectangle\", center = [0.5, 0.5], size = [0.4, 0.3], pivot = [0.5, 0.5], rotate = 30}\n"
                   : "[[initialize.regions]]\nrho = \"3 + x^2 - x * y + y * z\"\n"
                     "region = {type = \"box\", center = [0.5, 0.5, 0.5], size = [0.4, 0.3, 0.25], pivot = [0.5, 0.5, 0.5], "
                     "rotate = {axis = [0, 0, 1], angle = 30}}\n";
    const overlap::Rule & rule = overlap::simplex_rule(3);
    const auto integral = [&](const std::vector<overlap::Simplex> & simplices, const auto & f) {
        double sum = 0.0;
        for (const auto & t : simplices) {
            for (size_t q = 0; q < rule.weights.size(); q++) sum += overlap::volume(t) * rule.weights[q] * f(overlap::map(t, rule.points[q]));
        }
        return sum;
    };
    for (const char * mesh_type : {"cartesian", TRI}) {
        SCOPED_TRACE(mesh_type);
        const auto s = initialized(box(mesh_type, N_DIM == 2 ? 9 : 5, layer, base_state));
        const Mesh & mesh = *s->get_mesh();
        uint32_t cut = 0;
        for (uint32_t c = 0; c < mesh.n_owned(); c++) {
            const overlap::Region cell = overlap::region(overlap::cell_nodes(mesh, c));
            std::vector<overlap::Simplex> part;
            overlap::intersect(cell.simplices, box_region, part);
            double part_volume = 0.0;
            for (const auto & t : part) part_volume += overlap::volume(t);
            cut += part_volume > 1e-12 * cell.volume && part_volume < (1.0 - 1e-12) * cell.volume;
            const double expected = (integral(cell.simplices, base) - integral(part, base) + integral(part, inside)) / cell.volume;
            EXPECT_NEAR(double(s->h_conservatives(c, 0)), expected, 1e-13) << "cell " << c;
            EXPECT_NEAR(double(s->h_conservatives(c, N_DIM + 1)), 2.5, 1e-13) << "cell " << c;
        }
        EXPECT_GT(cut, 0u);
    }
}

TEST(InitialStateTest, VolumeFractionsOfCurvedRegionsConvergeAndUncutCellsArePure) {
    SKIP_IN_SINGLE_PRECISION("the 2D errors are within a few float roundings of the mass");
    const std::string sphere = N_DIM == 2 ? "{type = \"disk\", center = [0.47, 0.52], radius = 0.31}"
                                          : "{type = \"sphere\", center = [0.47, 0.52, 0.49], radius = 0.31}";
    const double volume = N_DIM == 2 ? PI * 0.31 * 0.31 : 4.0 / 3.0 * PI * 0.31 * 0.31 * 0.31;
    // Cut along the chords of finest-level simplices: second order in their size
    auto run = [&](const uint32_t n) {
        return initialized(box(TRI, n, "[[initialize.regions]]\nrho = \"2\"\nregion = " + sphere + "\n"));
    };
    const uint32_t n = N_DIM == 2 ? 8 : 4;
    const double coarse = std::abs(mass(*run(n)) - 1.0 - volume);
    const auto s = run(2 * n);
    const double fine = std::abs(mass(*s) - 1.0 - volume);
    EXPECT_LT(fine, (N_DIM == 2 ? 3e-6 : 1.5e-3) * volume);
    EXPECT_NEAR(std::log2(coarse / fine), 2.0, 0.3);
    const Mesh & mesh = *s->get_mesh();
    const double center[3] = {0.47, 0.52, 0.49};
    for (uint32_t c = 0; c < mesh.n_owned(); c++) {
        double r_min = 1e300, r_max = 0.0;
        for (uint32_t k = 0; k < mesh.h_n_nodes_of_cell(c); k++) {
            double d2 = 0.0;
            for (uint32_t d = 0; d < N_DIM; d++) {
                const double e = double(mesh.h_node_coords(mesh.h_node_of_cell(c, k), d)) - center[d];
                d2 += e * e;
            }
            r_min = std::min(r_min, std::sqrt(d2));
            r_max = std::max(r_max, std::sqrt(d2));
        }
        if (r_max < 0.31) {
            EXPECT_EQ(s->h_conservatives(c, 0), 2.0_r);
        }
        if (r_min > 0.31 + 0.1) {
            EXPECT_EQ(s->h_conservatives(c, 0), 1.0_r);
        }
    }
}

TEST(InitialStateTest, LayersStackByPriorityAndKeepThePressure) {
    // Two overlapping boxes: the one of higher priority wins in the overlap
    // whatever the order; a layer setting T alone keeps the pressure below
    const std::string a = N_DIM == 2 ? "{type = \"rectangle\", center = [0.35, 0.5], size = [0.5, 0.5]}"
                                     : "{type = \"box\", center = [0.35, 0.5, 0.5], size = [0.5, 0.5, 0.5]}";
    const std::string b = N_DIM == 2 ? "{type = \"rectangle\", center = [0.65, 0.5], size = [0.5, 0.5]}"
                                     : "{type = \"box\", center = [0.65, 0.5, 0.5], size = [0.5, 0.5, 0.5]}";
    const std::string layers = "[[initialize.regions]]\nT = \"4\"\npriority = 2\nregion = " + a + "\n"
                               "[[initialize.regions]]\nT = \"0.5\"\nregion = " + b + "\n";
    const auto s = initialized(box("cartesian", 10, layers));
    const Mesh & mesh = *s->get_mesh();
    const Euler & physics = s->get_physics();
    for (uint32_t c = 0; c < mesh.n_owned(); c++) {
        rtype W[N_CONSERVATIVE], U[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE U[i] = s->h_conservatives(c, i);
        physics.compute_W_from_conservatives(W, U);
        EXPECT_NEAR(W[N_DIM + 1], 1.0, roundoff(1e-12));
        const double x = double(mesh.h_cell_coords(c, 0)), y = double(mesh.h_cell_coords(c, 1));
        if (std::abs(y - 0.5) < 0.2 && std::abs(N_DIM == 3 ? double(mesh.h_cell_coords(c, 2)) - 0.5 : 0.0) < 0.2) {
            if (x > 0.45 && x < 0.55) {
                EXPECT_EQ(s->h_conservatives(c, 0), 0.25_r) << "overlap";
            }
            if (x > 0.65 && x < 0.85) {
                EXPECT_EQ(s->h_conservatives(c, 0), 2.0_r) << "b alone";
            }
        }
    }
}

TEST(InitialStateTest, CenterCellsTakeTheLayerWhole) {
    const std::string disk = N_DIM == 2 ? "{type = \"disk\", center = [0.5, 0.5], radius = 0.3}"
                                        : "{type = \"sphere\", center = [0.5, 0.5, 0.5], radius = 0.3}";
    const auto s = initialized(box(TRI, 10, "[[initialize.regions]]\nrho = \"2\"\ncells = \"center\"\nregion = " + disk + "\n"));
    const Mesh & mesh = *s->get_mesh();
    for (uint32_t c = 0; c < mesh.n_owned(); c++) {
        double d2 = 0.0;
        for (uint32_t d = 0; d < N_DIM; d++) d2 += std::pow(double(mesh.h_cell_coords(c, d)) - 0.5, 2);
        EXPECT_EQ(s->h_conservatives(c, 0), std::sqrt(d2) < 0.3 ? 2.0_r : 1.0_r);
    }
}

TEST(InitialStateTest, NamedRegionsLayerAsTheTablesTheyName) {
    const std::string disk = N_DIM == 2 ? "{type = \"disk\", center = [0.5, 0.5], radius = 0.3}"
                                        : "{type = \"sphere\", center = [0.5, 0.5, 0.5], radius = 0.3}";
    const auto inline_region = initialized(box(TRI, 8, "[[initialize.regions]]\nrho = \"2\"\nregion = " + disk + "\n"));
    const auto named = initialized(box(TRI, 8, "[[initialize.regions]]\nrho = \"2\"\nregion = \"blob\"\n") +
                                   "[regions]\nblob = " + disk + "\n");
    const Mesh & mesh = *named->get_mesh();
    size_t n_inside = 0;
    for (uint32_t c = 0; c < mesh.n_owned(); c++) {
        EXPECT_EQ(named->h_conservatives(c, 0), inline_region->h_conservatives(c, 0));
        n_inside += named->h_conservatives(c, 0) > 1.0_r;
    }
    EXPECT_GT(n_inside, 0u);
}

TEST(InitialStateTest, TableInterpolationHasItsDesignOrder) {
    // Linear: exact for linear data, error / 4 per halving; cubic: exact for
    // cubics, error / 16 per halving, on non-uniform grids too
    auto grid = [](const uint32_t n) {
        std::vector<double> x(n);
        for (uint32_t i = 0; i < n; i++) x[i] = double(i) / (n - 1) + 0.2 * std::sin(PI * double(i) / (n - 1)) / (n - 1);
        return x;
    };
    auto max_error = [&](const uint32_t n, const bool cubic, const std::function<double(double)> & f) {
        tabulated::Data data;
        const std::vector<double> x = grid(n);
        std::vector<double> v;
        for (const double xi : x) v.push_back(f(xi));
        data.add_table({x}, {v}, cubic, "t");
        const auto h = data.host();
        double e = 0.0;
        for (int i = 0; i <= 997; i++) {
            const double q = i / 997.0;
            e = std::max(e, std::abs(h(0, 0, &q) - f(q)));
        }
        return e;
    };
    const auto smooth = [](const double x) { return std::sin(3.0 * x) + x * x; };
    EXPECT_LT(max_error(9, false, [](const double x) { return 2.0 - 3.0 * x; }), 1e-14);
    EXPECT_LT(max_error(9, true, [](const double x) { return 1.0 + x - 2.0 * x * x + 0.7 * x * x * x; }), 1e-13);
    for (const bool cubic : {false, true}) {
        const double ratio = max_error(41, cubic, smooth) / max_error(81, cubic, smooth);
        EXPECT_NEAR(std::log2(ratio), cubic ? 4.0 : 2.0, 0.3) << (cubic ? "cubic" : "linear");
    }
    // Tensor products: exact for products of the 1D degrees, rows in any order
    tabulated::Data data;
    std::vector<std::vector<double>> c(2);
    std::vector<double> v;
    std::mt19937 rng(5);
    std::vector<std::pair<double, double>> nodes;
    for (const double x : grid(6)) {
        for (const double y : {0.0, 0.3, 0.45, 1.0, 1.7}) nodes.emplace_back(x, y);
    }
    std::shuffle(nodes.begin(), nodes.end(), rng);
    const auto f = [](const double x, const double y) { return (1.0 + x * x * x) * (2.0 - y + y * y); };
    for (const auto & [x, y] : nodes) {
        c[0].push_back(x);
        c[1].push_back(y);
        v.push_back(f(x, y));
    }
    data.add_table(c, {v}, true, "t");
    const auto h = data.host();
    for (int i = 0; i < 50; i++) {
        const double q[2] = {std::uniform_real_distribution<double>(0, 1)(rng), std::uniform_real_distribution<double>(0, 1.7)(rng)};
        EXPECT_NEAR(h(0, 0, q), f(q[0], q[1]), 1e-12);
    }
}

TEST(InitialStateTest, PointCloudsFindTheirNearestPoints) {
    // Against a brute-force search, ties by the lower index, for the nearest
    // point and inverse-distance weights of several
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<std::vector<double>> c(N_DIM);
    std::vector<double> v;
    for (int i = 0; i < 500; i++) {
        // Points on a coarse lattice repeat, so that ties occur
        for (uint32_t d = 0; d < N_DIM; d++) c[d].push_back(std::round(u(rng) * 12.0) / 12.0);
        v.push_back(u(rng));
    }
    tabulated::Data data;
    data.add_points(c, {v}, 1, "p");
    data.add_points(c, {v}, 5, "p");
    const auto h = data.host();
    for (int i = 0; i < 300; i++) {
        double q[3] = {0.0, 0.0, 0.0};
        for (uint32_t d = 0; d < N_DIM; d++) q[d] = i < 20 ? c[d][size_t(i)] : u(rng) * 1.2 - 0.1;
        std::vector<std::pair<double, size_t>> order;
        for (size_t p = 0; p < v.size(); p++) {
            double d2 = 0.0;
            for (uint32_t d = 0; d < N_DIM; d++) d2 += (q[d] - c[d][p]) * (q[d] - c[d][p]);
            order.emplace_back(d2, p);
        }
        std::sort(order.begin(), order.end());
        EXPECT_EQ(h(0, 0, q), v[order[0].second]);
        double expected = v[order[0].second];
        if (order[0].first > 0.0) {
            double sum = 0.0, weight = 0.0;
            for (size_t k = 0; k < 5; k++) {
                sum += v[order[k].second] / order[k].first;
                weight += 1.0 / order[k].first;
            }
            expected = sum / weight;
        }
        EXPECT_NEAR(h(1, 0, q), expected, 1e-14);
    }
}

TEST(InitialStateTest, ProfilesAndPointsFeedExpressions) {
    // A linear profile of x (in mm) read along a rotated axis: the cell
    // averages of a linear velocity are its values at the centroids; a point
    // cloud of one point per cell centroid gives each cell its own value
    const std::string profile = write_temp("profile.csv", "# s [mm], u [m/s]\ns, u\n-1000, -3\n0, 1\n2000, 9\n");
    std::ostringstream cloud;
    cloud << "x,y" << (N_DIM == 3 ? ",z" : "") << ",v\n";
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            for (int k = 0; k < (N_DIM == 3 ? 5 : 1); k++) {
                cloud << 0.1 + 0.2 * i << "," << 0.1 + 0.2 * j << (N_DIM == 3 ? "," + std::to_string(0.1 + 0.2 * k) : "")
                      << "," << i + 10 * j + 100 * k << "\n";
            }
        }
    }
    const std::string points = write_temp("points.csv", cloud.str());
    const std::string init =
        "[[initialize.data]]\nname = \"line\"\nfile = \"" + profile + "\"\nunits = {s = \"mm\"}\nat = [\"0.6 * x + 0.8 * y\"]\n"
        "[[initialize.data]]\nname = \"cloud\"\ntype = \"points\"\nfile = \"" + points + "\"\n"
        "[[initialize.regions]]\nregion = {type = \"half_space\", normal = [-1, 0, 0]}\n"
        "u = [\"line_u\", \"cloud_v\"" + std::string(N_DIM == 3 ? ", \"0\"" : "") + "]\n";
    const auto s = initialized(box("cartesian", 5, init));
    const Mesh & mesh = *s->get_mesh();
    for (uint32_t c = 0; c < mesh.n_owned(); c++) {
        const double x = double(mesh.h_cell_coords(c, 0)), y = double(mesh.h_cell_coords(c, 1));
        const double z = N_DIM == 3 ? double(mesh.h_cell_coords(c, 2)) : 0.0;
        const double rho = double(s->h_conservatives(c, 0));
        EXPECT_NEAR(double(s->h_conservatives(c, 1)) / rho, 1.0 + 4.0 * (0.6 * x + 0.8 * y), roundoff(1e-12));
        const double v = std::round((x - 0.1) / 0.2) + 10 * std::round((y - 0.1) / 0.2) +
                         100 * std::round((z - 0.1) / 0.2) * (N_DIM == 3);
        EXPECT_NEAR(double(s->h_conservatives(c, 2)) / rho, v, roundoff(1e-14) * (1.0 + v));
    }
}

TEST(InitialStateTest, RegionsOverARestartKeepTheCellsTheyMiss) {
    const std::string dir = (scratch_dir() / "restart").string();
    const std::string write = "[[write_data]]\nprefix = \"" + dir + "/r\"\nformat = \"restart\"\ninterval = 1\n";
    const std::string blob = std::string("type = \"analytical\"\nrho = \"1 + 0.5 * sin(3 * x) * cos(2 * y)\"\np = \"1\"\nu = ") +
                             (N_DIM == 2 ? "[\"0.1\", \"0\"]\n" : "[\"0.1\", \"0\", \"0\"]\n");
    const uint32_t n = 16;
    const double margin = 0.5 * std::sqrt(double(N_DIM)) / n;
    const auto first = initialized(box("cartesian", n, "", blob) + write);
    const auto initial = Kokkos::create_mirror(first->h_conservatives);
    Kokkos::deep_copy(initial, first->h_conservatives);
    first->run();
    const std::string disk = N_DIM == 2 ? "{type = \"disk\", center = [0.5, 0.5], radius = 0.2}"
                                        : "{type = \"sphere\", center = [0.5, 0.5, 0.5], radius = 0.2}";
    const auto second = initialized(box("cartesian", n, "[[initialize.regions]]\nT = \"3\"\nregion = " + disk + "\n",
                                        "type = \"restart\"\nfile = \"" + dir + "/r_000000.restart\"\n"));
    const Mesh & mesh = *second->get_mesh();
    uint32_t inside = 0;
    for (uint32_t c = 0; c < mesh.n_owned(); c++) {
        double d2 = 0.0;
        for (uint32_t d = 0; d < N_DIM; d++) d2 += std::pow(double(mesh.h_cell_coords(c, d)) - 0.5, 2);
        if (std::sqrt(d2) > 0.2 + margin) {
            FOR_I_CONSERVATIVE EXPECT_EQ(second->h_conservatives(c, i), initial(c, i));
        } else if (std::sqrt(d2) < 0.2 - margin) {
            // T = 3 at the restart's pressure (1), velocity kept: rho = p / (R T)
            const double rho = double(second->h_conservatives(c, 0));
            EXPECT_NEAR(rho, 1.0 / 3.0, roundoff(1e-13));
            EXPECT_NEAR(double(second->h_conservatives(c, 1)) / rho, 0.1, roundoff(1e-13));
            inside++;
        }
    }
    EXPECT_GT(inside, 0u);
}

namespace {

/** @brief The periodic box [0, 2] x [0, 1] (x [0, 1]) of 2n x n (x n) cells, uniform, with init added to [initialize]. */
std::string periodic_box(const uint32_t n, const std::string & init) {
    std::ostringstream s;
    s << "[run]\nn_steps = 1\ncfl = 0.25\n"
      << "[mesh]\ntype = \"cartesian\"\nNx = " << 2 * n << "\nNy = " << n << "\nLx = 2.0\nLy = 1.0\n"
      << (N_DIM == 3 ? "Nz = " + std::to_string(n) + "\nLz = 1.0\n" : "")
      << "[[periodic]]\nzones = [\"left\", \"right\"]\n[[periodic]]\nzones = [\"bottom\", \"top\"]\n"
      << (N_DIM == 3 ? "[[periodic]]\nzones = [\"back\", \"front\"]\n" : "")
      << "[initialize]\n" << UNIFORM << init
      << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 1000000\n";
    return s.str();
}

struct MeasuredSpectrum {
    std::vector<double> shells;  // energy of the shells [j - 1/2, j + 1/2) 2 pi, j = 1, 2, ...
    double divergence = 0.0;     // sum of |k . u_hat|^2 over sum of |k|^2 |u_hat|^2
};

/**
 * @brief The spectrum of the point velocity behind a periodic_box's cell
 *        averages: a direct Fourier transform of the cells, divided by the
 *        transfer function of exact box averages, sinc(k_i h_i / 2).
 */
MeasuredSpectrum measure_spectrum(const Solver & s, const uint32_t n) {
    const Mesh & mesh = *s.get_mesh();
    const int N[3] = {2 * int(n), int(n), N_DIM == 3 ? int(n) : 1};
    const double L[3] = {2.0, 1.0, 1.0};
    const size_t n_cells = size_t(N[0]) * size_t(N[1]) * size_t(N[2]);
    std::vector<std::array<double, 3>> u(n_cells, {0.0, 0.0, 0.0});
    std::vector<std::array<int, 3>> index(n_cells);
    for (uint32_t c = 0; c < mesh.n_owned(); c++) {
        std::array<int, 3> i = {0, 0, 0};
        for (uint32_t d = 0; d < N_DIM; d++) i[d] = int(std::floor(double(mesh.h_cell_coords(c, d)) / L[d] * N[d]));
        const size_t g = (size_t(i[0]) * size_t(N[1]) + size_t(i[1])) * size_t(N[2]) + size_t(i[2]);
        index[g] = i;
        for (uint32_t d = 0; d < N_DIM; d++) u[g][d] = double(s.h_conservatives(c, 1 + d)) / double(s.h_conservatives(c, 0));
    }
    // exp(-i k_d x_d) per direction, wavenumber index (mod N) and cell index
    std::vector<std::vector<std::vector<std::complex<double>>>> phase(3);
    for (size_t d = 0; d < 3; d++) {
        phase[d].resize(size_t(N[d]));
        for (int m = 0; m < N[d]; m++) {
            const double k = 2.0 * PI * (m <= N[d] / 2 ? m : m - N[d]) / L[d];
            for (int i = 0; i < N[d]; i++) phase[d][size_t(m)].push_back(std::polar(1.0, -k * (i + 0.5) * L[d] / N[d]));
        }
    }
    MeasuredSpectrum out;
    double div = 0.0, grad = 0.0;
    for (int m0 = 0; m0 < N[0] / 2; m0++) {
        for (int m1 = -(N[1] / 2 - 1); m1 < N[1] / 2; m1++) {
            for (int m2 = N_DIM == 3 ? -(N[2] / 2 - 1) : 0; m2 < (N_DIM == 3 ? N[2] / 2 : 1); m2++) {
                if (m0 == 0 && (m1 < 0 || (m1 == 0 && m2 <= 0))) continue;
                const int m[3] = {m0, m1, m2};
                double k[3], transfer = 1.0, k2 = 0.0;
                for (size_t d = 0; d < 3; d++) {
                    k[d] = 2.0 * PI * m[d] / L[d];
                    const double z = 0.5 * k[d] * L[d] / N[d];
                    transfer *= z == 0.0 ? 1.0 : std::sin(z) / z;
                    k2 += k[d] * k[d];
                }
                std::complex<double> u_hat[3] = {0.0, 0.0, 0.0};
                for (size_t g = 0; g < n_cells; g++) {
                    std::complex<double> e = 1.0;
                    for (size_t d = 0; d < 3; d++) e *= phase[d][size_t((m[d] + N[d]) % N[d])][size_t(index[g][d])];
                    for (size_t d = 0; d < 3; d++) u_hat[d] += u[g][d] * e;
                }
                double energy = 0.0;
                std::complex<double> k_dot_u = 0.0;
                for (size_t d = 0; d < 3; d++) {
                    u_hat[d] /= double(n_cells) * transfer;
                    energy += std::norm(u_hat[d]);
                    k_dot_u += k[d] * u_hat[d];
                }
                const size_t j = size_t(std::lround(std::sqrt(k2) / (2.0 * PI)));
                if (out.shells.size() < j) out.shells.resize(j, 0.0);
                out.shells[j - 1] += energy;
                div += std::norm(k_dot_u);
                grad += k2 * energy;
            }
        }
    }
    out.divergence = div / grad;
    return out;
}

/** @brief The integral of f over [a, b], composite Simpson. */
double simpson(const std::function<double(double)> & f, const double a, const double b) {
    const int n = 2000;
    const double h = (b - a) / n;
    double s = f(a) + f(b);
    for (int i = 1; i < n; i++) s += (i % 2 ? 4.0 : 2.0) * f(a + i * h);
    return s * h / 3.0;
}

} // namespace

TEST(InitialStateTest, TurbulenceHasItsSpectrumAndEnergyAndNoDivergence) {
    // An anisotropic periodic box (periods 2 and 1), so that wavevectors are
    // not their integer indices. The shells of width 2 pi up to the default
    // k_max (pi over the mean cell size) hold the spectrum's share of the
    // energy, scaled to (N_DIM / 2) u'^2. The model spectra are written here
    // from their definitions, with k_e from the integral length scale
    // L = C int E / k dk / int E dk (C = 3 pi / 4 in 3D, 2 in 2D) in closed form.
    const uint32_t n = N_DIM == 2 ? 32 : 12;
    const int n_shells = N_DIM == 2 ? 15 : 5;
    const double u_rms = 0.3, L = 0.2, C = N_DIM == 3 ? 0.75 * PI : 2.0;
    auto beta = [](const double a, const double b) { return std::tgamma(a) * std::tgamma(b) / std::tgamma(a + b); };
    // int x^3 e^(-2 x^2) = 1/8 and int x^4 e^(-2 x^2) = 3 sqrt(pi) / 2^(11/2);
    // int x^a / (1 + x^2)^(17/6) = B((a + 1) / 2, 17/6 - (a + 1) / 2) / 2
    const double k_pp = C * 0.125 / (3.0 * std::sqrt(PI) / std::pow(2.0, 5.5)) / L;
    const double k_vkp = C * beta(2.0, 5.0 / 6.0) / beta(2.5, 1.0 / 3.0) / L;
    auto passot_pouquet = [&](const double k) { return std::pow(k / k_pp, 4) * std::exp(-2.0 * std::pow(k / k_pp, 2)); };
    auto von_karman_pao = [&](const double k) { return std::pow(k / k_vkp, 4) / std::pow(1.0 + std::pow(k / k_vkp, 2), 17.0 / 6.0); };
    // A tabulated spectrum, without an intensity: its own values
    const std::string table = (scratch_dir() / "spectrum.csv").string();
    {
        std::ofstream f(table);
        f << "k,E\n";
        for (int i = 0; i <= 2000; i++) f << 0.1 * i << "," << 0.01 * passot_pouquet(0.1 * i) << "\n";
    }
    struct Case {
        std::string turbulence;
        std::function<double(double)> E;
        bool scaled;
        double tolerance;  // per shell: quadrature (measured 1e-5 in 3D), and the table's linear interpolation
    };
    const std::vector<Case> cases = {
        {"spectrum = \"passot_pouquet\", intensity = 0.3, length_scale = 0.2", passot_pouquet, true, 1e-4},
        {"spectrum = \"von_karman_pao\", intensity = 0.3, length_scale = 0.2, seed = 4", von_karman_pao, true, 1e-4},
        {"spectrum = \"table\", file = \"" + table + "\"", [&](const double k) { return 0.01 * passot_pouquet(k); }, false, 2e-3},
    };
    for (const auto & c : cases) {
        SCOPED_TRACE(c.turbulence);
        const auto s = initialized(periodic_box(n, "turbulence = {" + c.turbulence + "}\n"));
        const MeasuredSpectrum measured = measure_spectrum(*s, n);
        std::vector<double> target(static_cast<size_t>(n_shells));
        double sum = 0.0;
        for (int j = 1; j <= n_shells; j++) {
            target[size_t(j - 1)] = simpson(c.E, 2.0 * PI * (j - 0.5), 2.0 * PI * (j + 0.5));
            sum += target[size_t(j - 1)];
        }
        const double scale = c.scaled ? 0.5 * N_DIM * u_rms * u_rms / sum : 1.0;
        double energy = 0.0;
        for (size_t j = 0; j < measured.shells.size(); j++) {
            const double expected = j < target.size() ? scale * target[j] : 0.0;
            EXPECT_NEAR(measured.shells[j], expected, c.tolerance * expected + 1e-6 * scale * sum) << "shell " << j + 1;
            energy += measured.shells[j];
        }
        EXPECT_NEAR(energy, scale * sum, 1e-5 * scale * sum);
        // Every mode is normal to its wavevector, and cell averaging keeps that (measured 1e-28)
        EXPECT_LT(measured.divergence, 1e-12);
    }
}

TEST(InitialStateTest, TurbulenceInARegionIsTheBaseTurbulenceThere) {
    // The modes depend on the seed and the period alone, so a region's
    // turbulence is the base turbulence inside it, bit for bit, and cells
    // beyond its edge stay at rest
    const uint32_t n = N_DIM == 2 ? 16 : 6;
    const std::string turbulence = "turbulence = {spectrum = \"passot_pouquet\", intensity = 0.2, length_scale = 0.2, seed = 9}\n";
    const auto everywhere = initialized(periodic_box(n, turbulence));
    const auto region = initialized(periodic_box(
        n, std::string("[[initialize.regions]]\nregion = {type = \"half_space\", point = ") +
               (N_DIM == 2 ? "[0.75, 0], normal = [1, 0]}\n" : "[0.75, 0, 0], normal = [1, 0, 0]}\n") + turbulence));
    const Mesh & mesh = *region->get_mesh();
    const double h = 1.0 / n;
    uint32_t inside = 0, outside = 0;
    for (uint32_t c = 0; c < mesh.n_owned(); c++) {
        const double x = double(mesh.h_cell_coords(c, 0));
        if (x < 0.75 - h) {
            FOR_I_CONSERVATIVE EXPECT_EQ(region->h_conservatives(c, i), everywhere->h_conservatives(c, i));
            inside++;
        } else if (x > 0.75 + h) {
            FOR_I_DIM EXPECT_EQ(region->h_conservatives(c, 1 + i), 0.0_r);
            outside++;
        }
    }
    EXPECT_GT(inside, 0u);
    EXPECT_GT(outside, 0u);
}
