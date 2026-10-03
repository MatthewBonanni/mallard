/**
 * @file transport.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Species transport fits: a port of Cantera's GasTransport fitting
 *        (Cantera 3.2.0, src/transport/GasTransport.cpp and MMCollisionInt.cpp,
 *        BSD-3-Clause, Copyright (c) 2001-2025 Cantera Developers), so that
 *        Mallard's mixture-averaged properties are Cantera's.
 * @version 0.3
 * @date 2026-10-03
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "transport.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

#include "thermo.h"

namespace chemistry {

namespace {

/**
 * @brief Least-squares polynomial of degree deg through (x, y) with weights
 *        w (squared, as Cantera's polyfit; empty for unit weights): Householder QR.
 */
std::vector<double> polyfit(const std::vector<double> & x, const std::vector<double> & y,
                            const std::vector<double> & w, const size_t deg) {
    const size_t n = x.size(), m = deg + 1;
    std::vector<double> A(n * m), b(y);
    for (size_t i = 0; i < n; i++) {
        const double s = w.empty() ? 1.0 : std::sqrt(w[i]);
        double xp = 1.0;
        for (size_t j = 0; j < m; j++, xp *= x[i]) A[i * m + j] = s * xp;
        b[i] *= s;
    }
    for (size_t j = 0; j < m; j++) {
        double norm = 0.0;
        for (size_t i = j; i < n; i++) norm += A[i * m + j] * A[i * m + j];
        norm = std::sqrt(norm);
        const double alpha = A[j * m + j] > 0.0 ? -norm : norm;
        std::vector<double> v(n, 0.0);
        for (size_t i = j; i < n; i++) v[i] = A[i * m + j];
        v[j] -= alpha;
        double vv = 0.0;
        for (size_t i = j; i < n; i++) vv += v[i] * v[i];
        if (vv == 0.0) continue;
        for (size_t c = j; c < m; c++) {
            double d = 0.0;
            for (size_t i = j; i < n; i++) d += v[i] * A[i * m + c];
            d *= 2.0 / vv;
            for (size_t i = j; i < n; i++) A[i * m + c] -= d * v[i];
        }
        double d = 0.0;
        for (size_t i = j; i < n; i++) d += v[i] * b[i];
        d *= 2.0 / vv;
        for (size_t i = j; i < n; i++) b[i] -= d * v[i];
    }
    std::vector<double> p(m);
    for (size_t j = m; j-- > 0;) {
        double s = b[j];
        for (size_t c = j + 1; c < m; c++) s -= A[j * m + c] * p[c];
        p[j] = s / A[j * m + j];
    }
    return p;
}

double poly(const std::vector<double> & c, const double x) {
    double s = 0.0;
    for (size_t i = c.size(); i-- > 0;) s = s * x + c[i];
    return s;
}

/**
 * @brief Reduced collision integrals of the Stockmayer potential (Monchick &
 *        Mason 1961, as tabulated in Cantera): Omega(2,2)* and A* = Omega(2,2)* /
 *        Omega(1,1)* against reduced temperature T* and reduced dipole moment delta*.
 */
class CollisionIntegrals {
    public:
        CollisionIntegrals() {
            log_T.resize(37);
            const std::vector<double> delta(DELTA, DELTA + 8);
            for (int i = 0; i < 37; i++) {
                log_T[i] = std::log(TSTAR[i + 1]);
                o22_poly.push_back(polyfit(delta, std::vector<double>(OMEGA22 + 8 * i, OMEGA22 + 8 * i + 8), {}, 6));
                a_poly.push_back(polyfit(delta, std::vector<double>(ASTAR + 8 * (i + 1), ASTAR + 8 * (i + 2)), {}, 6));
            }
        }

        double omega22(const double ts, const double delta) const { return interpolate(ts, delta, OMEGA22, 0, o22_poly); }

        double omega11(const double ts, const double delta) const {
            return omega22(ts, delta) / interpolate(ts, delta, ASTAR, 1, a_poly);
        }

    private:
        /** @brief Quadratic interpolation in ln T* of the table, or of its fits in delta*. */
        double interpolate(const double ts, const double delta, const double * table, const int row_shift,
                           const std::vector<std::vector<double>> & fits) const {
            int i = 0;
            while (i < 37 && !(ts < TSTAR[i + 1])) i++;
            int i1 = std::max(i - 1, 0);
            int i2 = i1 + 3;
            if (i2 > 36) {
                i2 = 36;
                i1 = i2 - 3;
            }
            double values[3];
            for (int j = i1; j < i2; j++) {
                values[j - i1] = delta == 0.0 ? table[8 * (j + row_shift)] : poly(fits[j], delta);
            }
            const double * x = &log_T[i1];
            const double x0 = std::log(ts);
            const double dx21 = x[1] - x[0], dx32 = x[2] - x[1], dx31 = dx21 + dx32;
            const double dy32 = values[2] - values[1], dy21 = values[1] - values[0];
            const double a = (dx21 * dy32 - dy21 * dx32) / (dx21 * dx31 * dx32);
            return a * (x0 - x[0]) * (x0 - x[1]) + (dy21 / dx21) * (x0 - x[1]) + values[1];
        }

        std::vector<double> log_T;
        std::vector<std::vector<double>> o22_poly, a_poly;

        static constexpr double DELTA[8] = {0.0, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 2.5};
        // T* of the A* table; Omega(2,2)* starts at TSTAR[1]
        static constexpr double TSTAR[39] = {
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0,
            5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 75.0, 100.0,
            500.0};
        static constexpr double OMEGA22[37 * 8] = {
            4.1005, 4.266, 4.833, 5.742, 6.729, 8.624, 10.34, 11.89,
            3.2626, 3.305, 3.516, 3.914, 4.433, 5.57, 6.637, 7.618,
            2.8399, 2.836, 2.936, 3.168, 3.511, 4.329, 5.126, 5.874,
            2.531, 2.522, 2.586, 2.749, 3.004, 3.64, 4.282, 4.895,
            2.2837, 2.277, 2.329, 2.46, 2.665, 3.187, 3.727, 4.249,
            2.0838, 2.081, 2.13, 2.243, 2.417, 2.862, 3.329, 3.786,
            1.922, 1.924, 1.97, 2.072, 2.225, 2.614, 3.028, 3.435,
            1.7902, 1.795, 1.84, 1.934, 2.07, 2.417, 2.788, 3.156,
            1.6823, 1.689, 1.733, 1.82, 1.944, 2.258, 2.596, 2.933,
            1.5929, 1.601, 1.644, 1.725, 1.838, 2.124, 2.435, 2.746,
            1.4551, 1.465, 1.504, 1.574, 1.67, 1.913, 2.181, 2.451,
            1.3551, 1.365, 1.4, 1.461, 1.544, 1.754, 1.989, 2.228,
            1.28, 1.289, 1.321, 1.374, 1.447, 1.63, 1.838, 2.053,
            1.2219, 1.231, 1.259, 1.306, 1.37, 1.532, 1.718, 1.912,
            1.1757, 1.184, 1.209, 1.251, 1.307, 1.451, 1.618, 1.795,
            1.0933, 1.1, 1.119, 1.15, 1.193, 1.304, 1.435, 1.578,
            1.0388, 1.044, 1.059, 1.083, 1.117, 1.204, 1.31, 1.428,
            0.99963, 1.004, 1.016, 1.035, 1.062, 1.133, 1.22, 1.319,
            0.96988, 0.9732, 0.983, 0.9991, 1.021, 1.079, 1.153, 1.236,
            0.92676, 0.9291, 0.936, 0.9473, 0.9628, 1.005, 1.058, 1.121,
            0.89616, 0.8979, 0.903, 0.9114, 0.923, 0.9545, 0.9955, 1.044,
            0.87272, 0.8741, 0.878, 0.8845, 0.8935, 0.9181, 0.9505, 0.9893,
            0.85379, 0.8549, 0.858, 0.8632, 0.8703, 0.8901, 0.9164, 0.9482,
            0.83795, 0.8388, 0.8414, 0.8456, 0.8515, 0.8678, 0.8895, 0.916,
            0.82435, 0.8251, 0.8273, 0.8308, 0.8356, 0.8493, 0.8676, 0.8901,
            0.80184, 0.8024, 0.8039, 0.8065, 0.8101, 0.8201, 0.8337, 0.8504,
            0.78363, 0.784, 0.7852, 0.7872, 0.7899, 0.7976, 0.8081, 0.8212,
            0.76834, 0.7687, 0.7696, 0.7712, 0.7733, 0.7794, 0.7878, 0.7983,
            0.75518, 0.7554, 0.7562, 0.7575, 0.7592, 0.7642, 0.7711, 0.7797,
            0.74364, 0.7438, 0.7445, 0.7455, 0.747, 0.7512, 0.7569, 0.7642,
            0.71982, 0.72, 0.7204, 0.7211, 0.7221, 0.725, 0.7289, 0.7339,
            0.70097, 0.7011, 0.7014, 0.7019, 0.7026, 0.7047, 0.7076, 0.7112,
            0.68545, 0.6855, 0.6858, 0.6861, 0.6867, 0.6883, 0.6905, 0.6932,
            0.67232, 0.6724, 0.6726, 0.6728, 0.6733, 0.6743, 0.6762, 0.6784,
            0.65099, 0.651, 0.6512, 0.6513, 0.6516, 0.6524, 0.6534, 0.6546,
            0.61397, 0.6141, 0.6143, 0.6145, 0.6147, 0.6148, 0.6148, 0.6147,
            0.5887, 0.5889, 0.5894, 0.59, 0.5903, 0.5901, 0.5895, 0.5885};
        static constexpr double ASTAR[39 * 8] = {
            1.0065, 1.0840, 1.0840, 1.0840, 1.0840, 1.0840, 1.0840, 1.0840,
            1.0231, 1.0660, 1.0380, 1.0400, 1.0430, 1.0500, 1.0520, 1.0510,
            1.0424, 1.0450, 1.0480, 1.0520, 1.0560, 1.0650, 1.0660, 1.0640,
            1.0719, 1.0670, 1.0600, 1.0550, 1.0580, 1.0680, 1.0710, 1.0710,
            1.0936, 1.0870, 1.0770, 1.0690, 1.0680, 1.0750, 1.0780, 1.0780,
            1.1053, 1.0980, 1.0880, 1.0800, 1.0780, 1.0820, 1.0840, 1.0840,
            1.1104, 1.1040, 1.0960, 1.0890, 1.0860, 1.0890, 1.0900, 1.0900,
            1.1114, 1.1070, 1.1000, 1.0950, 1.0930, 1.0950, 1.0960, 1.0950,
            1.1104, 1.1070, 1.1020, 1.0990, 1.0980, 1.1000, 1.1000, 1.0990,
            1.1086, 1.1060, 1.1020, 1.1010, 1.1010, 1.1050, 1.1050, 1.1040,
            1.1063, 1.1040, 1.1030, 1.1030, 1.1040, 1.1080, 1.1090, 1.1080,
            1.1020, 1.1020, 1.1030, 1.1050, 1.1070, 1.1120, 1.1150, 1.1150,
            1.0985, 1.0990, 1.1010, 1.1040, 1.1080, 1.1150, 1.1190, 1.1200,
            1.0960, 1.0960, 1.0990, 1.1030, 1.1080, 1.1160, 1.1210, 1.1240,
            1.0943, 1.0950, 1.0990, 1.1020, 1.1080, 1.1170, 1.1230, 1.1260,
            1.0934, 1.0940, 1.0970, 1.1020, 1.1070, 1.1160, 1.1230, 1.1280,
            1.0926, 1.0940, 1.0970, 1.0990, 1.1050, 1.1150, 1.1230, 1.1300,
            1.0934, 1.0950, 1.0970, 1.0990, 1.1040, 1.1130, 1.1220, 1.1290,
            1.0948, 1.0960, 1.0980, 1.1000, 1.1030, 1.1120, 1.1190, 1.1270,
            1.0965, 1.0970, 1.0990, 1.1010, 1.1040, 1.1100, 1.1180, 1.1260,
            1.0997, 1.1000, 1.1010, 1.1020, 1.1050, 1.1100, 1.1160, 1.1230,
            1.1025, 1.1030, 1.1040, 1.1050, 1.1060, 1.1100, 1.1150, 1.1210,
            1.1050, 1.1050, 1.1060, 1.1070, 1.1080, 1.1110, 1.1150, 1.1200,
            1.1072, 1.1070, 1.1080, 1.1080, 1.1090, 1.1120, 1.1150, 1.1190,
            1.1091, 1.1090, 1.1090, 1.1100, 1.1110, 1.1130, 1.1150, 1.1190,
            1.1107, 1.1110, 1.1110, 1.1110, 1.1120, 1.1140, 1.1160, 1.1190,
            1.1133, 1.1140, 1.1130, 1.1140, 1.1140, 1.1150, 1.1170, 1.1190,
            1.1154, 1.1150, 1.1160, 1.1160, 1.1160, 1.1170, 1.1180, 1.1200,
            1.1172, 1.1170, 1.1170, 1.1180, 1.1180, 1.1180, 1.1190, 1.1200,
            1.1186, 1.1190, 1.1190, 1.1190, 1.1190, 1.1190, 1.1200, 1.1210,
            1.1199, 1.1200, 1.1200, 1.1200, 1.1200, 1.1210, 1.1210, 1.1220,
            1.1223, 1.1220, 1.1220, 1.1220, 1.1220, 1.1230, 1.1230, 1.1240,
            1.1243, 1.1240, 1.1240, 1.1240, 1.1240, 1.1240, 1.1250, 1.1250,
            1.1259, 1.1260, 1.1260, 1.1260, 1.1260, 1.1260, 1.1260, 1.1260,
            1.1273, 1.1270, 1.1270, 1.1270, 1.1270, 1.1270, 1.1270, 1.1280,
            1.1297, 1.1300, 1.1300, 1.1300, 1.1300, 1.1300, 1.1300, 1.1290,
            1.1339, 1.1340, 1.1340, 1.1350, 1.1350, 1.1340, 1.1340, 1.1320,
            1.1364, 1.1370, 1.1370, 1.1380, 1.1390, 1.1380, 1.1370, 1.1350,
            1.14187, 1.14187, 1.14187, 1.14187, 1.14187, 1.14187, 1.14187, 1.14187};
};

} // namespace

TransportFits fit_transport(const Mechanism & mechanism) {
    constexpr double PI = std::numbers::pi;
    const size_t n = mechanism.n_species();
    std::vector<double> mw(n), sigma(n), eps(n), dipole(n), alpha(n), zrot(n), crot(n);
    std::vector<bool> polar(n);
    double T_min = 0.0, T_max = 1e300;
    for (size_t k = 0; k < n; k++) {
        const Species & sp = mechanism.species[k];
        if (!sp.has_transport) {
            throw std::runtime_error("Mechanism " + mechanism.file + ": species " + sp.name +
                                     " has no gas transport data.");
        }
        mw[k] = sp.molecular_weight;
        sigma[k] = sp.transport.diameter;
        eps[k] = sp.transport.well_depth;
        dipole[k] = sp.transport.dipole;
        polar[k] = sp.transport.dipole > 0.0;
        alpha[k] = sp.transport.polarizability;
        zrot[k] = sp.transport.rotational_relaxation;
        crot[k] = sp.transport.geometry == MoleculeGeometry::ATOM ? 0.0
                  : sp.transport.geometry == MoleculeGeometry::LINEAR ? 1.0 : 1.5;
        T_min = std::max(T_min, sp.thermo.T_bounds.front());
        T_max = std::min(T_max, sp.thermo.T_bounds.back());
    }
    if (!(T_min > 0.0 && T_max > T_min && T_max < 1e300)) {
        throw std::runtime_error("Mechanism " + mechanism.file + ": transport fits need a common, finite thermo range.");
    }

    // Pair parameters, with the polar/nonpolar corrections of the well depth and diameter
    auto pair = [&](size_t i, size_t j) { return i * n + j; };
    std::vector<double> reduced_mass(n * n), diam(n * n), epsilon(n * n), delta(n * n);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i; j < n; j++) {
            const double m = mw[i] * mw[j] / (AVOGADRO * (mw[i] + mw[j]));
            double d = 0.5 * (sigma[i] + sigma[j]);
            double e = std::sqrt(eps[i] * eps[j]);
            const double mu = std::sqrt(dipole[i] * dipole[j]);
            const double dl = 0.5 * mu * mu / (4.0 * PI * EPSILON_0 * e * d * d * d);
            if (polar[i] != polar[j]) {
                const size_t kp = polar[i] ? i : j, knp = polar[i] ? j : i;
                const double alpha_star = alpha[knp] / std::pow(sigma[knp], 3);
                const double mu_p_star = dipole[kp] / std::sqrt(4.0 * PI * EPSILON_0 * std::pow(sigma[kp], 3) * eps[kp]);
                const double xi = 1.0 + 0.25 * alpha_star * mu_p_star * mu_p_star * std::sqrt(eps[kp] / eps[knp]);
                d *= std::pow(xi, -1.0 / 6.0);
                e *= xi * xi;
            }
            for (size_t p : {pair(i, j), pair(j, i)}) {
                reduced_mass[p] = m;
                diam[p] = d;
                epsilon[p] = e;
                delta[p] = dl;
            }
        }
    }

    const CollisionIntegrals integrals;
    const ThermoTable<Kokkos::HostSpace> thermo = make_thermo_table<Kokkos::HostSpace>(mechanism);
    constexpr size_t NP = 50;
    const double dt = (T_max - T_min) / (NP - 1);
    std::vector<double> T(NP), log_T(NP);
    for (size_t i = 0; i < NP; i++) {
        T[i] = T_min + dt * static_cast<double>(i);
        log_T[i] = std::log(T[i]);
    }
    auto to_array = [](const std::vector<double> & c) {
        std::array<double, 5> a;
        std::copy(c.begin(), c.end(), a.begin());
        return a;
    };

    TransportFits fits;
    std::vector<double> visc(NP), cond(NP), w(NP), w2(NP), diff(NP);
    for (size_t k = 0; k < n; k++) {
        const double ts_298 = BOLTZMANN * 298.0 / eps[k];
        const double fz_298 = 1.0 + std::pow(PI, 1.5) / std::sqrt(ts_298) * (0.5 + 1.0 / ts_298) +
                              (0.25 * PI * PI + 2.0) / ts_298;
        for (size_t i = 0; i < NP; i++) {
            const double t = T[i];
            const double cp_R = thermo.cp_R(static_cast<uint32_t>(k), ThermoTable<Kokkos::HostSpace>::powers(t));
            const double ts = BOLTZMANN * t / eps[k];
            const double om22 = integrals.omega22(ts, delta[pair(k, k)]);
            const double om11 = integrals.omega11(ts, delta[pair(k, k)]);
            const double D = 3.0 / 16.0 * std::sqrt(2.0 * PI / reduced_mass[pair(k, k)]) * std::pow(BOLTZMANN * t, 1.5) /
                             (PI * sigma[k] * sigma[k] * om11);
            const double mu = 5.0 / 16.0 * std::sqrt(PI * mw[k] * BOLTZMANN * t / AVOGADRO) /
                              (om22 * PI * sigma[k] * sigma[k]);
            // Conductivity with Parker's rotational relaxation, as Cantera
            const double f_int = mw[k] / (GAS_CONSTANT * t) * D / mu;
            const double A_factor = 2.5 - f_int;
            const double fz_t = 1.0 + std::pow(PI, 1.5) / std::sqrt(ts) * (0.5 + 1.0 / ts) + (0.25 * PI * PI + 2.0) / ts;
            const double B_factor = zrot[k] * fz_298 / fz_t + 2.0 / PI * (5.0 / 3.0 * crot[k] + f_int);
            const double c1 = 2.0 / PI * A_factor / B_factor;
            const double cv_int = cp_R - 2.5 - crot[k];
            const double f_rot = f_int * (1.0 + c1);
            const double f_trans = 2.5 * (1.0 - c1 * crot[k] / 1.5);
            const double lambda = (mu / mw[k]) * GAS_CONSTANT * (f_trans * 1.5 + f_rot * crot[k] + f_int * cv_int);
            visc[i] = std::sqrt(mu / std::sqrt(t));
            cond[i] = lambda / std::sqrt(t);
            w[i] = 1.0 / (visc[i] * visc[i]);
            w2[i] = 1.0 / (cond[i] * cond[i]);
        }
        fits.viscosity.push_back(to_array(polyfit(log_T, visc, w, 4)));
        fits.conductivity.push_back(to_array(polyfit(log_T, cond, w2, 4)));
    }
    for (size_t k = 0; k < n; k++) {
        for (size_t j = k; j < n; j++) {
            for (size_t i = 0; i < NP; i++) {
                const double t = T[i];
                const double ts = BOLTZMANN * t / epsilon[pair(j, k)];
                const double s = diam[pair(j, k)];
                const double om11 = integrals.omega11(ts, delta[pair(j, k)]);
                const double D = 3.0 / 16.0 * std::sqrt(2.0 * PI / reduced_mass[pair(k, j)]) *
                                 std::pow(BOLTZMANN * t, 1.5) / (PI * s * s * om11);
                diff[i] = D / std::pow(t, 1.5);
                w[i] = 1.0 / (diff[i] * diff[i]);
            }
            fits.diffusion.push_back(to_array(polyfit(log_T, diff, w, 4)));
        }
    }
    return fits;
}

} // namespace chemistry
