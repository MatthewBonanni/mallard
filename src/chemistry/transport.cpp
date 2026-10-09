/**
 * @file transport.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Species transport fits: a port of Cantera's GasTransport fitting
 *        (Cantera 3.2.0, src/transport/GasTransport.cpp and MMCollisionInt.cpp,
 *        BSD-3-Clause, Copyright (c) 2001-2025 Cantera Developers), so that
 *        Mallard's mixture-averaged properties are Cantera's, with the C*
 *        fits of its mixture-averaged thermal diffusion (MixTransport.cpp).
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
                c_poly.push_back(polyfit(delta, std::vector<double>(CSTAR + 8 * (i + 1), CSTAR + 8 * (i + 2)), {}, 6));
            }
        }

        double omega22(const double ts, const double delta) const { return interpolate(ts, delta, OMEGA22, 0, o22_poly); }

        double omega11(const double ts, const double delta) const {
            return omega22(ts, delta) / interpolate(ts, delta, ASTAR, 1, a_poly);
        }

        /**
         * @brief Cantera's fit of C* of degree 8 in ln T* (MMCollisionInt::fit):
         *        unweighted least squares through the table rows (or their
         *        fits in delta*) whose T* bracket [ts_min, ts_max].
         */
        std::vector<double> fit_cstar(const double ts_min, const double ts_max, const double delta) const {
            int n_min = -1, n_max = -1;
            for (int n = 0; n < 37; n++) {
                if (ts_min > TSTAR[n + 1]) n_min = n;
                if (ts_max > TSTAR[n + 1]) n_max = n + 1;
            }
            if (n_min < 0 || n_min >= 36 || n_max < 0 || n_max > 36) {
                n_min = 0;
                n_max = 36;
            }
            if (n_max - n_min + 1 < 9) {
                throw std::runtime_error("the reduced temperature range of the thermal diffusion fits is too narrow.");
            }
            std::vector<double> x, y;
            for (int i = n_min; i <= n_max; i++) {
                x.push_back(log_T[i]);
                y.push_back(delta == 0.0 ? CSTAR[8 * (i + 1)] : poly(c_poly[i], delta));
            }
            return polyfit(x, y, {}, 8);
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
        std::vector<std::vector<double>> o22_poly, a_poly, c_poly;

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
        static constexpr double CSTAR[39 * 8] = {
            0.8889, 0.77778, 0.77778, 0.77778, 0.77778, 0.77778, 0.77778, 0.77778,
            0.88575, 0.8988, 0.8378, 0.8029, 0.7876, 0.7805, 0.7799, 0.7801,
            0.87268, 0.8692, 0.8647, 0.8479, 0.8237, 0.7975, 0.7881, 0.7784,
            0.85182, 0.8525, 0.8366, 0.8198, 0.8054, 0.7903, 0.7839, 0.782,
            0.83542, 0.8362, 0.8306, 0.8196, 0.8076, 0.7918, 0.7842, 0.7806,
            0.82629, 0.8278, 0.8252, 0.8169, 0.8074, 0.7916, 0.7838, 0.7802,
            0.82299, 0.8249, 0.823, 0.8165, 0.8072, 0.7922, 0.7839, 0.7798,
            0.82357, 0.8257, 0.8241, 0.8178, 0.8084, 0.7927, 0.7839, 0.7794,
            0.82657, 0.828, 0.8264, 0.8199, 0.8107, 0.7939, 0.7842, 0.7796,
            0.8311, 0.8234, 0.8295, 0.8228, 0.8136, 0.796, 0.7854, 0.7798,
            0.8363, 0.8366, 0.8342, 0.8267, 0.8168, 0.7986, 0.7864, 0.7805,
            0.84762, 0.8474, 0.8438, 0.8358, 0.825, 0.8041, 0.7904, 0.7822,
            0.85846, 0.8583, 0.853, 0.8444, 0.8336, 0.8118, 0.7957, 0.7854,
            0.8684, 0.8674, 0.8619, 0.8531, 0.8423, 0.8186, 0.8011, 0.7898,
            0.87713, 0.8755, 0.8709, 0.8616, 0.8504, 0.8265, 0.8072, 0.7939,
            0.88479, 0.8831, 0.8779, 0.8695, 0.8578, 0.8338, 0.8133, 0.799,
            0.89972, 0.8986, 0.8936, 0.8846, 0.8742, 0.8504, 0.8294, 0.8125,
            0.91028, 0.9089, 0.9043, 0.8967, 0.8869, 0.8649, 0.8438, 0.8253,
            0.91793, 0.9166, 0.9125, 0.9058, 0.897, 0.8768, 0.8557, 0.8372,
            0.92371, 0.9226, 0.9189, 0.9128, 0.905, 0.8861, 0.8664, 0.8484,
            0.93135, 0.9304, 0.9274, 0.9226, 0.9164, 0.9006, 0.8833, 0.8662,
            0.93607, 0.9353, 0.9329, 0.9291, 0.924, 0.9109, 0.8958, 0.8802,
            0.93927, 0.9387, 0.9366, 0.9334, 0.9292, 0.9162, 0.905, 0.8911,
            0.94149, 0.9409, 0.9393, 0.9366, 0.9331, 0.9236, 0.9122, 0.8997,
            0.94306, 0.9426, 0.9412, 0.9388, 0.9357, 0.9276, 0.9175, 0.9065,
            0.94419, 0.9437, 0.9425, 0.9406, 0.938, 0.9308, 0.9219, 0.9119,
            0.94571, 0.9455, 0.9445, 0.943, 0.9409, 0.9353, 0.9283, 0.9201,
            0.94662, 0.9464, 0.9456, 0.9444, 0.9428, 0.9382, 0.9325, 0.9258,
            0.94723, 0.9471, 0.9464, 0.9455, 0.9442, 0.9405, 0.9355, 0.9298,
            0.94764, 0.9474, 0.9469, 0.9462, 0.945, 0.9418, 0.9378, 0.9328,
            0.9479, 0.9478, 0.9474, 0.9465, 0.9457, 0.943, 0.9394, 0.9352,
            0.94827, 0.9481, 0.948, 0.9472, 0.9467, 0.9447, 0.9422, 0.9391,
            0.94842, 0.9484, 0.9481, 0.9478, 0.9472, 0.9458, 0.9437, 0.9415,
            0.94852, 0.9484, 0.9483, 0.948, 0.9475, 0.9465, 0.9449, 0.943,
            0.94861, 0.9487, 0.9484, 0.9481, 0.9479, 0.9468, 0.9455, 0.943,
            0.94872, 0.9486, 0.9486, 0.9483, 0.9482, 0.9475, 0.9464, 0.9452,
            0.94881, 0.9488, 0.9489, 0.949, 0.9487, 0.9482, 0.9476, 0.9468,
            0.94863, 0.9487, 0.9489, 0.9491, 0.9493, 0.9491, 0.9483, 0.9476,
            0.94444, 0.94444, 0.94444, 0.94444, 0.94444, 0.94444, 0.94444, 0.94444};
};

/** @brief Pair parameters of the collision integrals, (i, j) flattened as i * n + j. */
struct PairParameters {
    std::vector<double> reduced_mass, diameter, well_depth, delta;
};

/** @brief Pair parameters, with Cantera's polar/nonpolar corrections of the well depth and diameter. */
PairParameters pair_parameters(const Mechanism & mechanism) {
    constexpr double PI = std::numbers::pi;
    const size_t n = mechanism.n_species();
    PairParameters P;
    P.reduced_mass.resize(n * n);
    P.diameter.resize(n * n);
    P.well_depth.resize(n * n);
    P.delta.resize(n * n);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i; j < n; j++) {
            const SpeciesTransport & a = mechanism.species[i].transport;
            const SpeciesTransport & b = mechanism.species[j].transport;
            const double mw_i = mechanism.species[i].molecular_weight, mw_j = mechanism.species[j].molecular_weight;
            const double m = mw_i * mw_j / (AVOGADRO * (mw_i + mw_j));
            double d = 0.5 * (a.diameter + b.diameter);
            double e = std::sqrt(a.well_depth * b.well_depth);
            const double mu = std::sqrt(a.dipole * b.dipole);
            const double dl = 0.5 * mu * mu / (4.0 * PI * EPSILON_0 * e * d * d * d);
            const bool polar_i = a.dipole > 0.0, polar_j = b.dipole > 0.0;
            if (polar_i != polar_j) {
                const SpeciesTransport & p = polar_i ? a : b;
                const SpeciesTransport & np = polar_i ? b : a;
                const double alpha_star = np.polarizability / std::pow(np.diameter, 3);
                const double mu_p_star = p.dipole / std::sqrt(4.0 * PI * EPSILON_0 * std::pow(p.diameter, 3) * p.well_depth);
                const double xi = 1.0 + 0.25 * alpha_star * mu_p_star * mu_p_star * std::sqrt(p.well_depth / np.well_depth);
                d *= std::pow(xi, -1.0 / 6.0);
                e *= xi * xi;
            }
            for (size_t p : {i * n + j, j * n + i}) {
                P.reduced_mass[p] = m;
                P.diameter[p] = d;
                P.well_depth[p] = e;
                P.delta[p] = dl;
            }
        }
    }
    return P;
}

/** @brief The common thermo temperature range of a mechanism's species. */
std::pair<double, double> temperature_range(const Mechanism & mechanism) {
    double T_min = 0.0, T_max = 1e300;
    for (const Species & sp : mechanism.species) {
        T_min = std::max(T_min, sp.thermo.T_bounds.front());
        T_max = std::min(T_max, sp.thermo.T_bounds.back());
    }
    if (!(T_min > 0.0 && T_max > T_min && T_max < 1e300)) {
        throw std::runtime_error("Mechanism " + mechanism.file + ": transport fits need a common, finite thermo range.");
    }
    return {T_min, T_max};
}

void check_transport_data(const Mechanism & mechanism) {
    for (const Species & sp : mechanism.species) {
        if (!sp.has_transport) {
            throw std::runtime_error("Mechanism " + mechanism.file + ": species " + sp.name +
                                     " has no gas transport data.");
        }
    }
}

} // namespace

TransportFits fit_transport(const Mechanism & mechanism) {
    constexpr double PI = std::numbers::pi;
    check_transport_data(mechanism);
    const size_t n = mechanism.n_species();
    std::vector<double> mw(n), sigma(n), eps(n), zrot(n), crot(n);
    for (size_t k = 0; k < n; k++) {
        const Species & sp = mechanism.species[k];
        mw[k] = sp.molecular_weight;
        sigma[k] = sp.transport.diameter;
        eps[k] = sp.transport.well_depth;
        zrot[k] = sp.transport.rotational_relaxation;
        crot[k] = sp.transport.geometry == MoleculeGeometry::ATOM ? 0.0
                  : sp.transport.geometry == MoleculeGeometry::LINEAR ? 1.0 : 1.5;
    }
    const auto [T_min, T_max] = temperature_range(mechanism);
    auto pair = [&](size_t i, size_t j) { return i * n + j; };
    const PairParameters P = pair_parameters(mechanism);
    const std::vector<double> & reduced_mass = P.reduced_mass;
    const std::vector<double> & diam = P.diameter;
    const std::vector<double> & epsilon = P.well_depth;
    const std::vector<double> & delta = P.delta;

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

ThermalDiffusionFits fit_thermal_diffusion(const Mechanism & mechanism) {
    check_transport_data(mechanism);
    const size_t n = mechanism.n_species();
    const auto [T_min, T_max] = temperature_range(mechanism);
    const PairParameters P = pair_parameters(mechanism);
    double ts_min = 1e8, ts_max = 0.0;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i; j < n; j++) {
            ts_min = std::min(ts_min, BOLTZMANN * T_min / P.well_depth[i * n + j]);
            ts_max = std::max(ts_max, BOLTZMANN * T_max / P.well_depth[i * n + j]);
        }
    }
    const CollisionIntegrals integrals;
    ThermalDiffusionFits fits;
    std::vector<double> deltas;
    std::vector<std::array<double, 9>> by_delta;
    for (size_t k = 0; k < n; k++) {
        for (size_t j = k; j < n; j++) {
            const double delta = P.delta[k * n + j];
            const auto found = std::find(deltas.begin(), deltas.end(), delta);
            if (found == deltas.end()) {
                std::array<double, 9> c;
                const std::vector<double> fit = integrals.fit_cstar(ts_min, ts_max, delta);
                std::copy(fit.begin(), fit.end(), c.begin());
                deltas.push_back(delta);
                by_delta.push_back(c);
                fits.cstar.push_back(c);
            } else {
                fits.cstar.push_back(by_delta[found - deltas.begin()]);
            }
            fits.well_depth.push_back(P.well_depth[k * n + j]);
        }
    }
    return fits;
}

} // namespace chemistry
