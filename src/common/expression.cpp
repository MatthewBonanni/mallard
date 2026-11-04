/**
 * @file expression.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Expressions evaluated on the device at mesh points.
 * @version 0.1
 * @date 2026-11-04
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "expression.h"

std::vector<double> evaluate_at(const std::vector<const Expression *> & expressions,
                                const Kokkos::View<rtype *[N_DIM]> & coords, const std::vector<uint32_t> & rows,
                                const double t) {
    const size_t n_rows = rows.size(), n_expressions = expressions.size();
    if (n_rows == 0 || n_expressions == 0) return std::vector<double>(n_rows * n_expressions);
    std::vector<const expr::Program *> programs;
    for (const Expression * e : expressions) programs.push_back(&e->program());
    const expr::ProgramTable<> table(programs);
    Kokkos::View<uint32_t *> d_rows("expression_rows", n_rows);
    Kokkos::deep_copy(d_rows, Kokkos::View<const uint32_t *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(rows.data(), n_rows));
    Kokkos::View<double **, Kokkos::LayoutRight> values("expression_values", n_rows, n_expressions);
    const Kokkos::View<rtype *[N_DIM]> x = coords;
    Kokkos::parallel_for("evaluate_expressions", n_rows, KOKKOS_LAMBDA(const size_t k) {
        const uint32_t r = d_rows(k);
        const double v[4] = {double(x(r, 0)), double(x(r, 1)), N_DIM > 2 ? double(x(r, N_DIM - 1)) : 0.0, t};
        for (uint32_t e = 0; e < table.size(); e++) values(k, e) = table(e, expr::ArrayVariables{v});
    });
    std::vector<double> result(n_rows * n_expressions);
    Kokkos::deep_copy(Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
                          result.data(), n_rows, n_expressions),
                      values);
    if (const uint32_t stopped = table.stopped(); stopped > 0) {
        throw std::runtime_error(programs[stopped - 1]->runaway_error());
    }
    return result;
}
