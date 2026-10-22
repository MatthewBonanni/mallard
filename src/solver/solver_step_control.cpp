/**
 * @file solver_step_control.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Error-controlled time steps ([numerics.error_control]).
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "solver.h"

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include "input.h"

void Solver::init_error_control() {
    step_controller = StepController();
    const bool requested = input.contains("numerics") && input.at("numerics").contains("error_control");
    if (!requested) return;
    const Input & table = input.at("numerics").at("error_control");
    const std::string name = TIME_INTEGRATOR_NAMES.at(time_integrator->get_type());
    if (time_integrator->get_error_order() == 0) {
        throw InputError("[numerics.error_control] needs a time integrator with an embedded error estimate "
                         "(RDPK3Sp35), not " + name + ".");
    }
    if (reacting) {
        throw InputError("[numerics.error_control] is not available with [chemistry] (a rejected step would "
                         "have to undo the chemistry substeps).");
    }
    if (double_flux) {
        throw InputError("[numerics.error_control] is not available with numerics.double_flux.");
    }
    const rtype tolerance = find_real_or(table, "tolerance", -1.0_r);
    const rtype abs_tol = find_real_or(table, "abs_tolerance", tolerance);
    const rtype rel_tol = find_real_or(table, "rel_tolerance", tolerance);
    if (!(abs_tol > 0.0_r) || !(rel_tol >= 0.0_r)) {
        throw InputError("numerics.error_control: give tolerance (or abs_tolerance > 0 and rel_tolerance >= 0).");
    }
    std::array<rtype, 3> beta = time_integrator->get_controller_beta();
    if (table.contains("beta")) {
        const std::vector<double> b = toml::find<std::vector<double>>(table, "beta");
        if (b.size() != 3) throw InputError("numerics.error_control.beta must have three entries.");
        for (size_t i = 0; i < 3; i++) beta[i] = static_cast<rtype>(b[i]);
    }
    step_controller.init(abs_tol, rel_tol, beta, time_integrator->get_error_order());
}

void Solver::check_rejections(const int tries) const {
    constexpr int MAX_TRIES = 50;
    if (tries >= MAX_TRIES) {
        throw std::runtime_error("the error controller rejected step " + std::to_string(step) + " " +
                                 std::to_string(MAX_TRIES) + " times (last dt " + std::to_string(double(dt)) +
                                 "); the solution may have blown up.");
    }
}

bool Solver::accept_controlled_step() {
    step_controller.estimate(solution_vec.back(), state(), rhs_vec.back(), mesh->n_owned());
    if (is_distributed()) {
        const Kokkos::View<rtype, Kokkos::MemoryTraits<Kokkos::Unmanaged>> error(step_controller.error_device());
        rtype local;
        Kokkos::deep_copy(local, error);
        Kokkos::deep_copy(error, comm::allreduce(local, comm::Op::MAX));
    }
    step_controller.update(dt, nullptr, state(), solution_vec.back());
    return step_controller.accepted();
}
