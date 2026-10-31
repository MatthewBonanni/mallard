/**
 * @file boundary_fields.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief The numbers of boundary conditions given as expressions of x, y, z
 *        and t, evaluated per face and, on the device, at the stage times.
 * @version 0.1
 * @date 2026-10-16
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef BOUNDARY_FIELDS_H
#define BOUNDARY_FIELDS_H

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "boundary.h"
#include "compiled_expression.h"
#include "input.h"
#include "synthetic_inflow.h"
#include "thermo.h"

/** @brief A number of a [[boundaries]] entry, or an expression of x, y, z and t, compiled. */
class BoundaryValue {
    public:
        BoundaryValue() = default;
        BoundaryValue(const Input & v, const std::string & name, const Dimension & dimension);

        /** @brief Whether v is an expression rather than a number or a quantity with a unit. */
        static bool is_expression(const Input & v);

        /** @brief The variables of the expressions, by index: x, y, z, t. */
        static const std::vector<std::string> & variables();

        bool varies() const { return program && !program->is_constant(); }
        bool depends_on_time() const { return program && program->uses(3); }
        double at(const Point & x, double t) const;

        /** @brief The compiled value (a constant for a number); null if absent. */
        const std::shared_ptr<expr::Program> & compiled() const { return program; }

    private:
        std::shared_ptr<expr::Program> program;
};

/**
 * @brief The numbers of a condition that may be expressions: the velocity of
 *        upt and of walls, p and T of upt and nscbc_inlet (whose velocity is
 *        InletProfile's), p0 and T0 of stagnation_inlet, mass_flux and T of
 *        mass_flow_inlet.
 */
class BoundaryFields {
    public:
        /** @brief Value slots: the velocity components, p (p0), T (T0), mass_flux. */
        enum Slot { SLOT_U = 0, SLOT_P = 3, SLOT_T = 4, SLOT_FLUX = 5, N_SLOTS = 6 };

        BoundaryFields(const Input & input, BoundaryType type, const std::string & where);

        /** @brief Whether some number is an expression: each face then gets a condition of its own. */
        bool varies() const;
        bool depends_on_time() const;

        /** @brief The temperature the condition prescribes (T, or T0 of stagnation_inlet), 0 if none. */
        double temperature(const Point & x, double t) const;

        /**
         * @brief Write the condition's numbers at x and t into bc; densities of
         *        upt and nscbc_inlet from p and T with gas constant R.
         * @param numbers_only Write only the numbers that are not expressions
         *        (the condition before it is split by face).
         */
        void apply(BoundaryCondition & bc, const Point & x, double t, double R, bool numbers_only = false) const;

        BoundaryType get_type() const { return type; }
        const std::string & get_where() const { return where; }

        /** @brief The value in slot s, null if the condition has none there. */
        const BoundaryValue * slot(int s) const;

        /** @brief The key of slot s in messages. */
        std::string slot_key(int s) const;

    private:
        BoundaryType type;
        std::string where;
        std::vector<BoundaryValue> u;  // upt, walls; empty if absent
        BoundaryValue p, T, flux;
        bool has_p = false, has_T = false, has_flux = false;

        std::vector<const BoundaryValue *> values() const;
};

/**
 * @brief A mixture composition given as X or Y by species, each a number or
 *        an expression of x, y, z, t, compiled: an optional balance species
 *        takes 1 - sum of the others, else the values are normalized.
 */
struct CompositionPrograms {
    std::vector<uint32_t> species;
    std::vector<std::shared_ptr<expr::Program>> fractions;
    int32_t balance = -1;
    bool mole = false;

    bool depends_on_time() const;
};

/**
 * @brief The conditions of the faces whose numbers depend on t, evaluated on
 *        the device at the stage time held in a device scalar, so that a
 *        captured CUDA graph replays them: velocities, densities and
 *        pressures, and for gas mixtures the composition, the thermodynamic
 *        surrogates [gamma, e0] and the transport values (T, X).
 */
class TimedBoundaryFields {
    public:
        /** @brief Condition i_bc of the face at x takes `fields`, and for mixtures `composition` (null: fixed). */
        void add(uint32_t i_bc, const Point & x, const std::shared_ptr<const BoundaryFields> & fields,
                 const std::shared_ptr<const CompositionPrograms> & composition);

        /** @brief Copy what add() gathered to the device; call once, after the last add(). */
        void finish();

        bool empty() const { return faces_host.empty(); }

        /** @brief The gas of the conditions: the gas constant R of a single gas, or the mixture's tables. */
        struct Gas {
            double R = 0.0;
            bool mixture = false;
            chemistry::ThermoTable<> thermo;
            Kokkos::View<rtype **, Kokkos::LayoutRight> bc_Y;
            Kokkos::View<rtype *[2]> bc_thermo;
            Kokkos::View<rtype **, Kokkos::LayoutRight> transport;  // (T, X) per condition; empty if inviscid
        };

        /** @brief The faces' conditions at the time held in `time`. */
        void apply(const Kokkos::View<rtype> & time, const Kokkos::View<BoundaryCondition *> & bcs,
                   const Gas & gas) const;

        /** @brief Throw the InputError of the first number that was not positive, or of an expression stopped
         *         at the iteration limit, if any. */
        void check() const;

    private:
        struct FaceHost {
            uint32_t i_bc;
            uint32_t set;
            Point x;
        };
        std::vector<std::shared_ptr<const BoundaryFields>> sets;
        std::vector<std::shared_ptr<const CompositionPrograms>> compositions;  // per set
        std::vector<FaceHost> faces_host;

        Kokkos::View<uint32_t *> face_bc;
        Kokkos::View<uint32_t *> face_set;
        Kokkos::View<double *[3]> face_x;
        Kokkos::View<uint8_t *> set_type;
        Kokkos::View<int32_t *[BoundaryFields::N_SLOTS]> set_slot;  // program, -1 if none
        Kokkos::View<int32_t *[4]> set_composition;                 // first, count, balance, mole
        Kokkos::View<uint32_t *> composition_species;
        Kokkos::View<uint32_t *> composition_program;
        expr::ProgramTable<> programs;
        std::vector<const expr::Program *> program_list;  // host, in the table's order
        Kokkos::View<int32_t> error_face;   // 1 + the first face whose number was not positive, 0 if none
        Kokkos::View<double[3]> error;      // its time, value and slot
};

/**
 * @brief The exterior states [rho, u, p] of dirichlet faces, expressions of
 *        x, y, z and t, evaluated on the device: those of t at the stage time
 *        held in a device scalar, so that a captured CUDA graph replays them,
 *        the others once.
 */
class DirichletStates {
    public:
        /** @brief Face states `states` (indices into BoundaryData::face_state) at x take the programs W. */
        void add(const std::vector<std::shared_ptr<const expr::Program>> & W, const std::vector<int32_t> & states,
                 const std::vector<Point> & x);

        /** @brief Copy what add() gathered to the device and write the states that do not depend on t. */
        void finish(const Kokkos::View<rtype *[N_DIM + 2]> & face_state);

        bool depends_on_time() const { return n_timed > 0; }

        /** @brief The states that depend on t, at the time held in `time`. */
        void apply(const Kokkos::View<rtype> & time, const Kokkos::View<rtype *[N_DIM + 2]> & face_state) const;

        /** @brief Throw the InputError of an expression stopped at the iteration limit, if any was. */
        void check() const;

    private:
        struct FaceHost {
            int32_t state;
            uint32_t set;
            Point x;
        };
        std::vector<std::shared_ptr<const expr::Program>> host_programs;  // N_DIM + 2 per set
        std::vector<bool> set_timed;
        std::vector<FaceHost> faces_host;
        uint32_t n_timed = 0;  // faces whose states depend on t, first on the device

        Kokkos::View<int32_t *> face_state_index;
        Kokkos::View<uint32_t *> face_set;
        Kokkos::View<double *[3]> face_x;
        expr::ProgramTable<> programs;
};

/**
 * @brief nscbc_inlet velocity targets that depend on t, evaluated on the
 *        device at a time into a target array (char_target or char_target_next).
 */
class TimedInletTargets {
    public:
        /** @brief Whether the velocity u (N_DIM numbers or expressions) depends on t. */
        static bool depends_on_time(const Input & u, const std::string & where);

        /** @brief Characteristic faces `chars` with centers x take the velocity u. */
        void add(const Input & u, const std::string & where, const std::vector<int32_t> & chars,
                 const std::vector<Point> & x);

        /** @brief Copy what add() gathered to the device; call once, after the last add(). */
        void finish();

        bool empty() const { return host_chars.empty(); }

        void fill(double t, const Kokkos::View<rtype *[N_DIM]> & target) const;

        /** @brief Throw the InputError of an expression stopped at the iteration limit, if any was. */
        void check() const;

    private:
        std::vector<std::shared_ptr<expr::Program>> host_programs;
        std::vector<std::array<uint32_t, N_DIM>> host_slots;
        std::vector<int32_t> host_chars;
        std::vector<Point> host_x;

        Kokkos::View<int32_t *> chars;
        Kokkos::View<double *[3]> x;
        Kokkos::View<uint32_t *[N_DIM]> slots;
        expr::ProgramTable<> programs;
};

#endif // BOUNDARY_FIELDS_H
