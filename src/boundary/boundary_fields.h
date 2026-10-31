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
 * @brief A vector key of a [[boundaries]] entry (u, mass_flux): N_DIM
 *        components, each a number or an expression, or one number or
 *        expression, a magnitude along `direction` (the inward normal of each
 *        face if absent). A vector together with `direction` is an error.
 */
struct VectorKey {
    std::vector<Input> items;  // the components, or the magnitude alone
    bool speed = false;        // a magnitude along direction
    bool directed = false;     // direction given (else the inward normal)
    Point direction{};         // unit

    /** @brief Read entry.key; empty items if absent. */
    VectorKey(const Input & entry, const std::string & key, const std::string & where);

    /** @brief The unit direction of the magnitude at a face of inward normal n_in. */
    Point direction_at(const Point & n_in) const { return directed ? direction : n_in; }
};

/**
 * @brief The numbers of a condition that may be expressions: the velocity of
 *        inlet_state, farfield and walls, given as components or as a speed
 *        along a direction; p and T of inlet_state (whose velocity with nscbc
 *        is InletProfile's) and farfield; p0 and T0 of inlet_stagnation;
 *        mass_flux (a number or a vector) and T of inlet_mass_flow; T of
 *        wall_isothermal, q of wall_heat_flux and p of outlet_pressure.
 */
class BoundaryFields {
    public:
        /**
         * @brief Value slots: the velocity components (or the speed in the first,
         *        or the mass flux vector), p (p0), T (T0), mass_flux (q of walls).
         */
        enum Slot { SLOT_U = 0, SLOT_P = 3, SLOT_T = 4, SLOT_FLUX = 5, N_SLOTS = 6 };

        /** @brief How the vector of the condition is given: absent, by components, or as a speed along a direction. */
        enum class Vector : uint8_t { NONE, COMPONENTS, SPEED };

        BoundaryFields(const Input & input, const BoundaryCondition & bc, const std::string & where);

        /** @brief Whether some number is an expression: each face then gets a condition of its own. */
        bool varies() const;
        bool depends_on_time() const;

        /**
         * @brief Whether the faces' conditions differ even when no number is an
         *        expression: a speed along the normal, a mass flux vector.
         */
        bool per_face() const {
            return (vector == Vector::SPEED && !directed) ||
                   (type == BoundaryType::INLET_MASS_FLOW && vector == Vector::COMPONENTS);
        }

        /** @brief The temperature the condition prescribes (T, or T0 of inlet_stagnation), 0 if none. */
        double temperature(const Point & x, double t) const;

        /**
         * @brief Write the condition's numbers at x and t into bc; densities of
         *        inlet_state and farfield from p and T with gas constant R.
         * @param n_in Unit inward normal of the face: the direction of a speed
         *        without one, and the normal of a mass flux vector.
         * @param numbers_only Write only the numbers that are not expressions
         *        (the condition before it is split by face).
         */
        void apply(BoundaryCondition & bc, const Point & x, double t, double R, const Point & n_in,
                   bool numbers_only = false) const;

        BoundaryType get_type() const { return type; }
        Vector get_vector() const { return vector; }
        const std::string & get_where() const { return where; }

        /** @brief Whether a speed has a direction of its own (else it is along the inward normal). */
        bool is_directed() const { return directed; }

        /** @brief The fixed unit direction of a speed (with is_directed). */
        const Point & get_direction() const { return direction; }

        /** @brief The value in slot s, null if the condition has none there. */
        const BoundaryValue * slot(int s) const;

        /** @brief The key of slot s in messages. */
        std::string slot_key(int s) const;

    private:
        BoundaryType type;
        std::string where;
        Vector vector = Vector::NONE;
        bool directed = false;
        Point direction{};
        std::vector<BoundaryValue> u;  // components, or the speed alone; empty if absent
        BoundaryValue p, T, flux;
        bool has_p = false, has_T = false, has_flux = false;
        std::string vector_key, p_key, T_key, flux_key;

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
        /**
         * @brief Condition i_bc of the face at x, of inward normal n_in, takes
         *        `fields`, and for mixtures `composition` (null: fixed).
         */
        void add(uint32_t i_bc, const Point & x, const Point & n_in, const std::shared_ptr<const BoundaryFields> & fields,
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
            Point n_in;
        };
        std::vector<std::shared_ptr<const BoundaryFields>> sets;
        std::vector<std::shared_ptr<const CompositionPrograms>> compositions;  // per set
        std::vector<FaceHost> faces_host;

        Kokkos::View<uint32_t *> face_bc;
        Kokkos::View<uint32_t *> face_set;
        Kokkos::View<double *[3]> face_x;
        Kokkos::View<double *[N_DIM]> face_direction;  // the speed's direction, or the normal of a mass flux vector
        Kokkos::View<uint8_t *[2]> set_type;            // type, Vector
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
 * @brief Velocity targets of inlet_state with nscbc that depend on t, evaluated on the
 *        device at a time into a target array (char_target or char_target_next).
 */
class TimedInletTargets {
    public:
        /** @brief Whether the velocity u of a [[boundaries]] entry (a VectorKey) depends on t. */
        static bool depends_on_time(const Input & entry, const std::string & where);

        /** @brief Characteristic faces `chars` with centers x and inward normals n_in take the entry's velocity u. */
        void add(const Input & entry, const std::string & where, const std::vector<int32_t> & chars,
                 const std::vector<Point> & x, const std::vector<Point> & n_in);

        /** @brief Copy what add() gathered to the device; call once, after the last add(). */
        void finish();

        bool empty() const { return host_chars.empty(); }

        void fill(double t, const Kokkos::View<rtype *[N_DIM]> & target) const;

        /** @brief Throw the InputError of an expression stopped at the iteration limit, if any was. */
        void check() const;

    private:
        std::vector<std::shared_ptr<expr::Program>> host_programs;
        std::vector<std::array<uint32_t, N_DIM>> host_slots;
        std::vector<Point> host_scale;  // 1, or the direction of a speed
        std::vector<int32_t> host_chars;
        std::vector<Point> host_x;

        Kokkos::View<int32_t *> chars;
        Kokkos::View<double *[3]> x;
        Kokkos::View<uint32_t *[N_DIM]> slots;
        Kokkos::View<double *[N_DIM]> scale;
        expr::ProgramTable<> programs;
};

#endif // BOUNDARY_FIELDS_H
