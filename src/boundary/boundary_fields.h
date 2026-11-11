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
#include "kexpr.h"
#include "input.h"
#include "thermo.h"

/** @brief Point in space. */
using Point = std::array<double, N_DIM>;

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
        const std::shared_ptr<kexpr::Program> & compiled() const { return program; }

    private:
        std::shared_ptr<kexpr::Program> program;
};

/**
 * @brief The `direction` of a [[boundaries]] entry: N_DIM numbers, or N_DIM
 *        components of which some are expressions of x, y, z, t, normalized
 *        where they are evaluated.
 */
class DirectionKey {
    public:
        DirectionKey() = default;
        /** @brief Read entry.direction, which must be present. */
        DirectionKey(const Input & entry, const std::string & where);

        /** @brief Whether entry.direction has an expression among its components. */
        static bool is_expression(const Input & entry);

        bool constant() const { return components.empty(); }
        bool depends_on_time() const;

        /** @brief The unit direction at x and t; an error if it is zero there. */
        Point at(const Point & x, double t) const;

        /** @brief The unit direction at x and t, checked to point into the domain at a face of inward normal n_in. */
        Point inward(const Point & x, double t, const Point & n_in) const;

        /** @brief The compiled components (expressions only). */
        const std::vector<BoundaryValue> & programs() const { return components; }

    private:
        std::string where;
        Point value{};                          // unit, if constant
        std::vector<BoundaryValue> components;  // if some component is an expression
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
    DirectionKey direction;

    /** @brief Read entry.key; empty items if absent. */
    VectorKey(const Input & entry, const std::string & key, const std::string & where);

    /** @brief The unit direction of the magnitude at x and t on a face of inward normal n_in. */
    Point direction_at(const Point & x, double t, const Point & n_in) const {
        return directed ? direction.at(x, t) : n_in;
    }
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
         *        expression: a speed along the normal or a varying direction, a mass flux vector.
         */
        bool per_face() const {
            return (vector == Vector::SPEED && (!directed || !direction.constant())) ||
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

        /** @brief Whether the condition has a direction of its own (else a speed is along the inward normal). */
        bool is_directed() const { return directed; }

        /** @brief The direction (with is_directed). */
        const DirectionKey & get_direction() const { return direction; }

        /** @brief The direction of a speed at x and t on a face of inward normal n_in. */
        Point speed_direction(const Point & x, double t, const Point & n_in) const {
            return directed ? direction.at(x, t) : n_in;
        }

        /** @brief The value in slot s, null if the condition has none there. */
        const BoundaryValue * slot(int s) const;

        /** @brief The key of slot s in messages. */
        std::string slot_key(int s) const;

    private:
        BoundaryType type;
        std::string where;
        Vector vector = Vector::NONE;
        bool directed = false;
        DirectionKey direction;
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
    std::vector<std::shared_ptr<kexpr::Program>> fractions;
    int32_t balance = -1;
    bool mole = false;

    bool depends_on_time() const;
};

/**
 * @brief The `profile` (1 if none) and `mdot` of an inlet_mass_flow, one of
 *        them of t, and every face of the inlet on all ranks, in the order of
 *        its area integral.
 */
struct MassFlowProfile {
    std::string where;
    std::shared_ptr<const kexpr::Program> shape;
    std::shared_ptr<const kexpr::Program> mdot;  // of t only
    std::vector<Point> x;          // face centers
    std::vector<double> measure;   // face measures as the flux integral weighs them
};

/**
 * @brief The conditions of the faces whose numbers depend on t, evaluated on
 *        the device at the stage time held in a device scalar, so that a
 *        captured CUDA graph replays them: velocities, densities and
 *        pressures, and for gas mixtures the composition, the thermodynamic
 *        surrogates [gamma, e0] and the transport values (T, X); and the
 *        mass fluxes of inlet_mass_flow profiles of t.
 */
class TimedBoundaryFields {
    public:
        /**
         * @brief Condition i_bc of the face at x, of inward normal n_in, takes
         *        `fields`, and for mixtures `composition` (null: fixed).
         *        Without a face (`at_face` false), x and n_in are 0 and errors
         *        give no position.
         */
        void add(uint32_t i_bc, const Point & x, const Point & n_in, const std::shared_ptr<const BoundaryFields> & fields,
                 const std::shared_ptr<const CompositionPrograms> & composition, bool at_face = true);

        /**
         * @brief The mass flux of conditions `bcs` at x on this rank: mdot(t) spread over the whole inlet by
         *        `profile`, at each stage. Every rank integrates the profile over every face in the same
         *        order, so the fluxes are the same on any rank count with no communication.
         */
        void add(const MassFlowProfile & profile, const std::vector<uint32_t> & bcs, const std::vector<Point> & x);

        /** @brief Copy what add() gathered to the device; call once, after the last add(). */
        void finish();

        bool empty() const { return faces_host.empty() && profiles.empty(); }

        /** @brief The errors of mass-flow profiles found on the device. */
        enum ProfileError { NEGATIVE_PROFILE, ZERO_INTEGRAL, NEGATIVE_MDOT };

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
            bool at_face;
        };
        std::vector<std::shared_ptr<const BoundaryFields>> sets;
        std::vector<std::shared_ptr<const CompositionPrograms>> compositions;  // per set
        std::vector<FaceHost> faces_host;

        Kokkos::View<uint32_t *> face_bc;
        Kokkos::View<uint32_t *> face_set;
        Kokkos::View<double *[3]> face_x;
        // The speed's direction, or the inward normal (of a mass flux vector, or of a direction of t)
        Kokkos::View<double *[N_DIM]> face_direction;
        Kokkos::View<uint8_t *[2]> set_type;            // type, Vector
        Kokkos::View<int32_t *[BoundaryFields::N_SLOTS]> set_slot;  // program, -1 if none
        Kokkos::View<int32_t *[4]> set_composition;                 // first, count, balance, mole
        Kokkos::View<int32_t *[N_DIM]> set_direction;               // programs of a direction of t, -1 if none
        Kokkos::View<uint32_t *> composition_species;
        Kokkos::View<uint32_t *> composition_program;
        kexpr::ProgramTable<> programs;
        std::vector<const kexpr::Program *> program_list;  // host, in the table's order
        Kokkos::View<int32_t> error_face;   // 1 + the first face whose number was not positive, 0 if none
        Kokkos::View<double[3]> error;      // its time, value and slot (N_SLOTS: composition, N_SLOTS + 1: direction)

        // Mass-flow profiles: all inlet faces of profile p are [first(p), first(p + 1)), summed in chunks
        // of CHUNK in order; the faces of this rank take the scale times their value
        static constexpr uint32_t CHUNK = 256;
        std::vector<MassFlowProfile> profiles;
        // This rank's faces: condition, center and profile
        std::vector<uint32_t> profile_bc_host;
        std::vector<Point> profile_x_host;
        std::vector<uint32_t> profile_of_host;
        std::vector<uint32_t> first;            // first inlet face of each profile, then the total
        Kokkos::View<uint32_t *> profile_program;
        Kokkos::View<uint32_t *> profile_mdot;    // program
        Kokkos::View<double *[3]> inlet_x;
        Kokkos::View<double *> inlet_measure;
        Kokkos::View<uint32_t *[3]> chunk_range;  // profile, first and end face
        Kokkos::View<uint32_t *> profile_chunks;  // first chunk of each profile, then the total
        Kokkos::View<double *> chunk_sum;
        Kokkos::View<double *> profile_scale;
        Kokkos::View<uint32_t *> profile_bc;
        Kokkos::View<uint32_t *> profile_of;
        Kokkos::View<double *[3]> profile_x;
        Kokkos::View<int32_t> profile_error;          // 1 + the inlet face or profile of the first error, 0 if none
        Kokkos::View<double[3]> profile_error_value;  // its time, value and ProfileError

        void finish_profiles(std::vector<const kexpr::Program *> & all);
        void apply_profiles(const Kokkos::View<rtype> & time, const Kokkos::View<BoundaryCondition *> & bcs) const;
        void check_profiles() const;
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
        void add(const std::vector<std::shared_ptr<const kexpr::Program>> & W, const std::vector<int32_t> & states,
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
        std::vector<std::shared_ptr<const kexpr::Program>> host_programs;  // N_DIM + 2 per set
        std::vector<bool> set_timed;
        std::vector<FaceHost> faces_host;
        uint32_t n_timed = 0;  // faces whose states depend on t, first on the device

        Kokkos::View<int32_t *> face_state_index;
        Kokkos::View<uint32_t *> face_set;
        Kokkos::View<double *[3]> face_x;
        kexpr::ProgramTable<> programs;
};

/**
 * @brief Velocity targets of inlet_state with nscbc that depend on t, evaluated on the
 *        device at a time into a target array (char_target or char_target_next).
 */
class TimedInletTargets {
    public:
        /** @brief Whether the velocity u of a [[boundaries]] entry (a VectorKey, with its direction) depends on t. */
        static bool depends_on_time(const Input & entry, const std::string & where);

        /** @brief Characteristic faces `chars` with centers x and inward normals n_in take the entry's velocity u. */
        void add(const Input & entry, const std::string & where, const std::vector<int32_t> & chars,
                 const std::vector<Point> & x, const std::vector<Point> & n_in);

        /** @brief Copy what add() gathered to the device; call once, after the last add(). */
        void finish();

        bool empty() const { return host_chars.empty(); }

        /** @brief The targets at the time on the device (kernels only, capturable in a step graph). */
        void fill(const Kokkos::View<rtype> & time, const Kokkos::View<rtype *[N_DIM]> & target) const;

        /** @brief Throw the InputError of an expression stopped at the iteration limit, or of a direction of t
         *         that was zero or pointed out of the domain, if any was. */
        void check() const;

    private:
        std::vector<std::shared_ptr<kexpr::Program>> host_programs;
        std::vector<std::array<uint32_t, N_DIM>> host_slots;
        std::vector<std::array<int32_t, N_DIM>> host_direction;  // programs of a direction of t, -1 if none
        std::vector<Point> host_scale;  // 1, the direction of a speed, or the inward normal for a direction of t
        std::vector<int32_t> host_chars;
        std::vector<Point> host_x;
        std::vector<std::string> host_where;  // per face

        Kokkos::View<int32_t *> chars;
        Kokkos::View<double *[3]> x;
        Kokkos::View<uint32_t *[N_DIM]> slots;
        Kokkos::View<int32_t *[N_DIM]> direction;
        Kokkos::View<double *[N_DIM]> scale;
        kexpr::ProgramTable<> programs;
        Kokkos::View<int32_t> error_face;  // 1 + the first face whose direction was zero or outward, 0 if none
        Kokkos::View<double[2]> error;     // its time and value along the inward normal
};

#endif // BOUNDARY_FIELDS_H
