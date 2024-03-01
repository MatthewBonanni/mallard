/**
 * @file solver_functors.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Functors for the Solver class.
 * @version 0.1
 * @date 2024-02-29
 * 
 * @copyright Copyright (c) 2024 Matthew Bonanni
 * 
 */

struct FluxFunctor {
    public:
        /**
         * @brief Construct a new Flux Functor object
         * @param normals Face normals.
         * @param face_area Face areas.
         * @param cells_of_face Cells of face.
         * @param face_solution Face solution.
         * @param rhs RHS.
         * @param riemann_solver Riemann solver.
         * @param physics Physics.
         */
        FluxFunctor(Kokkos::View<rtype *[N_DIM]> normals,
                    Kokkos::View<rtype *> face_area,
                    Kokkos::View<int32_t *[2]> cells_of_face,
                    Kokkos::View<rtype **[N_CONSERVATIVE]> face_solution,
                    Kokkos::View<rtype *[N_CONSERVATIVE]> rhs,
                    RiemannSolver riemann_solver,
                    Physics physics) :
                        normals(normals),
                        face_area(face_area),
                        cells_of_face(cells_of_face),
                        face_solution(face_solution),
                        rhs(rhs),
                        riemann_solver(riemann_solver),
                        physics(physics) {}
        
        /**
         * @brief Overloaded operator for functor.
         * @param i_face Local face index.
         */
        KOKKOS_INLINE_FUNCTION
        void operator()(const u_int32_t i_face) const {
            rtype flux[N_CONSERVATIVE];
            rtype conservatives_l[N_CONSERVATIVE];
            rtype conservatives_r[N_CONSERVATIVE];
            rtype primitives_l[N_PRIMITIVE];
            rtype primitives_r[N_PRIMITIVE];
            rtype n_vec[N_DIM];
            rtype n_unit[N_DIM];

            if (cells_of_face(i_face, 1) == -1) {
                // Boundary face
                return;
            }

            int32_t i_cell_l = cells_of_face(i_face, 0);
            int32_t i_cell_r = cells_of_face(i_face, 1);
            FOR_I_DIM n_vec[i] = normals(i_face, i);
            unit<N_DIM>(n_vec, n_unit);

            // Get face conservatives
            for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
                conservatives_l[j] = face_solution(i_face, 0, j);
                conservatives_r[j] = face_solution(i_face, 1, j);
            }

            // Compute relevant primitive variables
            physics.compute_primitives_from_conservatives(primitives_l, conservatives_l);
            physics.compute_primitives_from_conservatives(primitives_r, conservatives_r);

            // Calculate flux
            riemann_solver.calc_flux(flux, n_unit,
                                     conservatives_l[0], primitives_l,
                                     primitives_l[2], physics.get_gamma(), primitives_l[4],
                                     conservatives_r[0], primitives_r,
                                     primitives_r[2], physics.get_gamma(), primitives_r[4]);
            
            // Add flux to RHS
            for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
                Kokkos::atomic_add(&rhs(i_cell_l, j), -face_area(i_face) * flux[j]);
                Kokkos::atomic_add(&rhs(i_cell_r, j),  face_area(i_face) * flux[j]);
            }
        }

    private:
        Kokkos::View<rtype *[N_DIM]> normals;
        Kokkos::View<rtype *> face_area;
        Kokkos::View<int32_t *[2]> cells_of_face;
        Kokkos::View<rtype **[N_CONSERVATIVE]> face_solution;
        Kokkos::View<rtype *[N_CONSERVATIVE]> rhs;
        const RiemannSolver riemann_solver;
        const Physics physics;
};

struct DivideVolumeFunctor {
    public:
        /**
         * @brief Construct a new DivideVolumeFunctor object
         * @param cell_volume Cell volume.
         * @param rhs RHS.
         */
        DivideVolumeFunctor(Kokkos::View<rtype *> cell_volume,
                            Kokkos::View<rtype *[N_CONSERVATIVE]> rhs) :
                                cell_volume(cell_volume),
                                rhs(rhs) {}
        
        /**
         * @brief Overloaded operator for functor.
         * @param i_cell Cell index.
         */
        KOKKOS_INLINE_FUNCTION
        void operator()(const u_int32_t i_cell) const {
            for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
                rhs(i_cell, j) /= cell_volume(i_cell);
            }
        }

    private:
        Kokkos::View<rtype *> cell_volume;
        Kokkos::View<rtype *[N_CONSERVATIVE]> rhs;
};

struct UpdatePrimitivesFunctor {
    public:
        /**
         * @brief Construct a new UpdatePrimitivesFunctor object
         * @param physics Physics.
         * @param conservatives Conservatives.
         * @param primitives Primitives.
         */
        UpdatePrimitivesFunctor(Physics * physics,
                                Kokkos::View<rtype *[N_CONSERVATIVE]> conservatives,
                                Kokkos::View<rtype *[N_PRIMITIVE]> primitives) :
                                    physics(physics),
                                    conservatives(conservatives),
                                    primitives(primitives) {}
        
        /**
         * @brief Overloaded operator for functor.
         * @param i_cell Cell index.
         */
        KOKKOS_INLINE_FUNCTION
        void operator()(const u_int32_t i_cell) const {
            rtype cell_conservatives[N_CONSERVATIVE];
            rtype cell_primitives[N_PRIMITIVE];
            for (u_int16_t i = 0; i < N_CONSERVATIVE; i++) {
                cell_conservatives[i] = conservatives(i_cell, i);
            }
            physics->compute_primitives_from_conservatives(cell_primitives,
                                                           cell_conservatives);
            for (u_int16_t i = 0; i < N_PRIMITIVE; i++) {
                primitives(i_cell, i) = cell_primitives[i];
            }
        }
    
    private:
        Physics * physics;
        Kokkos::View<rtype *[N_CONSERVATIVE]> conservatives;
        Kokkos::View<rtype *[N_PRIMITIVE]> primitives;
};