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

template <typename T_physics, typename T_riemann_solver>
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
                    const T_riemann_solver riemann_solver,
                    const T_physics physics) :
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
        const T_riemann_solver riemann_solver;
        const T_physics physics;
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

template <typename T>
struct UpdatePrimitivesFunctor {
    public:
        /**
         * @brief Construct a new UpdatePrimitivesFunctor object
         * @param physics Physics.
         * @param conservatives Conservatives.
         * @param primitives Primitives.
         */
        UpdatePrimitivesFunctor(const T physics,
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
            physics.compute_primitives_from_conservatives(cell_primitives,
                                                          cell_conservatives);
            for (u_int16_t i = 0; i < N_PRIMITIVE; i++) {
                primitives(i_cell, i) = cell_primitives[i];
            }
        }
    
    private:
        const T physics;
        Kokkos::View<rtype *[N_CONSERVATIVE]> conservatives;
        Kokkos::View<rtype *[N_PRIMITIVE]> primitives;
};

template <typename T>
struct SpectralRadiusFunctor {
    public:
        /**
         * @brief Construct a new SpectralRadiusFunctor object
         * @param offsets_faces_of_cell Offsets of faces of cell.
         * @param cells_of_face Cells of face.
         * @param face_normals Face normals.
         * @param face_coords Face coordinates.
         * @param cell_coords Cell coordinates.
         * @param physics Physics.
         * @param conservatives Conservatives.
         * @param primitives Primitives.
         * @param cfl_local Local CFL.
         */
        SpectralRadiusFunctor(Kokkos::View<u_int32_t *> offsets_faces_of_cell,
                              Kokkos::View<int32_t *[2]> cells_of_face,
                              Kokkos::View<rtype *[N_DIM]> face_normals,
                              Kokkos::View<rtype *[N_DIM]> face_coords,
                              Kokkos::View<rtype *[N_DIM]> cell_coords,
                              const T physics,
                              Kokkos::View<rtype *[N_CONSERVATIVE]> conservatives,
                              Kokkos::View<rtype *[N_PRIMITIVE]> primitives,
                              Kokkos::View<rtype *> cfl_local) :
                                  offsets_faces_of_cell(offsets_faces_of_cell),
                                  cells_of_face(cells_of_face),
                                  face_normals(face_normals),
                                  face_coords(face_coords),
                                  cell_coords(cell_coords),
                                  physics(physics),
                                  conservatives(conservatives),
                                  primitives(primitives),
                                  cfl_local(cfl_local) {}
        
        /**
         * @brief Overloaded operator for functor.
         * @param i_cell Cell index.
         * @param max_spectral_radius_i Max spectral radius for cell i_cell.
         */
        KOKKOS_INLINE_FUNCTION
        void operator()(const u_int32_t i_cell, rtype & max_spectral_radius_i) const {
            rtype spectral_radius_convective;
            rtype spectral_radius_acoustic;
            // rtype spectral_radius_viscous;
            // rtype spectral_radius_heat;
            rtype spectral_radius_overall;
            rtype rho_l, rho_r, p_l, p_r, sos_l, sos_r, sos_f;
            rtype s[N_DIM], u_l[N_DIM], u_r[N_DIM], u_f[N_DIM];
            rtype dx_n, u_n;
            rtype geom_factor;
            rtype n_vec[N_DIM];
            rtype n_unit[N_DIM];

            spectral_radius_convective = 0.0;
            spectral_radius_acoustic = 0.0;
            // spectral_radius_viscous = 0.0;
            // spectral_radius_heat = 0.0;

            u_int32_t n_faces = offsets_faces_of_cell(i_cell + 1) - offsets_faces_of_cell(i_cell);

            for (u_int32_t i_face = 0; i_face < n_faces; i_face++) {
                int32_t i_cell_l = cells_of_face(i_face, 0);
                int32_t i_cell_r = cells_of_face(i_face, 1);
                FOR_I_DIM n_vec[i] = face_normals(i_face, i);
                unit<N_DIM>(n_vec, n_unit);

                if (i_cell_r == -1) {
                    // Boundary face, hack
                    s[0] = 2.0 * (face_coords(i_face, 0) - cell_coords(i_cell_l, 0));
                    s[1] = 2.0 * (face_coords(i_face, 1) - cell_coords(i_cell_l, 1));
                    i_cell_r = i_cell_l;
                } else {
                    s[0] = cell_coords(i_cell_r, 0) - cell_coords(i_cell_l, 0);
                    s[1] = cell_coords(i_cell_r, 1) - cell_coords(i_cell_l, 1);
                }

                dx_n = Kokkos::fabs(dot<N_DIM>(s, n_unit));

                rho_l = conservatives(i_cell_l, 0);
                rho_r = conservatives(i_cell_r, 0);
                u_l[0] = primitives(i_cell_l, 0);
                u_l[1] = primitives(i_cell_l, 1);
                u_r[0] = primitives(i_cell_r, 0);
                u_r[1] = primitives(i_cell_r, 1);
                p_l = primitives(i_cell_l, 2);
                p_r = primitives(i_cell_r, 2);
                sos_l = physics.get_sound_speed_from_pressure_density(p_l, rho_l);
                sos_r = physics.get_sound_speed_from_pressure_density(p_r, rho_r);
                sos_f = 0.5 * (sos_l + sos_r);

                u_f[0] = 0.5 * (u_l[0] + u_r[0]);
                u_f[1] = 0.5 * (u_l[1] + u_r[1]);
                u_n = Kokkos::fabs(dot<N_DIM>(u_f, n_unit));

                spectral_radius_convective += u_n / dx_n;
                spectral_radius_acoustic += Kokkos::pow(sos_f / dx_n, 2.0);
            }

            geom_factor = 3.0 / n_faces;
            spectral_radius_convective *= 1.37 * geom_factor;
            spectral_radius_acoustic = 1.37 * Kokkos::sqrt(geom_factor * spectral_radius_acoustic);

            /** \todo Implement viscous and heat spectral radii */
            spectral_radius_overall = spectral_radius_convective + spectral_radius_acoustic;

            // Update max spectral radius
            max_spectral_radius_i = Kokkos::max(max_spectral_radius_i,
                                                spectral_radius_overall);

            // Store spectral radius in cfl_local, will be used to compute local cfl
            cfl_local(i_cell) = spectral_radius_overall;
        }
    
    private:
        Kokkos::View<u_int32_t *> offsets_faces_of_cell;
        Kokkos::View<int32_t *[2]> cells_of_face;
        Kokkos::View<rtype *[N_DIM]> face_normals;
        Kokkos::View<rtype *[N_DIM]> face_coords;
        Kokkos::View<rtype *[N_DIM]> cell_coords;
        const T physics;
        Kokkos::View<rtype *[N_CONSERVATIVE]> conservatives;
        Kokkos::View<rtype *[N_PRIMITIVE]> primitives;
        Kokkos::View<rtype *> cfl_local;
};