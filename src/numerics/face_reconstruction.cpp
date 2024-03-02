/**
 * @file face_reconstruction.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Face reconstruction class implementation.
 * @version 0.1
 * @date 2023-12-24
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#include "face_reconstruction.h"

#include <iostream>

#include "common_math.h"

FaceReconstruction::FaceReconstruction() {
    // Empty
}

FaceReconstruction::~FaceReconstruction() {
    std::cout << "Destroying face reconstruction: " << FACE_RECONSTRUCTION_NAMES.at(type) << std::endl;
}

void FaceReconstruction::init() {
    print();
}

void FaceReconstruction::print() const {
    std::cout << LOG_SEPARATOR << std::endl;
    std::cout << "Face reconstruction: " << FACE_RECONSTRUCTION_NAMES.at(type) << std::endl;
    std::cout << LOG_SEPARATOR << std::endl;
}

void FaceReconstruction::set_cell_conservatives(view_2d * cell_conservatives) {
    this->cell_conservatives = cell_conservatives;
}

void FaceReconstruction::set_face_conservatives(view_3d * face_conservatives) {
    this->face_conservatives = face_conservatives;
}

void FaceReconstruction::set_mesh(std::shared_ptr<Mesh> mesh) {
    this->mesh = mesh;
}

FirstOrder::FirstOrder() {
    type = FaceReconstructionType::FirstOrder;
}

FirstOrder::~FirstOrder() {
    // Empty
}

struct FirstOrderFunctor {
    public:
        /**
         * @brief Construct a new FirstOrderFunctor object
         * @param cells_of_face Cells of face.
         * @param face_solution Face solution.
         * @param solution Cell solution.
         */
        FirstOrderFunctor(Kokkos::View<int32_t *[2]> cells_of_face,
                          Kokkos::View<rtype **[N_CONSERVATIVE]> face_solution,
                          Kokkos::View<rtype *[N_CONSERVATIVE]> solution) :
                              cells_of_face(cells_of_face),
                              face_solution(face_solution),
                              solution(solution) {}

        /**
         * @brief Overloaded operator for first order face reconstruction.
         * @param i_face Face index.
         */
        KOKKOS_INLINE_FUNCTION
        void operator()(const u_int32_t i_face) const {
            int32_t i_cell_l = cells_of_face(i_face, 0);
            int32_t i_cell_r = cells_of_face(i_face, 1);

            for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
                face_solution(i_face, 0, j) = solution(i_cell_l, j);
                if (i_cell_r != -1) {
                    face_solution(i_face, 1, j) = solution(i_cell_r, j);
                }
            }
        }

    private:
        Kokkos::View<int32_t *[2]> cells_of_face;
        Kokkos::View<rtype **[N_CONSERVATIVE]> face_solution;
        Kokkos::View<rtype *[N_CONSERVATIVE]> solution;
};

void FirstOrder::calc_face_values(view_2d * solution,
                                  view_3d * face_solution) {
    FirstOrderFunctor flux_functor(mesh->cells_of_face, *face_solution, *solution);
    Kokkos::parallel_for(mesh->n_faces(), flux_functor);
}

WENO3_JS::WENO3_JS() {
    type = FaceReconstructionType::WENO3_JS;
}

WENO3_JS::~WENO3_JS() {
    // Empty
}

struct WENO3_JSFunctor {
    public:
        /**
         * @brief Construct a new WENO3_JSFunctor object
         * @param cells_of_face Cells of face.
         * @param face_normals Face normals.
         * @param n_cells_x Number of cells in the x-direction.
         * @param n_cells_y Number of cells in the y-direction.
         * @param face_solution Face solution.
         * @param solution Cell solution.
         */
        WENO3_JSFunctor(Kokkos::View<int32_t *[2]> cells_of_face,
                        Kokkos::View<rtype *[2]> face_normals,
                        u_int32_t n_cells_x,
                        u_int32_t n_cells_y,
                        Kokkos::View<rtype **[N_CONSERVATIVE]> face_solution,
                        Kokkos::View<rtype *[N_CONSERVATIVE]> solution) :
                            cells_of_face(cells_of_face),
                            face_normals(face_normals),
                            n_cells_x(n_cells_x),
                            n_cells_y(n_cells_y),
                            face_solution(face_solution),
                            solution(solution) {}
        
        /**
         * @brief Overloaded operator for WENO3_JS face reconstruction.
         * @param i_face Face index.
         */
        KOKKOS_INLINE_FUNCTION
        void operator()(const u_int32_t i_face) const {
            /** \todo This is super hacky, only valid for a 2d uniform cartesian mesh */
            int32_t i_cell_l = cells_of_face(i_face, 0);
            int32_t i_cell_r = cells_of_face(i_face, 1);

            bool is_x_face = false;
            rtype n_vec[N_DIM];
            rtype n_unit[N_DIM];
            FOR_I_DIM n_vec[i] = face_normals(i_face, i);
            unit<N_DIM>(n_vec, n_unit);
            if (Kokkos::fabs(n_unit[0]) > 0.5) {
                is_x_face = true;
            }

            u_int32_t ic = i_cell_l / n_cells_y;
            u_int32_t jc = i_cell_l % n_cells_y;

            // Handle boundary conditions
            bool is_boundary = false;
            if (is_x_face) {
                is_boundary = (ic < 1 || ic > n_cells_x - 2);
            } else {
                is_boundary = (jc < 1 || jc > n_cells_y - 2);
            }
            if (is_boundary) {
                for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
                    face_solution(i_face, 0, j) = solution(i_cell_l, j);
                    face_solution(i_face, 1, j) = solution(i_cell_r, j);
                }
                return;
            }

            int32_t i_cell_im1, i_cell_i, i_cell_ip1;

            // -------------------------------------------------
            // Left side of face

            if (is_x_face) {
                i_cell_im1 = i_cell_l - n_cells_y;
                i_cell_i   = i_cell_l;
                i_cell_ip1 = i_cell_l + n_cells_y;
            } else {
                i_cell_im1 = i_cell_l - 1;
                i_cell_i   = i_cell_l;
                i_cell_ip1 = i_cell_l + 1;
            }

            for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
                // Smoothness indicators
                rtype beta_0 = Kokkos::pow(solution(i_cell_i,   j) - solution(i_cell_im1, j), 2.0);
                rtype beta_1 = Kokkos::pow(solution(i_cell_ip1, j) - solution(i_cell_i,   j), 2.0);

                rtype epsilon = 1e-6;
                rtype alpha_0 = (1.0 / 3.0) / Kokkos::pow(epsilon + beta_0, 2.0);
                rtype alpha_1 = (2.0 / 3.0) / Kokkos::pow(epsilon + beta_1, 2.0);

                rtype one_alpha = 1 / (alpha_0 + alpha_1);
                rtype w_0 = alpha_0 * one_alpha;
                rtype w_1 = alpha_1 * one_alpha;

                rtype p_0 = -0.5 * solution(i_cell_im1, j) + 1.5 * solution(i_cell_i,   j);
                rtype p_1 =  0.5 * solution(i_cell_i,   j) + 0.5 * solution(i_cell_ip1, j);

                face_solution(i_face, 0, j) = w_0 * p_0 + w_1 * p_1;
            }

            // -------------------------------------------------
            // Right side of face

            if (is_x_face) {
                i_cell_im1 = i_cell_l;
                i_cell_i   = i_cell_l + n_cells_y;
                i_cell_ip1 = i_cell_l + 2 * n_cells_y;
            } else {
                i_cell_im1 = i_cell_l;
                i_cell_i   = i_cell_l + 1;
                i_cell_ip1 = i_cell_l + 2;
            }

            for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
                // Smoothness indicators
                rtype beta_0 = Kokkos::pow(solution(i_cell_i,   j) - solution(i_cell_im1, j), 2.0);
                rtype beta_1 = Kokkos::pow(solution(i_cell_ip1, j) - solution(i_cell_i,   j), 2.0);

                rtype epsilon = 1e-6;
                rtype alpha_0 = (2.0 / 3.0) / Kokkos::pow(epsilon + beta_0, 2.0);
                rtype alpha_1 = (1.0 / 3.0) / Kokkos::pow(epsilon + beta_1, 2.0);

                rtype one_alpha = 1 / (alpha_0 + alpha_1);
                rtype w_0 = alpha_0 * one_alpha;
                rtype w_1 = alpha_1 * one_alpha;

                rtype p_0 = 0.5 * solution(i_cell_im1, j) + 0.5 * solution(i_cell_i,   j);
                rtype p_1 = 1.5 * solution(i_cell_i,   j) - 0.5 * solution(i_cell_ip1, j);

                face_solution(i_face, 1, j) = w_0 * p_0 + w_1 * p_1;
            }
        }
    
    private:
        Kokkos::View<int32_t *[2]> cells_of_face;
        Kokkos::View<rtype *[2]> face_normals;
        u_int32_t n_cells_x;
        u_int32_t n_cells_y;
        Kokkos::View<rtype **[N_CONSERVATIVE]> face_solution;
        Kokkos::View<rtype *[N_CONSERVATIVE]> solution;
};

void WENO3_JS::calc_face_values(view_2d * solution,
                                view_3d * face_solution) {
    WENO3_JSFunctor flux_functor(mesh->cells_of_face,
                                 mesh->face_normals,
                                 mesh->n_cells_x(),
                                 mesh->n_cells_y(),
                                 *face_solution,
                                 *solution);
    Kokkos::parallel_for(mesh->n_faces(), flux_functor);
}

WENO5_JS::WENO5_JS() {
    type = FaceReconstructionType::WENO5_JS;
}

WENO5_JS::~WENO5_JS() {
    // Empty
}

struct WENO5_JSFunctor {
    public:
        /**
         * @brief Construct a new WENO5_JSFunctor object
         * @param cells_of_face Cells of face.
         * @param face_normals Face normals.
         * @param n_cells_x Number of cells in the x-direction.
         * @param n_cells_y Number of cells in the y-direction.
         * @param face_solution Face solution.
         * @param solution Cell solution.
         */
        WENO5_JSFunctor(Kokkos::View<int32_t *[2]> cells_of_face,
                        Kokkos::View<rtype *[2]> face_normals,
                        u_int32_t n_cells_x,
                        u_int32_t n_cells_y,
                        Kokkos::View<rtype **[N_CONSERVATIVE]> face_solution,
                        Kokkos::View<rtype *[N_CONSERVATIVE]> solution) :
                            cells_of_face(cells_of_face),
                            face_normals(face_normals),
                            n_cells_x(n_cells_x),
                            n_cells_y(n_cells_y),
                            face_solution(face_solution),
                            solution(solution) {}
        
        /**
         * @brief Overloaded operator for WENO5_JS face reconstruction.
         * @param i_face Face index.
         */
        KOKKOS_INLINE_FUNCTION
        void operator()(const u_int32_t i_face) const {
            /** \todo This is super hacky, only valid for a 2d uniform cartesian mesh */
            int32_t i_cell_l = cells_of_face(i_face, 0);
            int32_t i_cell_r = cells_of_face(i_face, 1);

            bool is_x_face = false;
            rtype n_vec[N_DIM];
            rtype n_unit[N_DIM];
            FOR_I_DIM n_vec[i] = face_normals(i_face, i);
            unit<N_DIM>(n_vec, n_unit);
            if (Kokkos::fabs(n_unit[0]) > 0.5) {
                is_x_face = true;
            }

            u_int32_t ic = i_cell_l / n_cells_y;
            u_int32_t jc = i_cell_l % n_cells_y;

            // Handle boundary conditions
            bool is_boundary = false;
            if (is_x_face) {
                is_boundary = (ic < 2 || ic > n_cells_x - 3);
            } else {
                is_boundary = (jc < 2 || jc > n_cells_y - 3);
            }
            if (is_boundary) {
                for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
                    face_solution(i_face, 0, j) = solution(i_cell_l, j);
                    face_solution(i_face, 1, j) = solution(i_cell_r, j);
                }
                return;
            }

            int32_t i_cell_im2, i_cell_im1, i_cell_i, i_cell_ip1, i_cell_ip2;

            // -------------------------------------------------
            // Left side of face

            if (is_x_face) {
                i_cell_im2 = i_cell_l - 2 * n_cells_y;
                i_cell_im1 = i_cell_l -     n_cells_y;
                i_cell_i   = i_cell_l                        ;
                i_cell_ip1 = i_cell_l +     n_cells_y;
                i_cell_ip2 = i_cell_l + 2 * n_cells_y;
            } else {
                i_cell_im2 = i_cell_l - 2;
                i_cell_im1 = i_cell_l - 1;
                i_cell_i   = i_cell_l    ;
                i_cell_ip1 = i_cell_l + 1;
                i_cell_ip2 = i_cell_l + 2;
            }

            for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
                // Smoothness indicators
                rtype beta_0 = 13.0 / 12.0 * pow(        solution(i_cell_im2, j)
                                                 - 2.0 * solution(i_cell_im1, j)
                                                 +       solution(i_cell_i,   j), 2.0) +
                                      0.25 * pow(        solution(i_cell_im2, j)
                                                 - 4.0 * solution(i_cell_im1, j)
                                                 + 3.0 * solution(i_cell_i,   j), 2.0);
                rtype beta_1 = 13.0 / 12.0 * pow(        solution(i_cell_im1, j)
                                                 - 2.0 * solution(i_cell_i,   j)
                                                 +       solution(i_cell_ip1, j), 2.0) +
                                      0.25 * pow(        solution(i_cell_im1, j)
                                                 -       solution(i_cell_ip1, j), 2.0);
                rtype beta_2 = 13.0 / 12.0 * pow(        solution(i_cell_i,   j)
                                                 - 2.0 * solution(i_cell_ip1, j)
                                                 +       solution(i_cell_ip2, j), 2.0) +
                                      0.25 * pow(  3.0 * solution(i_cell_i,   j)
                                                 - 4.0 * solution(i_cell_ip1, j)
                                                 +       solution(i_cell_ip2, j), 2.0);

                rtype epsilon = 1e-6;
                rtype alpha_0 = 0.1 / pow(epsilon + beta_0, 2.0);
                rtype alpha_1 = 0.6 / pow(epsilon + beta_1, 2.0);
                rtype alpha_2 = 0.3 / pow(epsilon + beta_2, 2.0);

                rtype one_alpha = 1 / (alpha_0 + alpha_1 + alpha_2);
                rtype w_0 = alpha_0 * one_alpha;
                rtype w_1 = alpha_1 * one_alpha;
                rtype w_2 = alpha_2 * one_alpha;

                rtype p_0 =   ( 1.0 / 3.0) * solution(i_cell_im2, j)
                            - ( 7.0 / 6.0) * solution(i_cell_im1, j)
                            + (11.0 / 6.0) * solution(i_cell_i,   j);
                rtype p_1 = - ( 1.0 / 6.0) * solution(i_cell_im1, j)
                            + ( 5.0 / 6.0) * solution(i_cell_i,   j)
                            + ( 1.0 / 3.0) * solution(i_cell_ip1, j);
                rtype p_2 =   ( 1.0 / 3.0) * solution(i_cell_i,   j)
                            + ( 5.0 / 6.0) * solution(i_cell_ip1, j)
                            - ( 1.0 / 6.0) * solution(i_cell_ip2, j);

                face_solution(i_face, 0, j) = w_0 * p_0 + w_1 * p_1 + w_2 * p_2;
            }

            // -------------------------------------------------
            // Right side of face

            if (is_x_face) {
                i_cell_im2 = i_cell_l -     n_cells_y;
                i_cell_im1 = i_cell_l                        ;
                i_cell_i   = i_cell_l +     n_cells_y;
                i_cell_ip1 = i_cell_l + 2 * n_cells_y;
                i_cell_ip2 = i_cell_l + 3 * n_cells_y;
            } else {
                i_cell_im2 = i_cell_l - 1;
                i_cell_im1 = i_cell_l    ;
                i_cell_i   = i_cell_l + 1;
                i_cell_ip1 = i_cell_l + 2;
                i_cell_ip2 = i_cell_l + 3;
            }

            for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
                // Smoothness indicators
                rtype beta_0 = 13.0 / 12.0 * pow(        solution(i_cell_im2, j)
                                                 - 2.0 * solution(i_cell_im1, j)
                                                 +       solution(i_cell_i,   j), 2.0) +
                                      0.25 * pow(        solution(i_cell_im2, j)
                                                 - 4.0 * solution(i_cell_im1, j)
                                                 + 3.0 * solution(i_cell_i,   j), 2.0);
                rtype beta_1 = 13.0 / 12.0 * pow(        solution(i_cell_im1, j)
                                                 - 2.0 * solution(i_cell_i,   j)
                                                 +       solution(i_cell_ip1, j), 2.0) +
                                      0.25 * pow(        solution(i_cell_im1, j)
                                                 -       solution(i_cell_ip1, j), 2.0);
                rtype beta_2 = 13.0 / 12.0 * pow(        solution(i_cell_i,   j)
                                                 - 2.0 * solution(i_cell_ip1, j)
                                                 +       solution(i_cell_ip2, j), 2.0) +
                                      0.25 * pow(  3.0 * solution(i_cell_i,   j)
                                                 - 4.0 * solution(i_cell_ip1, j)
                                                 +       solution(i_cell_ip2, j), 2.0);

                rtype epsilon = 1e-6;
                rtype alpha_0 = 0.3 / pow(epsilon + beta_0, 2.0);
                rtype alpha_1 = 0.6 / pow(epsilon + beta_1, 2.0);
                rtype alpha_2 = 0.1 / pow(epsilon + beta_2, 2.0);

                rtype one_alpha = 1 / (alpha_0 + alpha_1 + alpha_2);
                rtype w_0 = alpha_0 * one_alpha;
                rtype w_1 = alpha_1 * one_alpha;
                rtype w_2 = alpha_2 * one_alpha;

                rtype p_0 = - ( 1.0 / 6.0) * solution(i_cell_im1, j)
                            + ( 5.0 / 6.0) * solution(i_cell_i,   j)
                            + ( 1.0 / 3.0) * solution(i_cell_ip1, j);
                rtype p_1 =   ( 1.0 / 3.0) * solution(i_cell_i,   j)
                            + ( 5.0 / 6.0) * solution(i_cell_ip1, j)
                            - ( 1.0 / 6.0) * solution(i_cell_ip2, j);
                rtype p_2 =   (11.0 / 6.0) * solution(i_cell_i,   j)
                            - ( 7.0 / 6.0) * solution(i_cell_ip1, j)
                            + ( 1.0 / 3.0) * solution(i_cell_ip2, j);

                face_solution(i_face, 1, j) = w_0 * p_0 + w_1 * p_1 + w_2 * p_2;
            }
        }
    
    private:
        Kokkos::View<int32_t *[2]> cells_of_face;
        Kokkos::View<rtype *[2]> face_normals;
        u_int32_t n_cells_x;
        u_int32_t n_cells_y;
        Kokkos::View<rtype **[N_CONSERVATIVE]> face_solution;
        Kokkos::View<rtype *[N_CONSERVATIVE]> solution;
};

void WENO5_JS::calc_face_values(view_2d * solution,
                                view_3d * face_solution) {
    WENO5_JSFunctor flux_functor(mesh->cells_of_face,
                                 mesh->face_normals,
                                 mesh->n_cells_x(),
                                 mesh->n_cells_y(),
                                 *face_solution,
                                 *solution);
    Kokkos::parallel_for(mesh->n_faces(), flux_functor);
}