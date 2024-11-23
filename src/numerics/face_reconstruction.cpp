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

void FaceReconstruction::set_cell_conservatives(Kokkos::View<rtype *[N_CONSERVATIVE]> * cell_conservatives) {
    this->cell_conservatives = cell_conservatives;
}

void FaceReconstruction::set_face_conservatives(Kokkos::View<rtype *[2][N_CONSERVATIVE]> * face_conservatives) {
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
                          Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution,
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
                face_solution(i_face, 1, j) = solution(i_cell_r, j);
            }
        }

    private:
        Kokkos::View<int32_t *[2]> cells_of_face;
        Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution;
        Kokkos::View<rtype *[N_CONSERVATIVE]> solution;
};

void FirstOrder::calc_face_values(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                  Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution) {
    FirstOrderFunctor flux_functor(mesh->cells_of_face, face_solution, solution);
    Kokkos::parallel_for(mesh->n_faces, flux_functor);
}

WENO::WENO(u_int8_t poly_order) :
        poly_order(poly_order) {
    type = FaceReconstructionType::WENO;
    switch (N_DIM) {
        case 2 : {
            n_dof = (poly_order + 1) * (poly_order + 2) / 2 - 1;
            break;
        }
        case 3 : {
            n_dof = (poly_order + 1) * (poly_order + 2) * (poly_order + 3) / 6 - 1;
            break;
        }
        default : {
            throw std::runtime_error("WENO has not been implemented for N_DIM = " + std::to_string(N_DIM) + ".");
        }
    }
    max_cells_per_stencil = 2 * n_dof;
    compute_stencils();
}

WENO::~WENO() {
    // Empty
}

std::vector<u_int32_t> WENO::compute_stencil_of_cell_centered(u_int32_t i_cell) {
    // Naive Cell Based (NCB) algorithm
    // Just fill up to max_cells_per_stencil for now
    /** \todo Condition number optimization */
    bool done = false;
    std::vector<std::vector<u_int32_t>> neighbor_rings;
    neighbor_rings.push_back({i_cell});
    u_int16_t stencil_size = 1;
    while (!done) {
        // Get the next ring of neighbors
        std::vector<u_int32_t> next_ring;
        for (auto i_cell : neighbor_rings.back()) {
            std::vector<u_int32_t> neighbors;
            mesh->neighbors_of_cell(i_cell, 1, neighbors);
            for (auto neighbor : neighbors) {
                if (std::find(neighbor_rings.back().begin(), neighbor_rings.back().end(), neighbor) == neighbor_rings.back().end()) {
                    next_ring.push_back(neighbor);
                }
            }
        }

        // Check if adding the next ring would exceed the maximum size
        u_int16_t next_size = stencil_size + next_ring.size();
        if (next_size < max_cells_per_stencil) {
            // Add the ring and continue
            neighbor_rings.push_back(next_ring);
            stencil_size = next_size;
        } else if (next_size == max_cells_per_stencil) {
            // Add the ring and stop
            neighbor_rings.push_back(next_ring);
            stencil_size = next_size;
            done = true;
        } else {
            // Adding the ring would exceed the maximum size,
            // so we sort the neighbors by distance to the target cell,
            // and add the closest neighbors until we reach the maximum size
            std::vector<std::pair<u_int32_t, rtype>> distances;
            for (auto i_cell : next_ring) {
                rtype distance = 0.0;
                for (u_int8_t i_dim = 0; i_dim < N_DIM; ++i_dim) {
                    distance += std::pow(mesh->cell_coords(i_cell, i_dim) - mesh->cell_coords(i_cell, i_dim), 2);
                }
                distances.push_back(std::make_pair(i_cell, distance));
            }
            std::sort(distances.begin(),
                      distances.end(),
                      [](auto & left, auto & right) {
                          return left.second < right.second;
                      });
            for (u_int16_t i = 0; i < max_cells_per_stencil - stencil_size; ++i) {
                next_ring.push_back(distances[i].first);
            }
            done = true;
        }
    }

    // The stencil is obtained by flattening the neighbor_rings vector
    std::vector<u_int32_t> stencil;
    for (auto ring : neighbor_rings) {
        for (auto i_cell : ring) {
            stencil.push_back(i_cell);
        }
    }
    return stencil;
}

std::vector<std::vector<u_int32_t>> WENO::compute_stencils_of_cell_directional(u_int32_t i_cell) {
    std::vector<std::vector<u_int32_t>> stencils;
    return stencils;
}

void WENO::compute_stencils_of_cell(u_int32_t i_cell,
                                    std::vector<u_int32_t> & v_offsets_stencils_of_cell,
                                    std::vector<u_int32_t> & v_stencils_of_cell,
                                    std::vector<u_int32_t> & v_stencils) {
    std::vector<std::vector<u_int32_t>> stencils;

    // Compute the centered stencil
    std::vector<u_int32_t> stencil_centered = compute_stencil_of_cell_centered(i_cell);
    stencils.push_back(stencil_centered);

    // Compute the directional stencils
    std::vector<std::vector<u_int32_t>> stencils_directional = compute_stencils_of_cell_directional(i_cell);
    for (auto stencil : stencils_directional) {
        stencils.push_back(stencil);
    }

    // Update the global arrays
    for (auto stencil : stencils) {
        for (auto i_cell : stencil) {
            v_stencils.push_back(i_cell);
        }
        v_stencils_of_cell.push_back(v_stencils.size());
    }
    v_offsets_stencils_of_cell.push_back(v_stencils_of_cell.size());
}

void WENO::compute_stencils() {
    std::vector<u_int32_t> v_offsets_stencils_of_cell;
    std::vector<u_int32_t> v_stencils_of_cell;
    std::vector<u_int32_t> v_stencils;

    for (u_int32_t i_cell = 0; i_cell < mesh->n_cells; ++i_cell) {
        compute_stencils_of_cell(i_cell,
                                 v_offsets_stencils_of_cell,
                                 v_stencils_of_cell,
                                 v_stencils);
    }

    // Allocate device arrays
    offsets_stencils_of_cell = Kokkos::View<u_int32_t *>("offsets_stencils_of_cell", v_offsets_stencils_of_cell.size());
    stencils_of_cell = Kokkos::View<u_int32_t *>("stencils_of_cell", v_stencils.size());
    stencils = Kokkos::View<u_int32_t *>("stencils", v_stencils.size());

    // Set up host mirrors
    h_offsets_stencils_of_cell = Kokkos::create_mirror_view(offsets_stencils_of_cell);
    h_stencils_of_cell = Kokkos::create_mirror_view(stencils_of_cell);
    h_stencils = Kokkos::create_mirror_view(stencils);

    // Fill host mirrors
    for (u_int32_t i = 0; i < v_offsets_stencils_of_cell.size(); ++i) {
        h_offsets_stencils_of_cell(i) = v_offsets_stencils_of_cell[i];
    }
    for (u_int32_t i = 0; i < v_stencils_of_cell.size(); ++i) {
        h_stencils_of_cell(i) = v_stencils_of_cell[i];
    }
    for (u_int32_t i = 0; i < v_stencils.size(); ++i) {
        h_stencils(i) = v_stencils[i];
    }

    // Copy from host to device
    Kokkos::deep_copy(offsets_stencils_of_cell, h_offsets_stencils_of_cell);
    Kokkos::deep_copy(stencils_of_cell, h_stencils_of_cell);
    Kokkos::deep_copy(stencils, h_stencils);
}

struct WENOFunctor {
    public:
        /**
         * @brief Construct a new WENOFunctor object
         * @param cells_of_face Cells of face.
         * @param face_normals Face normals.
         * @param face_solution Face solution.
         * @param solution Cell solution.
         */
        WENOFunctor(Kokkos::View<int32_t *[2]> cells_of_face,
                    Kokkos::View<rtype *[2]> face_normals,
                    Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution,
                    Kokkos::View<rtype *[N_CONSERVATIVE]> solution) :
                        cells_of_face(cells_of_face),
                        face_normals(face_normals),
                        face_solution(face_solution),
                        solution(solution) {}
        
        /**
         * @brief Overloaded operator for WENO face reconstruction.
         * @param i_face Face index.
         */
        KOKKOS_INLINE_FUNCTION
        void operator()(const u_int32_t i_face) const {
            // Empty
        }
    
    private:
        Kokkos::View<int32_t *[2]> cells_of_face;
        Kokkos::View<rtype *[2]> face_normals;
        Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution;
        Kokkos::View<rtype *[N_CONSERVATIVE]> solution;
};

void WENO::calc_face_values(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution) {
    WENOFunctor flux_functor(mesh->cells_of_face,
                             mesh->face_normals,
                             face_solution,
                             solution);
    Kokkos::parallel_for(mesh->n_faces, flux_functor);
}
