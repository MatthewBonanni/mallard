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

#include <Kokkos_Core.hpp>
#include <toml.hpp>

#include "common.h"
#include "basis.h"
#include "quadrature.h"

FaceReconstruction::FaceReconstruction() {
    // Empty
}

FaceReconstruction::~FaceReconstruction() {
    std::cout << "Destroying face reconstruction: " << FACE_RECONSTRUCTION_NAMES.at(type) << std::endl;
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

void FirstOrder::init(const toml::value & input) {
    (void)(input);
    print();
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

            FOR_I_CONSERVATIVE {
                face_solution(i_face, 0, i) = solution(i_cell_l, i);
                face_solution(i_face, 1, i) = solution(i_cell_r, i);
            }
        }

    private:
        Kokkos::View<int32_t *[2]> cells_of_face;
        Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution;
        Kokkos::View<rtype *[N_CONSERVATIVE]> solution;
};

void FirstOrder::calc_face_values(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                  Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution) {
    FirstOrderFunctor recon_functor(mesh->cells_of_face, face_solution, solution);
    Kokkos::parallel_for(mesh->n_faces, recon_functor);
}

TENO::TENO() {
    type = FaceReconstructionType::TENO;
}

TENO::~TENO() {
    // Empty
}

void TENO::init(const toml::value & input) {
    std::string basis_type_str = toml::find_or<std::string>(input, "basis_type", "monomial");
    poly_order = toml::find<u_int8_t>(input, "basis_order");
    max_stencil_size_factor = toml::find_or<rtype>(input, "max_stencil_size_factor", 2.0);
    std::string quadrature_type_str = toml::find_or<std::string>(input, "quadrature_type", "triangle_dunavant");
    u_int8_t quadrature_order = toml::find_or<u_int8_t>(input, "quadrature_order", 2 * poly_order);

    if (BASIS_TYPES.find(basis_type_str) == BASIS_TYPES.end()) {
        throw std::runtime_error("Unknown basis type: " + basis_type_str + ".");
    } else {
        basis_type = BASIS_TYPES.at(basis_type_str);
    }

    QuadratureType quadrature_type;
    if (QUADRATURE_TYPES.find(quadrature_type_str) == QUADRATURE_TYPES.end()) {
        throw std::runtime_error("Unknown quadrature type: " + quadrature_type_str + ".");
    } else {
        quadrature_type = QUADRATURE_TYPES.at(quadrature_type_str);
    }

    switch (quadrature_type) {
        case QuadratureType::TriangleCentroid:
            quadrature = TriangleCentroid();
            break;
        case QuadratureType::TriangleDunavant:
            quadrature = TriangleDunavant(quadrature_order);
            break;
        default:
            throw std::runtime_error("Unknown quadrature type.");
    }

    calc_max_stencil_size();
    calc_polynomial_indices();
    compute_stencils();
    compute_reconstruction_matrices();
    print();
}

void TENO::print() const {
    std::cout << LOG_SEPARATOR << std::endl;
    std::cout << "Face reconstruction: " << FACE_RECONSTRUCTION_NAMES.at(type) << std::endl;
    std::cout << "> Polynomial order: " << (u_int16_t)poly_order << std::endl;
    std::cout << "> Quadrature order: " << (u_int16_t)quadrature.order << std::endl;
    std::cout << "> Maximum stencil size factor: " << max_stencil_size_factor << std::endl;
    std::cout << "> Maximum cells per stencil: " << max_cells_per_stencil << std::endl;
    std::cout << "> Number of degrees of freedom: " << n_dof << std::endl;
    std::cout << LOG_SEPARATOR << std::endl;
}

void TENO::calc_max_stencil_size() {
    // Nd = (prod(r + m) from m=1 to N_DIM) / N_DIM!
    n_dof = 1;
    u_int16_t denom = 1;
    for (u_int8_t i = 1; i <= N_DIM; ++i) {
        denom *= i;
        n_dof *= poly_order + i;
    }
    n_dof /= denom;
    max_cells_per_stencil = max_stencil_size_factor * n_dof;
}

void TENO::calc_polynomial_indices() {
    poly_indices = Kokkos::View<u_int32_t *[N_DIM]>("poly_indices", n_dof);
    h_poly_indices = Kokkos::create_mirror_view(poly_indices);

    // Use a lexicographic ordering to generate the polynomial indices
    int idx = 0;
    std::vector<int> indices(N_DIM);
    for (int i_poly = 0; i_poly <= poly_order; i_poly++) {
        // First set of indices has i_dim set to i_poly
        std::fill(indices.begin(), indices.end(), 0);
        indices[0] = i_poly;

        // Write to the view
        for (int i_dim = 0; i_dim < N_DIM; ++i_dim) {
            h_poly_indices(idx, i_dim) = indices[i_dim];
        }
        ++idx;

        // Propagate 1 from each dimension's index to the right,
        // repeating until the last dimension's index is equal to i_poly
        while (indices[N_DIM-1] < i_poly) {
            for (int i_dim = 0; i_dim < N_DIM-1; ++i_dim) {
                if (indices[i_dim] > 0) {
                    indices[i_dim] -= 1;
                    indices[i_dim + 1] += 1;
                }
            }

            // Write to the view
            for (int i_dim = 0; i_dim < N_DIM; ++i_dim) {
                h_poly_indices(idx, i_dim) = indices[i_dim];
            }
            ++idx;
        }
    }

    Kokkos::deep_copy(poly_indices, h_poly_indices);
}

std::vector<u_int32_t> TENO::compute_stencil_of_cell_centered(u_int32_t i_cell) {
    // Naive Cell Based (NCB) algorithm (Tsoutsanis 2023)
    bool done = false;
    std::vector<std::vector<u_int32_t>> neighbor_rings;
    neighbor_rings.push_back({i_cell});
    u_int16_t stencil_size = 1;
    while (!done) {
        // Get the next ring of neighbors
        std::vector<u_int32_t> next_ring;
        for (auto i_neighbor_cell : neighbor_rings.back()) {
            std::vector<u_int32_t> neighbors;
            mesh->h_neighbors_of_cell(i_neighbor_cell, 1, neighbors);
            for (auto neighbor : neighbors) {
                // Check if this neighbor is already in a previous ring or the next ring
                bool in_old_ring = false;
                for (auto ring : neighbor_rings) {
                    if (std::find(ring.begin(), ring.end(), neighbor) != ring.end()) {
                        in_old_ring = true;
                        break;
                    }
                }
                bool in_next_ring = (std::find(next_ring.begin(),
                                               next_ring.end(),
                                               neighbor) != next_ring.end());
                if (!in_old_ring && !in_next_ring) {
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
            // so we sort the ring by distance to the target cell,
            // and add the closest neighbors until we reach the maximum size
            std::vector<std::pair<u_int32_t, rtype>> distances;
            for (auto i_neighbor_cell : next_ring) {
                rtype distance = 0.0;
                FOR_I_DIM {
                    distance += std::pow(mesh->cell_coords(i_neighbor_cell, i) -
                                         mesh->cell_coords(i_cell,          i), 2);
                }
                distances.push_back(std::make_pair(i_cell, distance));
            }
            std::sort(distances.begin(),
                      distances.end(),
                      [](auto & left, auto & right) {
                          return left.second < right.second;
                      });
            // Drop the farthest neighbors
            next_ring.resize(max_cells_per_stencil - stencil_size);
            neighbor_rings.push_back(next_ring);
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

std::vector<std::vector<u_int32_t>> TENO::compute_stencils_of_cell_directional(u_int32_t i_cell) {
    // Type 4 algorithm (Tsoutsanis 2023)
    u_int8_t n_stencils = mesh->n_faces_of_cell(i_cell);
    std::vector<std::vector<u_int32_t>> stencils(n_stencils);
    std::vector<bool> stencil_grew(n_stencils);
    bool all_done = false;

    // First, compute the transformation matrices for each subvolume of the target cell
    std::vector<std::vector<rtype>> J_sub_matrices;
    std::vector<std::vector<rtype>> J_sub_inverses;
    for (u_int8_t i_stencil = 0; i_stencil < n_stencils; ++i_stencil) {
        u_int32_t i_face = mesh->h_face_of_cell(i_cell, i_stencil); // Take i_face_local = i_stencil
        u_int32_t i_node_0 = mesh->h_node_of_face(i_face, 0);
        u_int32_t i_node_1 = mesh->h_node_of_face(i_face, 1);

        std::vector<rtype> v0(N_DIM);
        std::vector<rtype> v1(N_DIM);
        std::vector<rtype> v2(N_DIM);

        FOR_I_DIM {
            v0[i] = mesh->h_cell_coords(i_cell,   i);
            v1[i] = mesh->h_node_coords(i_node_0, i);
            v2[i] = mesh->h_node_coords(i_node_1, i);
        }

        std::vector<rtype> J(4);
        std::vector<rtype> J_inv(4);
        triangle_J(v0.data(), v1.data(), v2.data(), J.data());
        invert_matrix<2>(J.data(), J_inv.data());
        J_sub_matrices.push_back(J);
        J_sub_inverses.push_back(J_inv);
    }

    // Add the target cell to the stencils
    for (auto & stencil : stencils) {
        stencil.push_back(i_cell);
    }

    std::vector<std::vector<u_int32_t>> neighbor_rings;
    neighbor_rings.push_back({i_cell});
    while (!all_done) {
        // Get the next ring of neighbors
        std::vector<u_int32_t> next_ring;
        for (auto i_neighbor_cell : neighbor_rings.back()) {
            std::vector<u_int32_t> neighbors;
            mesh->h_neighbors_of_cell(i_neighbor_cell, 1, neighbors);
            for (auto neighbor : neighbors) {
                // Check if this neighbor is already in a previous ring or the next ring
                bool in_old_ring = false;
                for (auto ring : neighbor_rings) {
                    if (std::find(ring.begin(), ring.end(), neighbor) != ring.end()) {
                        in_old_ring = true;
                        break;
                    }
                }
                bool in_next_ring = (std::find(next_ring.begin(),
                                               next_ring.end(),
                                               neighbor) != next_ring.end());
                if (!in_old_ring && !in_next_ring) {
                    next_ring.push_back(neighbor);
                }
            }
            // Sort the ring by distance to the target cell
            std::vector<std::pair<u_int32_t, rtype>> distances;
            for (auto i_neighbor_cell : next_ring) {
                rtype distance = 0.0;
                FOR_I_DIM {
                    distance += std::pow(mesh->cell_coords(i_neighbor_cell, i) -
                                         mesh->cell_coords(i_cell,          i), 2);
                }
                distances.push_back(std::make_pair(i_neighbor_cell, distance));
            }
            std::sort(distances.begin(),
                      distances.end(),
                      [](auto & left, auto & right) {
                          return left.second < right.second;
                      });
            next_ring.clear();
            for (auto distance : distances) {
                next_ring.push_back(distance.first);
            }
        }
        neighbor_rings.push_back(next_ring);

        // Add the neighbors to the stencils as needed
        std::fill(stencil_grew.begin(), stencil_grew.end(), false);
        for (u_int8_t i_stencil = 0; i_stencil < n_stencils; ++i_stencil) {
            for (auto i_neighbor_cell : next_ring) {
                // Check if the stencil is full
                if (stencils[i_stencil].size() == max_cells_per_stencil) {
                    break;
                }

                // Get the transformed coordinates of the neighbor cell in the
                // subvolume's local system
                NVector dx;
                FOR_I_DIM dx[i] = mesh->h_cell_coords(i_neighbor_cell, i) -
                                  mesh->h_cell_coords(i_cell,          i);
                NVector dx_transformed;
                gemv<N_DIM>(J_sub_inverses[i_stencil].data(), dx.data(), dx_transformed.data());

                // Check if the neighbor cell is in the stencil region, and if it is, add it
                bool in_region = true;
                FOR_I_DIM if (dx_transformed[i] < 0.0) in_region = false;
                if (in_region) {
                    stencils[i_stencil].push_back(i_neighbor_cell);
                    stencil_grew[i_stencil] = true;
                }
            }
        }

        // Check if all stencils are done
        all_done = true;
        for (u_int8_t i_stencil = 0; i_stencil < n_stencils; ++i_stencil) {
            // If this face is a boundary face, it has no directional stencil,
            // and being empty is allowable so we can be done even if it is empty
            // Otherwise, if the stencil is not full, as long as the stencil grew in the last
            // iteration, there may be more neighbors to add so we are not done
            if ((mesh->cells_of_face(mesh->h_face_of_cell(i_cell, i_stencil), 1) != -1) &&
                (stencils[i_stencil].size() < max_cells_per_stencil) &&
                stencil_grew[i_stencil]) {
                all_done = false;
                break;
            }
        }
    }

    // Empty the stencils for boundary faces
    for (u_int8_t i_stencil = 0; i_stencil < n_stencils; ++i_stencil) {
        if (mesh->cells_of_face(mesh->h_face_of_cell(i_cell, i_stencil), 1) == -1) {
            stencils[i_stencil].clear();
        }
    }

    return stencils;
}

void TENO::compute_stencils_of_cell(u_int32_t i_cell,
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
    v_stencils_of_cell.push_back(v_stencils.size());
    for (auto stencil : stencils) {
        for (auto i_cell : stencil) {
            v_stencils.push_back(i_cell);
        }
        v_stencils_of_cell.push_back(v_stencils.size());
    }
    v_stencils_of_cell.pop_back();
    v_offsets_stencils_of_cell.push_back(v_stencils_of_cell.size());
}

void TENO::compute_stencils() {
    std::vector<u_int32_t> v_offsets_stencils_of_cell;
    std::vector<u_int32_t> v_stencils_of_cell;
    std::vector<u_int32_t> v_stencils;
    
    v_offsets_stencils_of_cell.push_back(0);
    for (u_int32_t i_cell = 0; i_cell < mesh->n_cells; ++i_cell) {
        compute_stencils_of_cell(i_cell,
                                 v_offsets_stencils_of_cell,
                                 v_stencils_of_cell,
                                 v_stencils);
    }
    v_offsets_stencils_of_cell.pop_back();

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

void TENO::compute_reconstruction_matrices() {
    for (u_int32_t i_cell = 0; i_cell < mesh->n_cells; ++i_cell) {
        if (mesh->n_nodes_of_cell(i_cell) != 3) {
            throw std::runtime_error("TENO has only been implemented for triangular cells.");
        }

        // Compute the transformation matrix for the target cell
        std::vector<rtype> J(N_DIM * N_DIM);
        std::vector<rtype> J_inv(N_DIM * N_DIM);
        std::vector<rtype> w0(N_DIM);
        std::vector<rtype> w1(N_DIM);
        std::vector<rtype> w2(N_DIM);
        FOR_I_DIM w0[i] = mesh->h_node_coords(mesh->h_node_of_cell(i_cell, 0), i);
        FOR_I_DIM w1[i] = mesh->h_node_coords(mesh->h_node_of_cell(i_cell, 1), i);
        FOR_I_DIM w2[i] = mesh->h_node_coords(mesh->h_node_of_cell(i_cell, 2), i);
        triangle_J(w0.data(), w1.data(), w2.data(), J.data());
        invert_matrix<N_DIM>(J.data(), J_inv.data());

        u_int8_t stencil_offset = h_offsets_stencils_of_cell(i_cell);
        u_int8_t n_stencils = h_offsets_stencils_of_cell(i_cell + 1) -
                              h_offsets_stencils_of_cell(i_cell    );
        for (u_int8_t i_stencil = 0; i_stencil < n_stencils; ++i_stencil) {
            u_int16_t stencil_size = h_stencils_of_cell(stencil_offset + i_stencil + 1) -
                                     h_stencils_of_cell(stencil_offset + i_stencil    );
            
            // Compute the reconstruction matrix for the target cell
            std::vector<rtype> A(stencil_size * n_dof);
            std::vector<rtype> Q(stencil_size * stencil_size);
            std::vector<rtype> R(stencil_size * n_dof);
            std::vector<rtype> R1(n_dof * n_dof);
            std::vector<rtype> A_pinv(n_dof * stencil_size);

            // Integrate the basis functions over each cell in the stencil
            for (u_int8_t i_neighbor = 0; i_neighbor < stencil_size; ++i_neighbor) {
                // Get the neighbor cell's vertex coordinates
                u_int32_t i_neighbor_cell = h_stencils(h_stencils_of_cell(stencil_offset + i_stencil) + i_neighbor);
                std::vector<rtype> v0(N_DIM);
                std::vector<rtype> v1(N_DIM);
                std::vector<rtype> v2(N_DIM);

                FOR_I_DIM v0[i] = mesh->h_node_coords(mesh->h_node_of_cell(i_neighbor_cell, 0), i);
                FOR_I_DIM v1[i] = mesh->h_node_coords(mesh->h_node_of_cell(i_neighbor_cell, 1), i);
                FOR_I_DIM v2[i] = mesh->h_node_coords(mesh->h_node_of_cell(i_neighbor_cell, 2), i);

                // Compute the neighbor cell's transformation matrix, to be used for mapping
                // the quadrature points to physical space
                std::vector<rtype> J_neighbor(N_DIM * N_DIM);
                triangle_J(v0.data(), v1.data(), v2.data(), J_neighbor.data());

                // Transform the neighbor vertex coordinates to the target cell's local coordinates
                std::vector<rtype> v0_trans(N_DIM);
                std::vector<rtype> v1_trans(N_DIM);
                std::vector<rtype> v2_trans(N_DIM);

                FOR_I_DIM v0_trans[i] = v0[i] - w0[i];
                FOR_I_DIM v1_trans[i] = v1[i] - w0[i];
                FOR_I_DIM v2_trans[i] = v2[i] - w0[i];

                gemv<N_DIM>(J_inv.data(), v0_trans.data(), v0_trans.data());
                gemv<N_DIM>(J_inv.data(), v1_trans.data(), v1_trans.data());
                gemv<N_DIM>(J_inv.data(), v2_trans.data(), v2_trans.data());

                rtype area_trans = triangle_area<2>(v0_trans.data(), v1_trans.data(), v2_trans.data());

                // Get all the properly transformed quadrature points
                std::vector<rtype> quad_points(quadrature.h_points.extent(0) * N_DIM);
                for (u_int16_t i_quad = 0; i_quad < quadrature.h_points.extent(0); ++i_quad) {
                    // Get the quadrature point and transform it to global coordinates
                    // using the neighbor cell's transformation matrix
                    std::vector<rtype> x(N_DIM);
                    FOR_I_DIM x[i] = quadrature.h_points(i_quad, i);
                    
                    // Transform the quadrature point into global coordinates
                    // using the neighbor cell's transformation matrix
                    gemv<N_DIM>(J_neighbor.data(), x.data(), x.data());
                    FOR_I_DIM x[i] += v0[i];

                    // Transform the quadrature point to the target cell's local coordinates
                    // using the target cell's transformation matrix
                    FOR_I_DIM x[i] -= w0[i];
                    gemv<N_DIM>(J_inv.data(), x.data(), x.data());

                    FOR_I_DIM quad_points[i_quad * N_DIM + i] = x[i];
                }

                for (u_int16_t i_dof = 0; i_dof < n_dof; ++i_dof) {
                    size_t ind = i_neighbor * n_dof + i_dof;
                    A[ind] = 0.0;
                    for (u_int16_t i_quad = 0; i_quad < quadrature.h_points.extent(0); ++i_quad) {
                        // Evaluate the basis function at the transformed point
                        rtype basis_value = basis_compute_2D(poly_indices(i_dof, 0),
                                                             poly_indices(i_dof, 1),
                                                             quad_points[i_quad * N_DIM + 0],
                                                             quad_points[i_quad * N_DIM + 1]);

                        // Add the weighted basis value the integral
                        A[ind] += quadrature.h_weights(i_quad) * basis_value;
                    }
                    A[ind] *= area_trans;
                }
            }

            // Subtract the integral of the basis function over the target cell
            for (u_int16_t i_dof = 0; i_dof < n_dof; ++i_dof) {
                size_t ind_target = i_dof;

                // Start at the last neighbor, because the first neighbor is the target cell
                // (i_neighbor needs to be signed so it can go negative and terminate)
                for (int8_t i_neighbor = stencil_size-1; i_neighbor >= 0; --i_neighbor) {
                    size_t ind = i_neighbor * n_dof + i_dof;
                    A[ind] -= A[ind_target];
                }
            }

            // Compute the Moore-Penrose pseudoinverse of the reconstruction matrix

            // Compute the QR decomposition of A
            // qr_householder(A.data(), Q.data(), R.data(), stencil_size, n_dof);

            // Now we need to solve the system R^T R A^+ = A^T
            // First, we use forward substitution to solve R^T Y = A^T
            // forward_substitution(R.data(), A.data(), Y.data(),
            //                      stencil_size, n_dof, stencil_size,
            //                      true, true);

            // Next, we use back substitution to solve R A^+ = Y
            // back_substitution(R.data(), Y.data(), A_pinv.data(),
            //                   stencil_size, n_dof, stencil_size,
            //                   false, false);
        }
    }
}

struct TENOFunctor {
    public:
        /**
         * @brief Construct a new TENOFunctor object
         * @param cells_of_face Cells of face.
         * @param face_normals Face normals.
         * @param face_solution Face solution.
         * @param solution Cell solution.
         */
        TENOFunctor(Kokkos::View<int32_t *[2]> cells_of_face,
                    Kokkos::View<rtype *[2]> face_normals,
                    Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution,
                    Kokkos::View<rtype *[N_CONSERVATIVE]> solution) :
                        cells_of_face(cells_of_face),
                        face_normals(face_normals),
                        face_solution(face_solution),
                        solution(solution) {}
        
        /**
         * @brief Overloaded operator for TENO face reconstruction.
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

void TENO::calc_face_values(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution) {
    TENOFunctor recon_functor(mesh->cells_of_face,
                              mesh->face_normals,
                              face_solution,
                              solution);
    Kokkos::parallel_for(mesh->n_faces, recon_functor);
}
