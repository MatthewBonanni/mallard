/**
 * @file boundary.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Boundary class declaration.
 * @version 0.1
 * @date 2023-12-20
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#ifndef BOUNDARY_H
#define BOUNDARY_H

#include <toml.hpp>

#include "common.h"
#include "zone.h"
#include "mesh.h"
#include "physics.h"
#include "riemann_solver.h"

enum class BoundaryType {
    SYMMETRY,
    EXTRAPOLATION,
    WALL_ADIABATIC,
    UPT,
    P_OUT,
};

static const std::unordered_map<std::string, BoundaryType> BOUNDARY_TYPES = {
    {"symmetry", BoundaryType::SYMMETRY},
    {"extrapolation", BoundaryType::EXTRAPOLATION},
    {"wall_adiabatic", BoundaryType::WALL_ADIABATIC},
    {"upt", BoundaryType::UPT},
    {"p_out", BoundaryType::P_OUT}
};

static const std::unordered_map<BoundaryType, std::string> BOUNDARY_NAMES = {
    {BoundaryType::SYMMETRY, "symmetry"},
    {BoundaryType::EXTRAPOLATION, "extrapolation"},
    {BoundaryType::WALL_ADIABATIC, "wall_adiabatic"},
    {BoundaryType::UPT, "upt"},
    {BoundaryType::P_OUT, "p_out"}
};

class Boundary {
    public:
        /**
         * @brief Construct a new Boundary object
         */
        Boundary();

        /**
         * @brief Destroy the Boundary object
         */
        virtual ~Boundary();

        /**
         * @brief Set the zone.
         * @param zone Pointer to the zone.
         */
        void set_zone(FaceZone * zone);

        /**
         * @brief Set the mesh.
         * @param mesh Pointer to the mesh.
         */
        void set_mesh(std::shared_ptr<Mesh> mesh);

        /**
         * @brief Set the face quadrature weights.
         * @param face_quad_weights Face quadrature weights.
         */
        void set_face_quad_weights(Kokkos::View<rtype *> face_quad_weights);

        /**
         * @brief Set the physics.
         * @param physics Pointer to the physics.
         */
        void set_physics(std::shared_ptr<PhysicsWrapper> physics);

        /**
         * @brief Set the Riemann solver.
         * @param riemann_solver Pointer to the Riemann solver.
         */
        void set_riemann_solver(std::shared_ptr<RiemannSolver> riemann_solver);

        /**
         * @brief Print the boundary.
         */
        virtual void print();

        /**
         * @brief Initialize the boundary.
         * @param input TOML input parameter table.
         */
        virtual void init(const toml::value & input);

        /**
         * @brief Copy data from the host to the device.
         */
        virtual void copy_host_to_device();

        /**
         * @brief Copy data from the device to the host.
         */
        virtual void copy_device_to_host();

        /**
         * @brief Compute and apply the boundary flux.
         * @param face_solution Pointer to the face solution.
         * @param rhs Pointer to the right hand side.
         */
        virtual void apply(Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                           Kokkos::View<rtype *[N_CONSERVATIVE]> rhs) = 0;
        
    protected:
        FaceZone * zone;
        std::shared_ptr<Mesh> mesh;
        Kokkos::View<rtype *> face_quad_weights;
        BoundaryType type;
        std::shared_ptr<PhysicsWrapper> physics;
        std::shared_ptr<RiemannSolver> riemann_solver;
    private:
};

#endif // BOUNDARY_H