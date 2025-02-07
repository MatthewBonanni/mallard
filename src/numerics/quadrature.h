/**
 * @file quadrature.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Quadrature class declaration.
 * @version 0.1
 * @date 2024-11-25
 * 
 * @copyright Copyright (c) 2024 Matthew Bonanni
 * 
 */

#ifndef QUADRATURE_H
#define QUADRATURE_H

// https://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tri/quadrature_rules_tri.html

#include "common.h"

enum class QuadratureType {
    GAUSS_LEGENDRE,
    TRIANGLE_DUNAVANT,
};

static const std::unordered_map<std::string, QuadratureType> QUADRATURE_TYPES = {
    {"gauss_legendre", QuadratureType::GAUSS_LEGENDRE},
    {"triangle_dunavant", QuadratureType::TRIANGLE_DUNAVANT},
};

static const std::unordered_map<QuadratureType, std::string> QUADRATURE_NAMES = {
    {QuadratureType::GAUSS_LEGENDRE, "gauss_legendre"},
    {QuadratureType::TRIANGLE_DUNAVANT, "triangle_dunavant"},
};

class Quadrature {
    public:
        /**
         * @brief Construct a new Quadrature object
         */
        Quadrature();

        /**
         * @brief Destroy the Quadrature object
         */
        ~Quadrature() = default;

        /**
         * @brief Copy the quadrature points to the device.
         */
        void copy_host_to_device();

        uint8_t dim;
        uint8_t order;
        Kokkos::View<rtype **> points;
        Kokkos::View<rtype **>::HostMirror h_points;
        Kokkos::View<rtype *> weights;
        Kokkos::View<rtype *>::HostMirror h_weights;
};

class GaussLegendre : public Quadrature {
    public:
        /**
         * @brief Construct a new GaussLegendre object
         */
        GaussLegendre(uint8_t order);
};

class TriangleDunavant : public Quadrature {
    public:
        /**
         * @brief Construct a new TriangleDunavant object
         */
        TriangleDunavant(uint8_t order);
};

#endif // QUADRATURE_H