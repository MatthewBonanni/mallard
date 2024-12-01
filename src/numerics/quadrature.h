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

class Quadrature {
    public:
        /**
         * @brief Construct a new Quadrature object
         */
        Quadrature();

        /**
         * @brief Destroy the Quadrature object
         */
        ~Quadrature();

        /**
         * @brief Copy the quadrature points to the device.
         */
        void copy_host_to_device();

        Kokkos::View<rtype **> points;
        Kokkos::View<rtype **>::HostMirror h_points;
        Kokkos::View<rtype *> weights;
        Kokkos::View<rtype *>::HostMirror h_weights;
};

class TriangleCentroid : public Quadrature {
    public:
        /**
         * @brief Construct a new Centroid object
         */
        TriangleCentroid();

        /**
         * @brief Destroy the Centroid object
         */
        ~TriangleCentroid();
};

template <u_int32_t N>
class TriangleDunavant : public Quadrature {
    public:
        /**
         * @brief Construct a new TriangleDunavant object
         */
        TriangleDunavant();

        /**
         * @brief Destroy the TriangleDunavant object
         */
        ~TriangleDunavant();
};

#endif // QUADRATURE_H