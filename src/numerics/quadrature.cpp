/**
 * @file quadrature.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Quadrature class implementation.
 * @version 0.1
 * @date 2024-11-25
 * 
 * @copyright Copyright (c) 2024 Matthew Bonanni
 * 
 */

#include "quadrature.h"

Quadrature::Quadrature() {
    // Empty
}

void Quadrature::copy_host_to_device() {
    Kokkos::deep_copy(points, h_points);
    Kokkos::deep_copy(weights, h_weights);
}

TriangleCentroid::TriangleCentroid() {
    points = Kokkos::View<rtype **>("points", 1, 2);
    weights = Kokkos::View<rtype *>("weights", 1);

    h_points = Kokkos::create_mirror_view(points);
    h_weights = Kokkos::create_mirror_view(weights);

    h_points(0, 0) = 1.0 / 3.0;
    h_points(0, 1) = 1.0 / 3.0;

    h_weights(0) = 1.0;

    copy_host_to_device();
}

template <>
TriangleDunavant<1>::TriangleDunavant() {
    points = Kokkos::View<rtype **>("points", 1, 2);
    weights = Kokkos::View<rtype *>("weights", 1);

    h_points = Kokkos::create_mirror_view(points);
    h_weights = Kokkos::create_mirror_view(weights);

    h_points(0, 0) = 1.0 / 3.0;
    h_points(0, 1) = 1.0 / 3.0;

    h_weights(0) = 1.0;

    copy_host_to_device();
}

template <>
TriangleDunavant<2>::TriangleDunavant() {
    points = Kokkos::View<rtype **>("points", 3, 2);
    weights = Kokkos::View<rtype *>("weights", 3);

    h_points = Kokkos::create_mirror_view(points);
    h_weights = Kokkos::create_mirror_view(weights);

    h_points(0, 0) = 1.0 / 6.0;
    h_points(0, 1) = 1.0 / 6.0;
    h_points(1, 0) = 2.0 / 3.0;
    h_points(1, 1) = 1.0 / 6.0;
    h_points(2, 0) = 1.0 / 6.0;
    h_points(2, 1) = 2.0 / 3.0;

    h_weights(0) = 1.0 / 3.0;
    h_weights(1) = 1.0 / 3.0;
    h_weights(2) = 1.0 / 3.0;

    copy_host_to_device();
}

template <>
TriangleDunavant<3>::TriangleDunavant() {
    print_warning("Quadrature rule TriangleDunavant<3> has negative weights.\n"
                  "This may cause numerical issues. Use with caution.");

    points = Kokkos::View<rtype **>("points", 4, 2);
    weights = Kokkos::View<rtype *>("weights", 4);

    h_points = Kokkos::create_mirror_view(points);
    h_weights = Kokkos::create_mirror_view(weights);

    h_points(0, 0) = 1.0 / 3.0;
    h_points(0, 1) = 1.0 / 3.0;
    h_points(1, 0) = 0.6;
    h_points(1, 1) = 0.2;
    h_points(2, 0) = 0.2;
    h_points(2, 1) = 0.6;
    h_points(3, 0) = 0.2;
    h_points(3, 1) = 0.2;

    h_weights(0) = -0.5625;
    h_weights(1) = 0.52083333333333333333333333333333;
    h_weights(2) = 0.52083333333333333333333333333333;
    h_weights(3) = 0.52083333333333333333333333333333;

    copy_host_to_device();
}

template <>
TriangleDunavant<4>::TriangleDunavant() {
    points = Kokkos::View<rtype **>("points", 6, 2);
    weights = Kokkos::View<rtype *>("weights", 6);

    h_points = Kokkos::create_mirror_view(points);
    h_weights = Kokkos::create_mirror_view(weights);

    h_points(0, 0) = 0.108103018168070;
    h_points(0, 1) = 0.445948490915965;
    h_points(1, 0) = 0.445948490915965;
    h_points(1, 1) = 0.108103018168070;
    h_points(2, 0) = 0.445948490915965;
    h_points(2, 1) = 0.445948490915965;
    h_points(3, 0) = 0.816847572980459;
    h_points(3, 1) = 0.091576213509771;
    h_points(4, 0) = 0.091576213509771;
    h_points(4, 1) = 0.816847572980459;
    h_points(5, 0) = 0.091576213509771;
    h_points(5, 1) = 0.091576213509771;

    h_weights(0) = 0.223381589678011;
    h_weights(1) = 0.223381589678011;
    h_weights(2) = 0.223381589678011;
    h_weights(3) = 0.109951743655322;
    h_weights(4) = 0.109951743655322;
    h_weights(5) = 0.109951743655322;

    copy_host_to_device();
}

template <>
TriangleDunavant<5>::TriangleDunavant() {
    points = Kokkos::View<rtype **>("points", 7, 2);
    weights = Kokkos::View<rtype *>("weights", 7);

    h_points = Kokkos::create_mirror_view(points);
    h_weights = Kokkos::create_mirror_view(weights);

    h_points(0, 0) = 0.333333333333333;
    h_points(0, 1) = 0.333333333333333;
    h_points(1, 0) = 0.059715871789770;
    h_points(1, 1) = 0.470142064105115;
    h_points(2, 0) = 0.470142064105115;
    h_points(2, 1) = 0.059715871789770;
    h_points(3, 0) = 0.470142064105115;
    h_points(3, 1) = 0.470142064105115;
    h_points(4, 0) = 0.797426985353087;
    h_points(4, 1) = 0.101286507323456;
    h_points(5, 0) = 0.101286507323456;
    h_points(5, 1) = 0.797426985353087;
    h_points(6, 0) = 0.101286507323456;
    h_points(6, 1) = 0.101286507323456;

    h_weights(0) = 0.225;
    h_weights(1) = 0.132394152788506;
    h_weights(2) = 0.132394152788506;
    h_weights(3) = 0.132394152788506;
    h_weights(4) = 0.125939180544827;
    h_weights(5) = 0.125939180544827;
    h_weights(6) = 0.125939180544827;

    copy_host_to_device();
}
