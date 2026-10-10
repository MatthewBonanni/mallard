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

#include "log.h"

Quadrature::Quadrature() {
    // Empty
}

void Quadrature::copy_host_to_device() {
    Kokkos::deep_copy(points, h_points);
    Kokkos::deep_copy(weights, h_weights);
}

GaussLegendre::GaussLegendre(uint8_t order_in) {
    this->dim = 1;
    this->order = order_in;
    switch (order) {
        case 1:
            points = Kokkos::View<rtype **>("points", 1, dim);
            weights = Kokkos::View<rtype *>("weights", 1);

            h_points = Kokkos::create_mirror_view(points);
            h_weights = Kokkos::create_mirror_view(weights);

            h_points(0, 0) = 0.0;

            h_weights(0) = 2.0;
            break;
        case 2:
            points = Kokkos::View<rtype **>("points", 2, dim);
            weights = Kokkos::View<rtype *>("weights", 2);

            h_points = Kokkos::create_mirror_view(points);
            h_weights = Kokkos::create_mirror_view(weights);

            h_points(0, 0) = -0.577350269189625764509149_r;
            h_points(1, 0) = 0.577350269189625764509149_r;

            h_weights(0) = 1.0;
            h_weights(1) = 1.0;
            break;
        case 3:
            points = Kokkos::View<rtype **>("points", 3, dim);
            weights = Kokkos::View<rtype *>("weights", 3);

            h_points = Kokkos::create_mirror_view(points);
            h_weights = Kokkos::create_mirror_view(weights);

            h_points(0, 0) = -0.774596669241483377035853_r;
            h_points(1, 0) = 0.0;
            h_points(2, 0) = 0.774596669241483377035853_r;

            h_weights(0) = 0.55555555555555555555556_r;
            h_weights(1) = 0.88888888888888888888889_r;
            h_weights(2) = 0.55555555555555555555556_r;
            break;
        case 4:
            points = Kokkos::View<rtype **>("points", 4, dim);
            weights = Kokkos::View<rtype *>("weights", 4);

            h_points = Kokkos::create_mirror_view(points);
            h_weights = Kokkos::create_mirror_view(weights);

            h_points(0, 0) = -0.861136311594052575223946_r;
            h_points(1, 0) = -0.339981043584856264802666_r;
            h_points(2, 0) = 0.339981043584856264802666_r;
            h_points(3, 0) = 0.861136311594052575223946_r;

            h_weights(0) = 0.34785484513745385737306_r;
            h_weights(1) = 0.65214515486254614262694_r;
            h_weights(2) = 0.65214515486254614262694_r;
            h_weights(3) = 0.34785484513745385737306_r;
            break;
        case 5:
            points = Kokkos::View<rtype **>("points", 5, dim);
            weights = Kokkos::View<rtype *>("weights", 5);

            h_points = Kokkos::create_mirror_view(points);
            h_weights = Kokkos::create_mirror_view(weights);

            h_points(0, 0) = -0.906179845938663992797627_r;
            h_points(1, 0) = -0.538469310105683091036314_r;
            h_points(2, 0) = 0.0;
            h_points(3, 0) = 0.538469310105683091036314_r;
            h_points(4, 0) = 0.906179845938663992797627_r;

            h_weights(0) = 0.23692688505618908751426_r;
            h_weights(1) = 0.47862867049936646804129_r;
            h_weights(2) = 0.56888888888888888888889_r;
            h_weights(3) = 0.47862867049936646804129_r;
            h_weights(4) = 0.23692688505618908751426_r;
            break;
        case 6:
            points = Kokkos::View<rtype **>("points", 6, dim);
            weights = Kokkos::View<rtype *>("weights", 6);

            h_points = Kokkos::create_mirror_view(points);
            h_weights = Kokkos::create_mirror_view(weights);

            h_points(0, 0) = -0.932469514203152027812302_r;
            h_points(1, 0) = -0.661209386466264513661400_r;
            h_points(2, 0) = -0.238619186083196908630502_r;
            h_points(3, 0) = 0.238619186083196908630502_r;
            h_points(4, 0) = 0.661209386466264513661400_r;
            h_points(5, 0) = 0.932469514203152027812302_r;

            h_weights(0) = 0.17132449237917034504030_r;
            h_weights(1) = 0.36076157304813860756983_r;
            h_weights(2) = 0.46791393457269104738987_r;
            h_weights(3) = 0.46791393457269104738987_r;
            h_weights(4) = 0.36076157304813860756983_r;
            h_weights(5) = 0.17132449237917034504030_r;
            break;
        case 7:
            points = Kokkos::View<rtype **>("points", 7, dim);
            weights = Kokkos::View<rtype *>("weights", 7);

            h_points = Kokkos::create_mirror_view(points);
            h_weights = Kokkos::create_mirror_view(weights);

            h_points(0, 0) = -0.949107912342758524526190_r;
            h_points(1, 0) = -0.741531185599394439863865_r;
            h_points(2, 0) = -0.405845151377397166906607_r;
            h_points(3, 0) = 0.0;
            h_points(4, 0) = 0.405845151377397166906607_r;
            h_points(5, 0) = 0.741531185599394439863865_r;
            h_points(6, 0) = 0.949107912342758524526190_r;

            h_weights(0) = 0.12948496616886969327061_r;
            h_weights(1) = 0.27970539148927666790147_r;
            h_weights(2) = 0.38183005050511894495037_r;
            h_weights(3) = 0.41795918367346938775510_r;
            h_weights(4) = 0.38183005050511894495037_r;
            h_weights(5) = 0.27970539148927666790147_r;
            h_weights(6) = 0.12948496616886969327061_r;
            break;
        default:
            throw std::runtime_error("Gauss-Legendre quadrature rule of order " + std::to_string(order) + " not implemented.");
    }

    copy_host_to_device();
}

TriangleDunavant::TriangleDunavant(uint8_t order_in) {
    this->dim = 2;
    this->order = order_in;
    switch (order) {
        case 1:
            points = Kokkos::View<rtype **>("points", 1, dim);
            weights = Kokkos::View<rtype *>("weights", 1);

            h_points = Kokkos::create_mirror_view(points);
            h_weights = Kokkos::create_mirror_view(weights);

            h_points(0, 0) = 1.0_r / 3.0_r;
            h_points(0, 1) = 1.0_r / 3.0_r;

            h_weights(0) = 1.0;
            break;
        case 2:
            points = Kokkos::View<rtype **>("points", 3, dim);
            weights = Kokkos::View<rtype *>("weights", 3);

            h_points = Kokkos::create_mirror_view(points);
            h_weights = Kokkos::create_mirror_view(weights);

            h_points(0, 0) = 1.0_r / 6.0_r;
            h_points(0, 1) = 1.0_r / 6.0_r;
            h_points(1, 0) = 2.0_r / 3.0_r;
            h_points(1, 1) = 1.0_r / 6.0_r;
            h_points(2, 0) = 1.0_r / 6.0_r;
            h_points(2, 1) = 2.0_r / 3.0_r;

            h_weights(0) = 1.0_r / 3.0_r;
            h_weights(1) = 1.0_r / 3.0_r;
            h_weights(2) = 1.0_r / 3.0_r;
            break;
        case 3:
            logging::warning("quadrature rule TriangleDunavant<3> has negative weights; use with caution.");

            points = Kokkos::View<rtype **>("points", 4, dim);
            weights = Kokkos::View<rtype *>("weights", 4);

            h_points = Kokkos::create_mirror_view(points);
            h_weights = Kokkos::create_mirror_view(weights);

            h_points(0, 0) = 1.0_r / 3.0_r;
            h_points(0, 1) = 1.0_r / 3.0_r;
            h_points(1, 0) = 0.6_r;
            h_points(1, 1) = 0.2_r;
            h_points(2, 0) = 0.2_r;
            h_points(2, 1) = 0.6_r;
            h_points(3, 0) = 0.2_r;
            h_points(3, 1) = 0.2_r;

            h_weights(0) = -0.5625;
            h_weights(1) = 0.52083333333333333333333333333333_r;
            h_weights(2) = 0.52083333333333333333333333333333_r;
            h_weights(3) = 0.52083333333333333333333333333333_r;
            break;
        case 4:
            points = Kokkos::View<rtype **>("points", 6, dim);
            weights = Kokkos::View<rtype *>("weights", 6);

            h_points = Kokkos::create_mirror_view(points);
            h_weights = Kokkos::create_mirror_view(weights);

            h_points(0, 0) = 0.108103018168070_r;
            h_points(0, 1) = 0.445948490915965_r;
            h_points(1, 0) = 0.445948490915965_r;
            h_points(1, 1) = 0.108103018168070_r;
            h_points(2, 0) = 0.445948490915965_r;
            h_points(2, 1) = 0.445948490915965_r;
            h_points(3, 0) = 0.816847572980459_r;
            h_points(3, 1) = 0.091576213509771_r;
            h_points(4, 0) = 0.091576213509771_r;
            h_points(4, 1) = 0.816847572980459_r;
            h_points(5, 0) = 0.091576213509771_r;
            h_points(5, 1) = 0.091576213509771_r;

            h_weights(0) = 0.223381589678011_r;
            h_weights(1) = 0.223381589678011_r;
            h_weights(2) = 0.223381589678011_r;
            h_weights(3) = 0.109951743655322_r;
            h_weights(4) = 0.109951743655322_r;
            h_weights(5) = 0.109951743655322_r;
            break;
        case 5:
            points = Kokkos::View<rtype **>("points", 7, dim);
            weights = Kokkos::View<rtype *>("weights", 7);

            h_points = Kokkos::create_mirror_view(points);
            h_weights = Kokkos::create_mirror_view(weights);

            h_points(0, 0) = 0.333333333333333_r;
            h_points(0, 1) = 0.333333333333333_r;
            h_points(1, 0) = 0.059715871789770_r;
            h_points(1, 1) = 0.470142064105115_r;
            h_points(2, 0) = 0.470142064105115_r;
            h_points(2, 1) = 0.059715871789770_r;
            h_points(3, 0) = 0.470142064105115_r;
            h_points(3, 1) = 0.470142064105115_r;
            h_points(4, 0) = 0.797426985353087_r;
            h_points(4, 1) = 0.101286507323456_r;
            h_points(5, 0) = 0.101286507323456_r;
            h_points(5, 1) = 0.797426985353087_r;
            h_points(6, 0) = 0.101286507323456_r;
            h_points(6, 1) = 0.101286507323456_r;

            h_weights(0) = 0.225000000000000_r;
            h_weights(1) = 0.132394152788506_r;
            h_weights(2) = 0.132394152788506_r;
            h_weights(3) = 0.132394152788506_r;
            h_weights(4) = 0.125939180544827_r;
            h_weights(5) = 0.125939180544827_r;
            h_weights(6) = 0.125939180544827_r;
            break;
        default:
            throw std::runtime_error("Triangle Dunavant quadrature rule of order " + std::to_string(order) + " not implemented.");
    }

    copy_host_to_device();
}
