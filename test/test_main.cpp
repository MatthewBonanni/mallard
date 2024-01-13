#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

int main(int argc, char** argv) {
    // Initialize Google Test framework
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize Kokkos
    Kokkos::initialize(argc, argv);

    // Run all tests
    return RUN_ALL_TESTS();

    // Finalize Kokkos
    Kokkos::finalize();
}