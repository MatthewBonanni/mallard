/**
 * @file test_main.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Test entry point.
 * @version 0.2
 * @date 2024-01-17
 *
 * @copyright Copyright (c) 2024 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "comm.h"

int main(int argc, char** argv) {
    comm::Session session(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    if (!comm::is_root()) {
        // One report per run: other ranks stay quiet
        auto & listeners = ::testing::UnitTest::GetInstance()->listeners();
        delete listeners.Release(listeners.default_result_printer());
    }
    Kokkos::initialize(argc, argv);
    int result = RUN_ALL_TESTS();
    Kokkos::finalize();
    return result;
}
