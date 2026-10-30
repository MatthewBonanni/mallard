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

#include <filesystem>
#include <string>
#include <string_view>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "comm.h"
#include "scratch.h"

int main(int argc, char** argv) {
    // The help page starts neither MPI nor the device: MPI without its launcher (a singleton) can hang
    for (int i = 1; i < argc; i++) {
        const std::string_view arg = argv[i];
        if (arg == "--help" || arg == "--gtest_help") {
            ::testing::InitGoogleTest(&argc, argv);
            return 0;
        }
    }
    comm::Session session(argc, argv);
    // Tests run in their own directories, so tests that rerun this binary need its absolute path
    std::string self = argv[0];
    if (self.find('/') != std::string::npos) self = std::filesystem::absolute(self).string();
    argv[0] = self.data();
    ::testing::InitGoogleTest(&argc, argv);
    if (!comm::is_root()) {
        // One report per run: other ranks stay quiet
        auto & listeners = ::testing::UnitTest::GetInstance()->listeners();
        delete listeners.Release(listeners.default_result_printer());
    }
    install_scratch_directories();
    Kokkos::initialize(argc, argv);
    int result = RUN_ALL_TESTS();
    Kokkos::finalize();
    return result;
}
