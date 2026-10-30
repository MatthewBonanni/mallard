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
#include <iostream>
#include <string>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "comm.h"
#include "scratch.h"

namespace {

/** @brief Failures of a rank other than 0, which prints nothing else. */
class RankFailures : public ::testing::EmptyTestEventListener {
    public:
        void OnTestStart(const ::testing::TestInfo & test) override {
            name = std::string(test.test_suite_name()) + "." + test.name();
        }

        void OnTestPartResult(const ::testing::TestPartResult & result) override {
            if (!result.failed()) return;
            std::cerr << "[rank " << comm::rank() << "] " << name << " "
                      << (result.file_name() ? result.file_name() : "unknown file") << ":" << result.line_number()
                      << ": " << result.summary() << std::endl;
        }

    private:
        std::string name;
};

} // namespace

int main(int argc, char** argv) {
    comm::Session session(argc, argv);
    // Tests run in their own directories, so tests that rerun this binary need its absolute path
    std::string self = argv[0];
    if (self.find('/') != std::string::npos) self = std::filesystem::absolute(self).string();
    argv[0] = self.data();
    ::testing::InitGoogleTest(&argc, argv);
    if (!comm::is_root()) {
        // One report per run: other ranks print only their failures
        auto & listeners = ::testing::UnitTest::GetInstance()->listeners();
        delete listeners.Release(listeners.default_result_printer());
        listeners.Append(new RankFailures);
    }
    install_scratch_directories();
    Kokkos::initialize(argc, argv);
    int result = RUN_ALL_TESTS();
    Kokkos::finalize();
    return result;
}
