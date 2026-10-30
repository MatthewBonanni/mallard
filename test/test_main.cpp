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

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

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

/**
 * @brief Aborts the process when a test runs longer than the limit, naming it: a
 *        test that hangs (a collective some ranks never reach) fails in minutes.
 */
class Watchdog : public ::testing::EmptyTestEventListener {
    public:
        explicit Watchdog(const double limit) {
            // Never freed: the thread outlives the listener, which GoogleTest deletes at exit
            State * s = state;
            std::thread([s, limit] {
                while (true) {
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    std::lock_guard<std::mutex> lock(s->mutex);
                    if (!s->running || std::chrono::steady_clock::now() - s->start < std::chrono::duration<double>(limit)) {
                        continue;
                    }
                    std::cout.flush();
                    std::cerr << "[rank " << comm::rank() << "] " << s->name << " still running after " << limit
                              << " s (MALLARD_TEST_TIMEOUT): aborting" << std::endl;
                    std::abort();
                }
            }).detach();
        }

        void OnTestStart(const ::testing::TestInfo & test) override {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->name = std::string(test.test_suite_name()) + "." + test.name();
            state->start = std::chrono::steady_clock::now();
            state->running = true;
        }

        void OnTestEnd(const ::testing::TestInfo &) override {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->running = false;
        }

    private:
        struct State {
            std::mutex mutex;
            std::string name;
            std::chrono::steady_clock::time_point start;
            bool running = false;
        };
        State * state = new State;
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
    if (const char * limit = std::getenv("MALLARD_TEST_TIMEOUT")) {
        ::testing::UnitTest::GetInstance()->listeners().Append(new Watchdog(std::atof(limit)));
    }
    install_scratch_directories();
    Kokkos::initialize(argc, argv);
    int result = RUN_ALL_TESTS();
    Kokkos::finalize();
    return result;
}
