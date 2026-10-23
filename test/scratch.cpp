/**
 * @file scratch.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief A private directory for each test's files.
 * @version 0.1
 * @date 2026-10-16
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "scratch.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "comm.h"

namespace {

std::filesystem::path current;

std::filesystem::path make_unique_directory(const testing::TestInfo & test) {
    std::string name = std::string(test.test_suite_name()) + "." + test.name();
    auto other = [](char ch) { return !std::isalnum(static_cast<unsigned char>(ch)) && ch != '.'; };
    std::replace_if(name.begin(), name.end(), other, '_');
    const std::string pattern = (std::filesystem::temp_directory_path() / ("mallard-" + name + "-XXXXXX")).string();
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back('\0');
    if (mkdtemp(buffer.data()) == nullptr) throw std::runtime_error("cannot create a directory like " + pattern);
    return std::filesystem::path(buffer.data());
}

/** @brief Rank 0's path, on every rank. */
std::filesystem::path from_root(const std::filesystem::path & path) {
#ifdef Mallard_HAS_MPI
    std::string s = path.string();
    int n = static_cast<int>(s.size());
    MPI_Bcast(&n, 1, MPI_INT, 0, comm::world());
    s.resize(static_cast<size_t>(n));
    MPI_Bcast(s.data(), n, MPI_CHAR, 0, comm::world());
    return s;
#else
    return path;
#endif
}

class ScratchDirectories : public testing::EmptyTestEventListener {
    public:
        void OnTestStart(const testing::TestInfo & test) override {
            current = from_root(comm::is_root() ? make_unique_directory(test) : std::filesystem::path());
            previous = std::filesystem::current_path();
            std::filesystem::current_path(current);
        }

        void OnTestEnd(const testing::TestInfo & test) override {
            std::filesystem::current_path(previous);
            const bool passed = comm::allreduce(int32_t(test.result()->Failed()), comm::Op::MAX) == 0;
            if (comm::is_root()) {
                if (passed) {
                    std::filesystem::remove_all(current);
                } else {
                    std::cerr << "Files of " << test.test_suite_name() << "." << test.name() << " kept in "
                              << current.string() << "\n";
                }
            }
            current.clear();
        }

    private:
        std::filesystem::path previous;
};

} // namespace

const std::filesystem::path & scratch_dir() {
    if (current.empty()) throw std::logic_error("scratch_dir() is only available while a test runs");
    return current;
}

void install_scratch_directories() {
    testing::UnitTest::GetInstance()->listeners().Append(new ScratchDirectories);
}
