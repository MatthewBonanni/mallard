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
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <fcntl.h>
#include <unistd.h>
#ifdef __APPLE__
#include <libproc.h>
#include <sys/param.h>
#endif

#include <gtest/gtest.h>

#include "comm.h"
#include "run_log.h"

namespace {

std::filesystem::path current;

std::filesystem::path make_unique_directory(std::string name) {
    auto other = [](char ch) { return !std::isalnum(static_cast<unsigned char>(ch)) && ch != '.'; };
    std::replace_if(name.begin(), name.end(), other, '_');
    const std::string pattern = (std::filesystem::temp_directory_path() / ("mallard-" + name + "-XXXXXX")).string();
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back('\0');
    if (mkdtemp(buffer.data()) == nullptr) throw std::runtime_error("cannot create a directory like " + pattern);
    return std::filesystem::path(buffer.data());
}

/** @brief Rank 0's string, on every rank. */
std::string from_root(std::string s) {
#ifdef Mallard_HAS_MPI
    int n = static_cast<int>(s.size());
    MPI_Bcast(&n, 1, MPI_INT, 0, comm::communicator());
    s.resize(static_cast<size_t>(n));
    MPI_Bcast(s.data(), n, MPI_CHAR, 0, comm::communicator());
#endif
    return s;
}

class ScratchDirectories : public testing::EmptyTestEventListener {
    public:
        void OnTestStart(const testing::TestInfo & test) override {
            current = from_root(comm::is_root() ? make_unique_directory(std::string(test.test_suite_name()) + "." + test.name()).string() : "");
            previous = std::filesystem::current_path();
            std::filesystem::current_path(current);
        }

        void OnTestEnd(const testing::TestInfo & test) override {
            std::filesystem::current_path(previous);
            // Rank 0 keeps a distributed run's log open after the run, and NFS cannot remove an open file
            run_log::close();
            const bool passed = comm::allreduce(int32_t(test.result()->Failed()), comm::Op::MAX) == 0;
            if (passed) {
                const std::string failure = remove_shared_directory(current);
                if (!failure.empty()) ADD_FAILURE() << failure;
            } else if (comm::is_root()) {
                std::cerr << "Files of " << test.test_suite_name() << "." << test.name() << " kept in "
                          << current.string() << "\n";
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

std::filesystem::path rank_scratch_dir() {
    const std::filesystem::path dir = scratch_dir() / ("rank" + std::to_string(comm::rank()));
    std::filesystem::create_directories(dir);
    return dir;
}

std::string serial_name(const std::string & name) {
    return comm::is_root() ? name : name + std::to_string(comm::rank());
}

std::vector<std::string> open_files_under(const std::filesystem::path & dir) {
    std::error_code error;
    const std::string root = std::filesystem::weakly_canonical(dir, error).string() + "/";
    if (error) return {};
    std::set<std::string> held;
    auto add = [&](const std::string & path) {
        if (path.compare(0, root.size(), root) == 0) held.insert(path);
    };
#ifdef __APPLE__
    char path[MAXPATHLEN];
    for (int fd = 0; fd < getdtablesize(); fd++) {
        if (fcntl(fd, F_GETPATH, path) != -1) add(path);
    }
    proc_regionwithpathinfo region{};
    uint64_t address = 0;
    while (proc_pidinfo(getpid(), PROC_PIDREGIONPATHINFO, address, &region, sizeof(region)) == int(sizeof(region))) {
        add(region.prp_vip.vip_path);
        address = region.prp_prinfo.pri_address + region.prp_prinfo.pri_size;
    }
#elif defined(__linux__)
    for (const auto & fd : std::filesystem::directory_iterator("/proc/self/fd", error)) {
        std::error_code unreadable;
        const auto target = std::filesystem::read_symlink(fd.path(), unreadable);
        if (!unreadable) add(target.string());
    }
    std::ifstream maps("/proc/self/maps");
    for (std::string line; std::getline(maps, line);) {
        const size_t slash = line.find('/');
        if (slash != std::string::npos) add(line.substr(slash));
    }
#endif
    return {held.begin(), held.end()};
}

std::string remove_shared_directory(const std::filesystem::path & dir) {
    std::string held;
    for (const std::string & file : open_files_under(dir)) held += (held.empty() ? "" : ", ") + file;
    const std::vector<std::string> every_rank = comm::allgather(held);
    std::string failure;
    for (size_t rank = 0; rank < every_rank.size(); rank++) {
        if (every_rank[rank].empty()) continue;
        failure += (failure.empty() ? "" : "; ") + std::string("rank ") + std::to_string(rank) + " still has open " +
                   every_rank[rank];
    }
    // Shared storage (NFS) cannot remove a file that a process on its node has open: it renames it to .nfsXXXX
    if (!failure.empty()) return "files outlive the test, so " + dir.string() + " is kept: " + failure;
    if (comm::is_root()) {
        std::error_code error;
        std::filesystem::remove_all(dir, error);
        if (error) failure = "cannot remove " + dir.string() + ": " + error.message();
    }
    return from_root(failure);
}

bool install_scratch_directories() {
    const std::filesystem::path probe = from_root(comm::is_root() ? make_unique_directory("shared").string() : "");
    std::error_code error;
    const bool seen = std::filesystem::is_directory(probe, error);
    if (!seen) {
        std::cerr << "[rank " << comm::rank() << "] " << probe.string()
                  << ", made by rank 0, is not here: the tests need a temporary directory every rank shares (set TMPDIR "
                     "to one)\n";
    }
    const bool shared = comm::allreduce(int32_t(seen), comm::Op::MIN) == 1;
    if (comm::is_root()) std::filesystem::remove(probe);
    if (shared) testing::UnitTest::GetInstance()->listeners().Append(new ScratchDirectories);
    return shared;
}
