/**
 * @file scratch.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief A private directory for each test's files.
 * @version 0.1
 * @date 2026-10-16
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef SCRATCH_H
#define SCRATCH_H

#include <filesystem>
#include <string>
#include <vector>

/**
 * @brief The running test's own directory, shared by all ranks and unique to
 *        this process, and the working directory while the test runs. It is
 *        removed if the test passes on every rank and kept otherwise.
 */
const std::filesystem::path & scratch_dir();

/**
 * @brief A directory in scratch_dir() of this rank's own, for the files of a
 *        test that every rank writes on its own (not those of a distributed run).
 */
std::filesystem::path rank_scratch_dir();

/**
 * @brief name on rank 0, and name followed by the rank on the others: the
 *        files of an undistributed solver, which every rank writes on its own,
 *        under names that only rank 0's copy keeps.
 */
std::string serial_name(const std::string & name);

/**
 * @brief Files under dir that this process still has open or mapped (Linux
 *        and macOS; none elsewhere).
 */
std::vector<std::string> open_files_under(const std::filesystem::path & dir);

/**
 * @brief Removes dir, which every rank shares, on rank 0 once every rank is
 *        done with it: "" on every rank, or on every rank why it could not.
 *        A file that any rank still has open fails it on every file system
 *        (shared storage could not remove it) and keeps dir.
 */
std::string remove_shared_directory(const std::filesystem::path & dir);

/**
 * @brief Gives every test a scratch_dir(); call once, before RUN_ALL_TESTS().
 *        False, with the reason on stderr, when the ranks do not share the
 *        temporary directory (TMPDIR), which the tests' files need.
 */
bool install_scratch_directories();

#endif // SCRATCH_H
