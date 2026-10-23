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

/**
 * @brief The running test's own directory, shared by all ranks and unique to
 *        this process, and the working directory while the test runs. It is
 *        removed if the test passes on every rank and kept otherwise.
 */
const std::filesystem::path & scratch_dir();

/** @brief Gives every test a scratch_dir(); call once, before RUN_ALL_TESTS(). */
void install_scratch_directories();

#endif // SCRATCH_H
