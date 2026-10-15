/**
 * @file compression.h
 * @brief Output compression: zlib streams (optionally byte-shuffled) and
 *        error-bounded rounding of floating-point values.
 * @version 0.9
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef COMPRESSION_H
#define COMPRESSION_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace compression {

/** @brief Whether this build compresses with zlib (Mallard_ENABLE_ZLIB). */
bool have_zlib();

/**
 * @brief zlib stream of n_bytes at data. With width > 1 the bytes are first
 *        shuffled (byte k of every width-byte item together), which groups
 *        the exponents and leading mantissa bytes of floating-point values.
 */
std::vector<char> deflate(const void * data, size_t n_bytes, size_t width = 1, int level = 1);

/** @brief Inverse of deflate: exactly n_out bytes into out. */
void inflate(const char * in, size_t n_in, void * out, size_t n_out, size_t width = 1);

/**
 * @brief Round each finite value to the fewest mantissa bits that keep it
 *        within relative_error of itself (|x' - x| <= relative_error |x|),
 *        rounding to nearest even; zeros, subnormals, infinities and NaNs
 *        are unchanged.
 *        The dropped bits become zeros, which compress well.
 */
void round_to_relative_error(std::span<double> values, double relative_error);
void round_to_relative_error(std::span<float> values, double relative_error);

/** @brief Run task(0) ... task(n - 1) on up to n_threads threads. */
void parallel_for(size_t n, unsigned n_threads, const std::function<void(size_t)> & task);

} // namespace compression

#endif // COMPRESSION_H
