/**
 * @file compression.cpp
 * @brief Output compression: zlib streams and error-bounded rounding.
 * @version 0.9
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "compression.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#ifdef Mallard_HAS_ZLIB
#include <zlib.h>
#endif

namespace compression {

namespace {

void shuffle(const char * in, char * out, size_t n_bytes, size_t width) {
    const size_t n = n_bytes / width;
    for (size_t i = 0; i < n; i++) {
        for (size_t b = 0; b < width; b++) out[b * n + i] = in[i * width + b];
    }
    std::memcpy(out + n * width, in + n * width, n_bytes - n * width);
}

void unshuffle(const char * in, char * out, size_t n_bytes, size_t width) {
    const size_t n = n_bytes / width;
    for (size_t i = 0; i < n; i++) {
        for (size_t b = 0; b < width; b++) out[i * width + b] = in[b * n + i];
    }
    std::memcpy(out + n * width, in + n * width, n_bytes - n * width);
}

template <typename Float, typename Bits, int MANTISSA>
void round_values(std::span<Float> values, double relative_error) {
    if (!(relative_error > 0.0 && relative_error < 1.0)) {
        throw std::invalid_argument("compression: relative error must be in (0, 1).");
    }
    // Keeping k mantissa bits errs by at most 2^-(k + 1) relative to the value
    const int keep = std::clamp(static_cast<int>(std::ceil(-std::log2(relative_error))) - 1, 0, MANTISSA);
    const int drop = MANTISSA - keep;
    if (drop == 0) return;
    const Bits half = Bits(1) << (drop - 1);
    const Bits mask = ~((Bits(1) << drop) - 1);
    for (Float & x : values) {
        if (std::fpclassify(x) != FP_NORMAL) continue;
        Bits bits;
        std::memcpy(&bits, &x, sizeof(bits));
        const Bits rounded = (bits + half - 1 + ((bits >> drop) & 1)) & mask;
        Float y;
        std::memcpy(&y, &rounded, sizeof(y));
        // Rounding up the largest finite values would overflow
        if (std::isfinite(y)) x = y;
    }
}

} // namespace

#ifdef Mallard_HAS_ZLIB

bool have_zlib() { return true; }

std::vector<char> deflate(const void * data, size_t n_bytes, size_t width, int level) {
    const char * source = static_cast<const char *>(data);
    std::vector<char> shuffled;
    if (width > 1) {
        shuffled.resize(n_bytes);
        shuffle(source, shuffled.data(), n_bytes, width);
        source = shuffled.data();
    }
    uLongf n_out = compressBound(static_cast<uLong>(n_bytes));
    std::vector<char> out(n_out);
    if (compress2(reinterpret_cast<Bytef *>(out.data()), &n_out, reinterpret_cast<const Bytef *>(source),
                  static_cast<uLong>(n_bytes), level) != Z_OK) {
        throw std::runtime_error("compression: zlib failed.");
    }
    out.resize(n_out);
    return out;
}

void inflate(const char * in, size_t n_in, void * out, size_t n_out, size_t width) {
    std::vector<char> shuffled(width > 1 ? n_out : 0);
    char * target = width > 1 ? shuffled.data() : static_cast<char *>(out);
    uLongf n = static_cast<uLongf>(n_out);
    if (uncompress(reinterpret_cast<Bytef *>(target), &n, reinterpret_cast<const Bytef *>(in),
                   static_cast<uLong>(n_in)) != Z_OK ||
        n != n_out) {
        throw std::runtime_error("compression: corrupt zlib stream.");
    }
    if (width > 1) unshuffle(shuffled.data(), static_cast<char *>(out), n_out, width);
}

#else

bool have_zlib() { return false; }

std::vector<char> deflate(const void *, size_t, size_t, int) {
    throw std::runtime_error("compression: this build has no zlib (configure with Mallard_ENABLE_ZLIB=ON).");
}

void inflate(const char *, size_t, void *, size_t, size_t) {
    throw std::runtime_error("compression: this build has no zlib (configure with Mallard_ENABLE_ZLIB=ON).");
}

#endif

void round_to_relative_error(std::span<double> values, double relative_error) {
    static_assert(std::numeric_limits<double>::digits == 53);
    round_values<double, uint64_t, 52>(values, relative_error);
}

void round_to_relative_error(std::span<float> values, double relative_error) {
    static_assert(std::numeric_limits<float>::digits == 24);
    round_values<float, uint32_t, 23>(values, relative_error);
}

void parallel_for(size_t n, unsigned n_threads, const std::function<void(size_t)> & task) {
    const size_t n_workers = std::min<size_t>(std::max(n_threads, 1u), n);
    if (n_workers <= 1) {
        for (size_t i = 0; i < n; i++) task(i);
        return;
    }
    std::atomic<size_t> next{0};
    std::exception_ptr error;
    std::mutex error_mutex;
    auto work = [&]() {
        for (size_t i; (i = next.fetch_add(1)) < n;) {
            try {
                task(i);
            } catch (...) {
                const std::lock_guard<std::mutex> lock(error_mutex);
                if (!error) error = std::current_exception();
            }
        }
    };
    std::vector<std::thread> workers;
    for (size_t w = 1; w < n_workers; w++) workers.emplace_back(work);
    work();
    for (auto & worker : workers) worker.join();
    if (error) std::rethrow_exception(error);
}

} // namespace compression
