/**
 * @file sha256_hw.h
 * @brief SHA-256 block compression on the CPU's SHA instructions (ARMv8 SHA2,
 *        x86 SHA-NI), where the build targets them and the CPU has them.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef SHA256_HW_H
#define SHA256_HW_H

#include <cstddef>
#include <cstdint>

namespace sha256_hw {

/** @brief Whether blocks() is used: the CPU has SHA instructions this build uses, and set_enabled has not turned them off. */
bool enabled();

/** @brief Turn the SHA instructions off (the portable compression is used) or back on where available; for tests. */
void set_enabled(bool on);

/** @brief Compress n 64-byte blocks of data into state (a..h); only where enabled(). */
void blocks(uint32_t state[8], const uint8_t * data, size_t n);

} // namespace sha256_hw

#endif // SHA256_HW_H
