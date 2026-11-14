/**
 * @file sha256_hw.cpp
 * @brief SHA-256 on the ARMv8 SHA2 and x86 SHA-NI instructions, compiled for
 *        them whatever the build targets and used where the CPU has them.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "sha256_hw.h"

#include <atomic>

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__) || defined(__SYCL_DEVICE_ONLY__)
#elif defined(__aarch64__) && (defined(__APPLE__) || defined(__linux__)) && (defined(__GNUC__) || defined(__clang__))
#define MALLARD_SHA256_ARM 1
#include <arm_neon.h>
#ifdef __linux__
#include <sys/auxv.h>
#endif
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define MALLARD_SHA256_X86 1
#include <cpuid.h>
#include <immintrin.h>
#endif

namespace sha256_hw {

namespace {

#if defined(MALLARD_SHA256_ARM) || defined(MALLARD_SHA256_X86)
alignas(16) constexpr uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
#endif

#if defined(MALLARD_SHA256_ARM)

#ifdef __clang__
#define MALLARD_SHA256_TARGET __attribute__((target("sha2")))
#else
#define MALLARD_SHA256_TARGET __attribute__((target("+sha2")))
#endif

#ifdef __APPLE__
bool cpu_has_sha() { return true; }
#else
bool cpu_has_sha() { return (getauxval(AT_HWCAP) & HWCAP_SHA2) != 0; }
#endif

MALLARD_SHA256_TARGET void compress(uint32_t state[8], const uint8_t * data, size_t n) {
    uint32x4_t abcd = vld1q_u32(&state[0]);
    uint32x4_t efgh = vld1q_u32(&state[4]);
    for (; n > 0; n--, data += 64) {
        const uint32x4_t abcd_in = abcd, efgh_in = efgh;
        uint32x4_t w[4];
        for (int k = 0; k < 4; k++) w[k] = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(data + 16 * k)));
        // Four rounds at a time; the schedule's next four words replace the four just used
        for (int i = 0; i < 16; i++) {
            const uint32x4_t wk = vaddq_u32(w[i % 4], vld1q_u32(&K[4 * i]));
            const uint32x4_t previous = abcd;
            abcd = vsha256hq_u32(abcd, efgh, wk);
            efgh = vsha256h2q_u32(efgh, previous, wk);
            if (i < 12) w[i % 4] = vsha256su1q_u32(vsha256su0q_u32(w[i % 4], w[(i + 1) % 4]), w[(i + 2) % 4], w[(i + 3) % 4]);
        }
        abcd = vaddq_u32(abcd, abcd_in);
        efgh = vaddq_u32(efgh, efgh_in);
    }
    vst1q_u32(&state[0], abcd);
    vst1q_u32(&state[4], efgh);
}

#elif defined(MALLARD_SHA256_X86)

bool cpu_has_sha() {
    unsigned a = 0, b = 0, c = 0, d = 0;
    if (!__get_cpuid(1, &a, &b, &c, &d)) return false;
    const bool ssse3 = (c >> 9) & 1u, sse41 = (c >> 19) & 1u;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return false;
    return ssse3 && sse41 && ((b >> 29) & 1u);
}

__attribute__((target("sha,sse4.1,ssse3"))) void compress(uint32_t state[8], const uint8_t * data, size_t n) {
    const __m128i swap = _mm_set_epi64x(0x0c0d0e0f08090a0bLL, 0x0405060700010203LL);
    // The instructions hold the state as (a, b, e, f) and (c, d, g, h)
    __m128i dcba = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&state[0]));
    __m128i hgfe = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&state[4]));
    const __m128i cdab = _mm_shuffle_epi32(dcba, 0xB1);
    const __m128i efgh = _mm_shuffle_epi32(hgfe, 0x1B);
    __m128i abef = _mm_alignr_epi8(cdab, efgh, 8);
    __m128i cdgh = _mm_blend_epi16(efgh, cdab, 0xF0);
    for (; n > 0; n--, data += 64) {
        const __m128i abef_in = abef, cdgh_in = cdgh;
        __m128i w[4];
        for (int k = 0; k < 4; k++) {
            w[k] = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(data + 16 * k)), swap);
        }
        for (int i = 0; i < 16; i++) {
            __m128i wk = _mm_add_epi32(w[i % 4], _mm_load_si128(reinterpret_cast<const __m128i *>(&K[4 * i])));
            cdgh = _mm_sha256rnds2_epu32(cdgh, abef, wk);
            wk = _mm_shuffle_epi32(wk, 0x0E);
            abef = _mm_sha256rnds2_epu32(abef, cdgh, wk);
            if (i < 12) {
                const __m128i t = _mm_add_epi32(_mm_sha256msg1_epu32(w[i % 4], w[(i + 1) % 4]),
                                                _mm_alignr_epi8(w[(i + 3) % 4], w[(i + 2) % 4], 4));
                w[i % 4] = _mm_sha256msg2_epu32(t, w[(i + 3) % 4]);
            }
        }
        abef = _mm_add_epi32(abef, abef_in);
        cdgh = _mm_add_epi32(cdgh, cdgh_in);
    }
    const __m128i feba = _mm_shuffle_epi32(abef, 0x1B);
    const __m128i dchg = _mm_shuffle_epi32(cdgh, 0xB1);
    dcba = _mm_blend_epi16(feba, dchg, 0xF0);
    hgfe = _mm_alignr_epi8(dchg, feba, 8);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(&state[0]), dcba);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(&state[4]), hgfe);
}

#else

bool cpu_has_sha() { return false; }

void compress(uint32_t *, const uint8_t *, size_t) {}

#endif

const bool AVAILABLE = cpu_has_sha();
std::atomic<bool> on{AVAILABLE};

} // namespace

bool enabled() { return on.load(std::memory_order_relaxed); }

void set_enabled(bool enable) { on.store(enable && AVAILABLE, std::memory_order_relaxed); }

void blocks(uint32_t state[8], const uint8_t * data, size_t n) { compress(state, data, n); }

} // namespace sha256_hw
