/**
 * @file growing_buffer.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Device memory that grows in place.
 * @version 0.1
 * @date 2026-10-05
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef GROWING_BUFFER_H
#define GROWING_BUFFER_H

#include <cstddef>
#include <vector>

/**
 * @brief Device memory at a fixed address whose backing grows as it fills:
 *        a reserved virtual address range with physical pages mapped on
 *        demand (CUDA virtual memory management). A table whose final size
 *        is known only once it is complete can then go to the device piece
 *        by piece, without being staged whole on the host or copied when it
 *        grows. Only CUDA builds have it (supported()).
 */
class GrowingBuffer {
    public:
        static bool supported();

        /** @brief Reserve addresses for up to max_bytes; nothing is mapped yet. */
        explicit GrowingBuffer(size_t max_bytes);
        ~GrowingBuffer();
        GrowingBuffer(const GrowingBuffer &) = delete;
        GrowingBuffer & operator=(const GrowingBuffer &) = delete;

        /** @brief Map pages so that the first bytes are backed by device memory. */
        void ensure(size_t bytes);

        void * data() const { return reinterpret_cast<void *>(base); }

    private:
        unsigned long long base = 0;
        [[maybe_unused]] size_t reserved = 0;
        [[maybe_unused]] size_t mapped = 0;
        [[maybe_unused]] size_t granularity = 0;
        [[maybe_unused]] int device = 0;
        std::vector<unsigned long long> handles;
};

#endif // GROWING_BUFFER_H
