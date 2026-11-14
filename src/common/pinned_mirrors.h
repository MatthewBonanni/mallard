/**
 * @file pinned_mirrors.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Host copies of device arrays that the device can write, in step
 *        graphs: snapshots and restarts staged without stopping the device.
 * @version 0.1
 * @date 2026-11-13
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef PINNED_MIRRORS_H
#define PINNED_MIRRORS_H

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

/**
 * @brief Owns host mirrors in pinned memory, never the device's own arrays
 *        (also off GPUs, where create_mirror_view would return them). The
 *        mirrors are of the usual host mirror type and live as long as this.
 */
class PinnedMirrors {
    public:
        template <class V>
        typename V::host_mirror_type operator()(const V & v) {
            using T = typename V::non_const_value_type;
            const size_t bytes = v.span() * sizeof(T);
            Kokkos::View<char *, Kokkos::SharedHostPinnedSpace> block(std::string(v.label()) + "_pinned", bytes);
            blocks.push_back(block);
            return typename V::host_mirror_type(reinterpret_cast<T *>(block.data()), v.layout());
        }

        /** @brief Whether bytes from p lie in one mirror. */
        bool holds(const void * p, const size_t bytes) const {
            const char * c = static_cast<const char *>(p);
            for (const auto & b : blocks) {
                if (c >= b.data() && c + bytes <= b.data() + b.span()) return true;
            }
            return false;
        }

    private:
        std::vector<Kokkos::View<char *, Kokkos::SharedHostPinnedSpace>> blocks;
};

/**
 * @brief Enqueues a copy of a device array into its pinned mirror: a kernel,
 *        so that step graphs capture it (also under conditional nodes).
 */
template <class M, class V>
void copy_to_pinned(const PinnedMirrors & mirrors, const M & mirror, const V & v) {
    using T = typename V::non_const_value_type;
    const size_t n = v.span();
    if (n == 0) return;
    if (mirror.span() != n || !v.span_is_contiguous() || !mirrors.holds(mirror.data(), n * sizeof(T))) {
        throw std::logic_error(std::string("copy_to_pinned: ") + v.label() + " has no pinned mirror");
    }
    T * const dst = mirror.data();
    const T * const src = v.data();
    Kokkos::parallel_for("copy_to_pinned", n, KOKKOS_LAMBDA(const size_t i) { dst[i] = src[i]; });
}

#endif // PINNED_MIRRORS_H
