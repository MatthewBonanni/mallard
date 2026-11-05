/**
 * @file memory_tracker.cpp
 * @brief Bytes held in Kokkos allocations, by memory space, from Kokkos' tool hooks.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "memory_tracker.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <Kokkos_Core.hpp>

namespace memory {

namespace {

struct Allocation {
    int space;
    uint64_t size;
    std::string label;
};

struct State {
    bool active = false;
    Usage spaces[2];  // device, host
    std::unordered_map<const void *, Allocation> live;
};

State state;

int space_index(const Kokkos::Profiling::SpaceHandle handle) {
    return std::strncmp(handle.name, "Host", 4) == 0 || std::strstr(handle.name, "HostPinned") != nullptr ? 1 : 0;
}

void on_allocate(const Kokkos::Profiling::SpaceHandle handle, const char * label, const void * ptr,
                 const uint64_t size) {
    const int s = space_index(handle);
    state.live[ptr] = {s, size, label ? label : ""};
    Usage & u = state.spaces[s];
    u.current += size;
    if (u.current > u.peak) u.peak = u.current;
}

void on_deallocate(const Kokkos::Profiling::SpaceHandle, const char *, const void * ptr, const uint64_t) {
    const auto it = state.live.find(ptr);
    if (it == state.live.end()) return;
    state.spaces[it->second.space].current -= it->second.size;
    state.live.erase(it);
}

} // namespace

Tracker::Tracker() {
    if (state.active) throw std::logic_error("memory::Tracker: one at a time.");
    state = State();
    state.active = true;
    Kokkos::Tools::Experimental::set_allocate_data_callback(on_allocate);
    Kokkos::Tools::Experimental::set_deallocate_data_callback(on_deallocate);
}

Tracker::~Tracker() {
    Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
    Kokkos::Tools::Experimental::set_deallocate_data_callback(nullptr);
    state = State();
}

Usage Tracker::device() const { return state.spaces[0]; }

Usage Tracker::host() const { return state.spaces[1]; }

std::map<std::string, uint64_t> Tracker::live(const bool device) const {
    std::map<std::string, uint64_t> bytes;
    for (const auto & [ptr, a] : state.live) {
        if ((a.space == 0) == device) bytes[a.label] += a.size;
    }
    return bytes;
}

} // namespace memory
