/**
 * @file step_graph_guard.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Records, with Kokkos Tools callbacks, what would break the capture of
 *        a step's CUDA graph on any backend.
 * @version 0.1
 * @date 2026-11-10
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef STEP_GRAPH_GUARD_H
#define STEP_GRAPH_GUARD_H

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ProfileSection.hpp>

/**
 * @brief While alive, records the fences, allocations and copies between host
 *        and device memory issued inside the profiling section
 *        "step_graph_capture" (what a capture records, Solver::graph_step) but
 *        outside "host_staged" (comm::HostStaged: round trips the graph makes
 *        with NCCL or CUDA events instead). One guard at a time.
 */
class StepGraphGuard {
    public:
        StepGraphGuard() : saved(Kokkos::Tools::Experimental::get_callbacks()) {
            state() = State{};
#ifdef KOKKOS_ENABLE_THREADS
            // Each resize of the Threads backend's scratch sets both its reduction and team parts, so alternating
            // reductions and team kernels reallocate it at every step; reserved here, only growth past it is seen
            Kokkos::Impl::ThreadsInternal::resize_scratch(size_t(1) << 16, size_t(1) << 20);
#endif
            using namespace Kokkos::Tools::Experimental;
            set_create_profile_section_callback(&create_section);
            set_start_profile_section_callback(&start_section);
            set_stop_profile_section_callback(&stop_section);
            set_begin_fence_callback(&begin_fence);
            set_allocate_data_callback(&allocate);
            set_begin_deep_copy_callback(&begin_deep_copy);
#ifndef KOKKOS_ENABLE_CUDA
            // Kernel callbacks make Kokkos fence around each kernel, which a capture would not survive
            set_begin_parallel_for_callback(&begin_kernel);
            set_begin_parallel_reduce_callback(&begin_kernel);
            set_begin_parallel_scan_callback(&begin_kernel);
#endif
        }
        ~StepGraphGuard() { Kokkos::Tools::Experimental::set_callbacks(saved); }
        StepGraphGuard(const StepGraphGuard &) = delete;
        StepGraphGuard & operator=(const StepGraphGuard &) = delete;

        /** @brief What would break the capture, each once (with the kernel before it off CUDA), in order. */
        const std::vector<std::string> & violations() const { return state().violations; }
        /** @brief Sections "step_graph_capture" entered. */
        uint32_t captures() const { return state().captures; }

    private:
        struct State {
            std::map<uint32_t, std::string> names;
            uint32_t next_id = 0, capture_depth = 0, staged_depth = 0, captures = 0;
            std::vector<std::string> violations;
            std::string kernel;
        };
        static State & state() {
            static State s;
            return s;
        }
        static bool armed() { return state().capture_depth > 0 && state().staged_depth == 0; }
        static void record(std::string what) {
            if (!armed()) return;
            if (!state().kernel.empty()) what += " after kernel \"" + state().kernel + "\"";
            std::vector<std::string> & v = state().violations;
            if (std::find(v.begin(), v.end(), what) == v.end()) v.push_back(what);
        }
        static void begin_kernel(const char * name, uint32_t, uint64_t * id) {
            *id = 0;
            state().kernel = name;
        }

        static void create_section(const char * name, uint32_t * id) {
            *id = state().next_id++;
            state().names[*id] = name;
        }
        static void start_section(const uint32_t id) {
            const std::string & name = state().names[id];
            if (name == "step_graph_capture") {
                state().capture_depth++;
                state().captures++;
            }
            if (name == "host_staged") state().staged_depth++;
        }
        static void stop_section(const uint32_t id) {
            const std::string & name = state().names[id];
            if (name == "step_graph_capture") state().capture_depth--;
            if (name == "host_staged") state().staged_depth--;
        }
        static void begin_fence(const char * name, uint32_t, uint64_t * handle) {
            *handle = 0;
            // The Threads backend's wait at the end of each kernel, and the fences of the kernel callbacks
            if (std::strncmp(name, "Kokkos::ThreadsInternal::fence", 30) == 0) return;
            if (std::strstr(name, "Kokkos Profile Tool Fence")) return;
            record(std::string("fence \"") + name + "\"");
        }
        static void allocate(const Kokkos::Profiling::SpaceHandle space, const char * label, const void *,
                             const uint64_t size) {
            // Kokkos sizes a team kernel's level-1 scratch at every launch until it is not empty: 0 bytes is no allocation
            if (size == 0) return;
            record(std::string("allocation \"") + label + "\" in " + space.name);
        }
        static void begin_deep_copy(const Kokkos::Profiling::SpaceHandle dst_space, const char * dst_label, const void *,
                                    const Kokkos::Profiling::SpaceHandle src_space, const char * src_label, const void *,
                                    uint64_t) {
            // Device builds only: pageable host memory cannot take part in a capture
            const bool dst_host = std::strcmp(dst_space.name, "Host") == 0;
            const bool src_host = std::strcmp(src_space.name, "Host") == 0;
            // deep_copy(exec, view, value) fills on the execution space (a memset or a kernel on its stream), and
            // Kokkos reports the value as a host source "(none)"; fills that must fence are reported as fences
            const bool fill = src_host && std::strcmp(src_label, "(none)") == 0;
            if (dst_host != src_host && !fill) {
                record(std::string("copy \"") + src_label + "\" (" + src_space.name + ") to \"" + dst_label + "\" (" +
                       dst_space.name + ")");
            }
        }

        Kokkos::Tools::Experimental::EventSet saved;
};

#endif // STEP_GRAPH_GUARD_H
