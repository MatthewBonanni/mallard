/**
 * @file comm.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Process-level communication: MPI when built with Mallard_ENABLE_MPI,
 *        otherwise a single-rank stub with the same interface.
 * @version 0.3
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef COMM_H
#define COMM_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

#ifdef Mallard_HAS_MPI
#include <mpi.h>
#endif

namespace comm {

/**
 * @brief Initializes MPI on construction (unless it already is) and finalizes it
 *        on destruction if this session initialized it. Create one in main()
 *        before Kokkos is initialized, so Kokkos can map ranks to devices.
 */
class Session {
    public:
        Session(int & argc, char **& argv);
        ~Session();
        Session(const Session &) = delete;
        Session & operator=(const Session &) = delete;

    private:
        bool owns_mpi = false;
};

/** @brief Rank of this process in the world communicator. */
int rank();

/** @brief Number of processes in the world communicator. */
int size();

/** @brief Whether this is rank 0, which does all logging. */
inline bool is_root() { return rank() == 0; }

/** @brief Block until all ranks arrive. */
void barrier();

enum class Op { SUM, MIN, MAX };

/**
 * @brief In-place all-reduce over all ranks of the values in data.
 */
template <typename T>
void allreduce(std::span<T> data, Op op);

template <typename T>
T allreduce(T value, Op op) {
    allreduce(std::span<T>(&value, 1), op);
    return value;
}

template <typename T, std::size_t N>
std::array<T, N> allreduce(std::array<T, N> values, Op op) {
    allreduce(std::span<T>(values), op);
    return values;
}

#ifdef Mallard_HAS_MPI
/** @brief The world communicator. */
MPI_Comm world();
#endif

} // namespace comm

#endif // COMM_H
