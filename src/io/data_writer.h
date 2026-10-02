/**
 * @file data_writer.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Data writer class declaration.
 * @version 0.2
 * @date 2024-01-11
 *
 * @copyright Copyright (c) 2024 Matthew Bonanni
 *
 */

#ifndef DATA_WRITER_H
#define DATA_WRITER_H

#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <toml.hpp>

#include "data.h"
#include "mesh.h"

enum class DataFormat {
    VTU,
    RESTART,
};

static const std::unordered_map<std::string, DataFormat> FORMAT_TYPES = {
    {"vtu", DataFormat::VTU},
    {"restart", DataFormat::RESTART},
};

static const std::unordered_map<DataFormat, std::string> FORMAT_NAMES = {
    {DataFormat::VTU, "vtu"},
    {DataFormat::RESTART, "restart"},
};

/**
 * @brief Contents of a restart file.
 */
struct RestartData {
    uint64_t step = 0;
    double t = 0.0;
    uint64_t n_cells = 0;
    std::vector<std::vector<rtype>> conservatives;  // [variable][cell]
};

/**
 * @brief Read a restart file written by a DataWriter with format = "restart".
 */
RestartData read_restart(const std::string & filename);

/**
 * @brief Writes snapshots either every `interval` steps or every
 *        `time_interval` units of simulation time. Time-based writers also
 *        constrain the time step so that snapshots land exactly on output times.
 */
class DataWriter {
    public:
        void init(const toml::value & input,
                  std::vector<Data> & data,
                  std::shared_ptr<Mesh> mesh);

        /**
         * @brief Whether a snapshot is due at this step/time.
         */
        bool due(uint64_t step, rtype t) const;

        /**
         * @brief Write a snapshot if due (or forced).
         */
        void write(uint64_t step, rtype t, bool force = false);

        /**
         * @brief Next simulation time at which a snapshot is due
         *        (infinity for step-based writers).
         */
        rtype next_time() const;

        /**
         * @brief Continue numbering and the .pvd series of a previous run that
         *        stopped at (step, t), whose snapshot at t was already written.
         */
        void resume(uint64_t step, rtype t);

    protected:
        void write_vtu(const std::string & filename, rtype t) const;
        void write_restart(const std::string & filename, uint64_t step, rtype t) const;
        void write_pvd() const;

        std::string prefix;
        uint64_t interval = 0;
        rtype time_interval = 0.0;
        uint64_t n_written = 0;
        rtype t_last = -std::numeric_limits<rtype>::infinity();
        uint64_t step_last = std::numeric_limits<uint64_t>::max();
        DataFormat format;
        std::vector<const Data *> data_ptrs;
        std::shared_ptr<Mesh> mesh;
        std::vector<std::pair<rtype, std::string>> history;
};

#endif // DATA_WRITER_H
