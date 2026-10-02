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
    uint64_t n_cells = 0;                           // in the file
    std::vector<std::vector<rtype>> conservatives;  // [variable][cell read]
};

/**
 * @brief Read a restart file written by a DataWriter with format = "restart":
 *        every cell, or with cells, only those global cells (conservatives
 *        then hold their values in the order of cells).
 */
RestartData read_restart(const std::string & filename, const std::vector<uint64_t> * cells = nullptr);

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

        /**
         * @brief Format and the output key-value line for the run log.
         */
        std::pair<std::string, std::string> summary() const;

        DataFormat get_format() const { return format; }
        const std::string & get_prefix() const { return prefix; }
        uint64_t files_written() const { return n_files; }

    protected:
        void write_vtu(const std::string & filename, rtype t) const;
        void write_vtu_faces(const std::string & filename, rtype t) const;
        void write_restart(const std::string & filename, uint64_t step, rtype t) const;
        void write_pvd() const;

        /**
         * @brief An output array: one scalar, or the N_DIM components of a
         *        vector (written as 3 components, zero-padded in 2D).
         */
        struct Field {
            std::string name;
            std::vector<const Data *> components;
            uint32_t n_vtk_components() const { return components.size() == 1 ? 1 : 3; }
            rtype value(uint32_t i_cell, uint32_t i_comp) const {
                return i_comp < components.size() ? (*components[i_comp])[i_cell] : rtype(0);
            }
        };

        std::string prefix;
        uint64_t interval = 0;
        rtype time_interval = 0.0;
        uint64_t n_written = 0;
        rtype t_last = -std::numeric_limits<rtype>::infinity();
        uint64_t step_last = std::numeric_limits<uint64_t>::max();
        DataFormat format;
        std::vector<Field> fields;
        std::shared_ptr<Mesh> mesh;
        std::vector<std::pair<rtype, std::string>> history;
        bool surface = false;
        std::string geometry = "all";
        uint64_t n_files = 0;
        void write_pvtu(const std::string & filename, const std::string & stem) const;
        void write_restart_distributed(const std::string & filename, uint64_t step, rtype t) const;
        std::vector<uint32_t> geometry_faces;  // Empty: write all cells
};

#endif // DATA_WRITER_H
