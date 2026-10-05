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
#include "global_order.h"
#include "mesh.h"

enum class DataFormat {
    VTU,
    RESTART,
    HDF5,
};

static const std::unordered_map<std::string, DataFormat> FORMAT_TYPES = {
    {"vtu", DataFormat::VTU},
    {"restart", DataFormat::RESTART},
    {"hdf5", DataFormat::HDF5},
};

static const std::unordered_map<DataFormat, std::string> FORMAT_NAMES = {
    {DataFormat::VTU, "vtu"},
    {DataFormat::RESTART, "restart"},
    {DataFormat::HDF5, "hdf5"},
};

/** @brief Named scalars of a restart file (version 3), e.g. the statistics' weight. */
using RestartAttributes = std::vector<std::pair<std::string, double>>;

/** @brief Upper bound on the faces of a cell, for the keys of RestartFaces. */
constexpr uint64_t RESTART_FACES_PER_CELL = 8;

/**
 * @brief Face records of a restart file (version 4): the [p, u_n] of the
 *        characteristic boundary faces (docs/design/nscbc.md). A face's key is
 *        RESTART_FACES_PER_CELL times the global id of its cell, plus its local
 *        face in that cell, so that it does not depend on the rank count.
 */
struct RestartFaces {
    static constexpr uint32_t WIDTH = 2;
    std::vector<uint64_t> keys;  // [face]
    std::vector<rtype> values;   // [face][WIDTH]
};

/**
 * @brief Contents of a restart file.
 */
struct RestartData {
    uint64_t step = 0;
    double t = 0.0;
    uint64_t n_cells = 0;                    // in the file
    std::vector<std::string> names;          // [variable]
    std::vector<std::vector<rtype>> fields;  // [variable][cell read]
    RestartAttributes attributes;
    RestartFaces faces;                      // every face of the file, in increasing key

    /** @brief Values of the named variable, or nullptr if the file has none. */
    const std::vector<rtype> * find(const std::string & name) const;

    /** @brief Value of the named attribute, or nullptr if the file has none. */
    const double * attribute(const std::string & name) const;

    /** @brief The RestartFaces::WIDTH values of the face with this key, or nullptr if the file has none. */
    const rtype * face(uint64_t key) const;
};

/**
 * @brief Read a restart file written by a DataWriter with format = "restart":
 *        every cell, or with cells, only those global cells (fields then hold
 *        their values in the order of cells).
 *
 * Version 2 files list their variable names, version 3 files also named
 * scalar attributes, version 4 files also face records; version 1 files hold
 * the flow block CONSERVATIVE_NAMES only. Either must contain the flow block of
 * this build's dimension.
 */
RestartData read_restart(const std::string & filename, const std::vector<uint64_t> * cells = nullptr);

/**
 * @brief Writes snapshots either every `interval` steps or every
 *        `time_interval` units of simulation time. Time-based writers also
 *        constrain the time step so that snapshots land exactly on output times.
 */
class DataWriter {
    public:
        /**
         * @brief Set up from a [[write_data]] entry.
         * @param restart_variables Variables of a restart file, in order
         *        (default: CONSERVATIVE_NAMES); ignored for other formats.
         */
        void init(const toml::value & input,
                  std::vector<Data> & data,
                  std::shared_ptr<Mesh> mesh,
                  const std::vector<std::string> & restart_variables = {});

        /**
         * @brief Whether a snapshot is due at this step/time.
         */
        bool due(uint64_t step, rtype t) const;

        /**
         * @brief Write a snapshot if due (or forced); restart files also store
         *        the attributes and the faces (of the owned cells; collective).
         */
        void write(uint64_t step, rtype t, bool force = false, const RestartAttributes & attributes = {},
                   const RestartFaces & faces = {});

        /**
         * @brief Next simulation time at which a snapshot is due
         *        (infinity for step-based writers).
         */
        rtype next_time() const;

        /**
         * @brief Continue numbering and the .pvd (.xmf) series of a previous
         *        run that stopped at (step, t), whose snapshot at t was already written.
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
        void write_restart(const std::string & filename, uint64_t step, rtype t,
                           const RestartAttributes & attributes, const RestartFaces & faces) const;
        void write_pvd() const;

        /**
         * @brief HDF5 output: the mesh once (<prefix>_mesh.h5), then per
         *        snapshot a file of cell fields, every array in global-id
         *        order, and the XDMF indexes that let ParaView read them.
         */
        void write_hdf5_mesh();
        void write_hdf5(const std::string & stem, uint64_t step, rtype t) const;
        void write_xdmf(const std::string & stem) const;
        std::string xdmf_grid(const std::string & stem, double t, const std::string & indent) const;
        void resume_xdmf(rtype t);

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
        void write_restart_distributed(const std::string & filename, uint64_t step, rtype t,
                                       const RestartAttributes & attributes, const RestartFaces & faces) const;
        std::vector<uint32_t> geometry_faces;  // Empty: write all cells
        // HDF5 output: owned cells in global order, and the sizes of the mesh written
        GlobalOrder cell_order;
        bool hdf5_mesh_written = false;
        uint64_t hdf5_n_nodes = 0, hdf5_topology_size = 0;
};

#endif // DATA_WRITER_H
